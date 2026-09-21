"""Document management use case: upload, list, and delete indexed documents.

Encapsulates the multipart -> temp dir -> persist original to blob -> ingest /
chunk / embed -> index -> record in Cosmos pipeline so presentation routes stay
thin and the flow can be reused from other entrypoints.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import shutil
import tempfile
from collections import Counter
from http import HTTPStatus
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile

from rag.chunking.pipeline import MultiModalChunkingPipeline
from rag.core.errors import AppError
from rag.document_ingestion.enhanced_file_reader import (
    DocumentIngestionPipeline,
    suppress_stderr,
)
from rag.storage import BlobStore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
_MAX_UPLOAD_CHUNK = 1 << 20  # 1 MiB


async def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    """Persist an UploadFile to ``dest_dir`` with a sanitized name."""
    safe_name = Path(file.filename or 'upload').name
    dest = dest_dir / safe_name
    with dest.open('wb') as out:
        while True:
            chunk = await file.read(_MAX_UPLOAD_CHUNK)
            if not chunk:
                break
            out.write(chunk)
    return dest


def _should_snapshot(settings: Any) -> bool:
    return False


class DocumentService:
    """Owns document upload, listing, and deletion against a live chatbot."""

    def __init__(self, chatbot: Any) -> None:
        self._chatbot = chatbot

    def _blob(self, settings: Any) -> Any:
        """Reuse a chatbot-owned blob store or build one from settings."""
        owned = getattr(self._chatbot, 'blob', None)
        if owned is not None:
            return owned
        return BlobStore(settings)

    def _memory(self) -> Any:
        return getattr(self._chatbot, 'memory', None)

    def _chunk_counts(self, chunks: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = Counter()
        for chunk in chunks:
            meta = None
            if hasattr(chunk, 'chunk') and hasattr(chunk.chunk, 'metadata'):
                meta = chunk.chunk.metadata
            elif hasattr(chunk, 'metadata'):
                meta = chunk.metadata
            elif isinstance(chunk, dict):
                meta = chunk.get('metadata')
            if not isinstance(meta, dict):
                continue
            fn = meta.get('file_name') or 'unknown'
            counts[fn] += 1
        return dict(counts)

    async def upload(
        self,
        files: list[UploadFile],
        settings: Any,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Extract, chunk, embed, index, and persist originals to blob storage."""
        if not files:
            raise AppError(
                "At least one file is required",
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        unsupported = []
        for f in files:
            filename = f.filename or ''
            if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
                unsupported.append(filename)
        if unsupported:
            raise AppError(
                f"Unsupported file type(s): {', '.join(unsupported)}. "
                f"Accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
                status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )

        store = self._chatbot.store
        memory = self._memory()
        blob = self._blob(settings)
        blob_is_temp = getattr(self._chatbot, 'blob', None) is None
        tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
        try:
            saved = await asyncio.gather(*[_save_upload(f, tmp_dir) for f in files])

            with suppress_stderr():
                documents = DocumentIngestionPipeline().ingest_documents(tmp_dir)
            summary = DocumentIngestionPipeline().get_content_summary(documents)

            chunking = MultiModalChunkingPipeline(embedder=self._chatbot.embedder)
            chunks = chunking.process_documents(documents)
            if not chunks:
                raise AppError(
                    "No extractable content in the uploaded file(s)",
                    status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                )

            store.add_chunks([c.to_dict() for c in chunks])

            if _should_snapshot(settings):
                store.save(self._chatbot.store_path)

            per_file = self._chunk_counts(chunks)
            blob_paths = []
            for path in saved:
                blob_name = f"{project_id or 'default'}/{path.name}"
                content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                blob_path = None
                try:
                    with path.open('rb') as fh:
                        blob_path = blob.upload_file(blob_name, fh, content_type)
                except Exception as e:  # noqa: BLE001 - blob outage surfaces loudly
                    if blob_is_temp:
                        close = getattr(blob, 'close', None)
                        if callable(close):
                            close()
                    raise AppError(
                        "Blob storage is unavailable or misconfigured",
                        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                    ) from e
                blob_paths.append(blob_path)
                if memory is not None:
                    try:
                        memory.upsert_file(
                            file_name=path.name,
                            blob_path=blob_path,
                            content_type=content_type,
                            size_bytes=path.stat().st_size,
                            chunk_count=per_file.get(path.name, 0),
                            project_id=project_id,
                        )
                    except Exception as e:  # noqa: BLE001 - memory outage surfaces
                        raise AppError(
                            "Memory store is unavailable or misconfigured",
                            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                        ) from e

            backend = (store.get_stats() or {}).get('backend') or settings.vector_store.backend
            logger.info(
                "Uploaded %d file(s) -> %d chunk(s) (backend=%s, blobs=%d)",
                len(saved),
                len(chunks),
                backend,
                len(blob_paths),
            )
            return {
                'files_uploaded': len(saved),
                'documents_processed': summary['successful'],
                'documents_failed': summary['failed'],
                'total_chunks': len(chunks),
                'tables_extracted': summary['total_tables'],
                'images_extracted': summary['total_images'],
                'embedding_dimension': chunking.embedding_dimension,
                'vector_store_backend': backend,
                'blob_paths': blob_paths,
            }
        finally:
            if blob_is_temp:
                close = getattr(blob, 'close', None)
                if callable(close):
                    close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def list(self) -> dict[str, Any]:
        """List indexed documents, enriched with blob record metadata."""
        docs = [
            {'file_name': d.get('file_name', 'unknown'), 'chunks': int(d.get('chunks', 0))}
            for d in self._chatbot.store.list_documents()
        ]

        memory = self._memory()
        if memory is not None:
            try:
                by_name = {r['file_name']: r for r in memory.list_files()}
            except Exception:  # noqa: BLE001 - enrichment is best-effort
                by_name = {}
            for doc in docs:
                rec = by_name.get(doc['file_name'])
                if rec:
                    doc['blob_path'] = rec.get('blob_path')
                    doc['size_bytes'] = rec.get('size_bytes')
        return {
            'documents': docs,
            'total_documents': len(docs),
            'total_chunks': sum(d['chunks'] for d in docs),
        }

    def delete(self, file_name: str, settings: Any) -> dict[str, Any]:
        """Remove every chunk for ``file_name`` plus its blob and record."""
        try:
            deleted = self._chatbot.store.delete_by_document(file_name)
        except NotImplementedError as e:
            raise AppError(str(e), status_code=HTTPStatus.NOT_IMPLEMENTED) from e

        if deleted and _should_snapshot(settings):
            self._chatbot.store.save(self._chatbot.store_path)

        memory = self._memory()
        if memory is not None:
            try:
                for rec in memory.list_files():
                    if rec.get('file_name') == file_name and rec.get('blob_path'):
                        blob_name = rec['blob_path'].split('/', 1)[-1]
                        try:
                            self._blob(settings).delete(blob_name)
                        except Exception:  # noqa: BLE001 - best-effort
                            logger.debug("Failed to delete blob for %s", file_name)
                memory.delete_file(file_name)
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                logger.debug("Failed to clean file record for %s", file_name)

        return {'file_name': file_name, 'deleted_chunks': deleted}

    def download(self, file_name: str, settings: Any) -> tuple[bytes, str]:
        """Return (bytes, content_type) of a file's original upload from blob."""
        memory = self._memory()
        rec = None
        if memory is not None:
            by_name = {r['file_name']: r for r in memory.list_files()}
            rec = by_name.get(file_name)
        if rec is None or not rec.get('blob_path'):
            raise AppError(
                f"No stored blob for file: {file_name}", status_code=HTTPStatus.NOT_FOUND
            )
        blob = self._blob(settings)
        blob_name = rec['blob_path'].split('/', 1)[-1]
        try:
            data = blob.download(blob_name)
        except Exception as e:  # noqa: BLE001 - blob outage surfaces
            raise AppError(
                "Blob storage is unavailable",
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            ) from e
        content_type = rec.get('content_type') or 'application/octet-stream'
        return data, content_type