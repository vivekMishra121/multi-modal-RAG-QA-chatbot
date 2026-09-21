"""Multi-modal chunking and embedding pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .base import Chunk
from .chunkers import ImageChunker, TableChunker, TextChunker

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """Chunk with its embedding vector."""

    chunk: Chunk
    embedding: list[float]
    sparse_embedding: Optional[dict[str, float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            'chunk_id': self.chunk.chunk_id,
            'content': self.chunk.content,
            'chunk_type': self.chunk.chunk_type.value,
            'metadata': self.chunk.metadata,
            'embedding': self.embedding,
        }
        if self.sparse_embedding is not None:
            d['sparse_embedding'] = self.sparse_embedding
        return d


class MultiModalChunkingPipeline:
    """Orchestrates chunking and embedding for multi-modal content."""

    def __init__(
        self,
        text_chunk_size: Optional[int] = None,
        text_chunk_overlap: Optional[int] = None,
        embedding_model: Optional[str] = None,
        use_openai: Optional[bool] = None,
        batch_size: Optional[int] = None,
        embedder: Optional[Any] = None,
    ):
        """
        Args:
            text_chunk_size: Size of text chunks (defaults to settings).
            text_chunk_overlap: Overlap between text chunks (defaults to settings).
            embedding_model: Embedding model/deployment name override.
            use_openai: Backwards-compatible switch. ``True`` -> OpenAI/Azure
                provider, ``False`` -> open-source BGE-M3. ``None`` -> settings.
            batch_size: Embedding batch size override.
            embedder: Pre-built embedder instance (bypasses the factory).
        """
        from rag.core.config import get_settings

        settings = get_settings()

        self.text_chunk_size = text_chunk_size or settings.chunking.text_chunk_size
        self.text_chunk_overlap = (
            text_chunk_overlap
            if text_chunk_overlap is not None
            else settings.chunking.text_chunk_overlap
        )
        self.text_chunker = TextChunker(self.text_chunk_size, self.text_chunk_overlap)
        self.table_chunker = TableChunker()
        self.image_chunker = ImageChunker()

        self._use_sparse = False
        if embedder is not None:
            self.embedder = embedder
        elif use_openai is False:
            # Explicitly requested open-source path: BGE-M3 (dense + sparse).
            from .bge_m3_embedder import BGEM3Embedder

            self.embedder = BGEM3Embedder(model_name=embedding_model or "BAAI/bge-m3")
            self._use_sparse = True
        else:
            from .factory import build_embedder

            self.embedder = build_embedder(
                settings,
                model_name=embedding_model,
                batch_size=batch_size,
            )
            self._use_sparse = hasattr(self.embedder, 'embed_query_with_sparse')

        logger.info(
            "Initialized multi-modal chunking pipeline "
            "(provider=%s, sparse=%s, chunk_size=%s, overlap=%s)",
            getattr(self.embedder, 'model_name', type(self.embedder).__name__),
            self._use_sparse,
            self.text_chunk_size,
            self.text_chunk_overlap,
        )

    def process_document(self, document: dict[str, Any]) -> list[ChunkWithEmbedding]:
        """Process a single document into chunks with embeddings."""
        if 'error' in document:
            logger.warning("Skipping document with error: %s", document.get('file_path'))
            return []

        content = document.get('content', {})
        file_name = (
            content.get('metadata', {}).get('file_name')
            or Path(document.get('file_path', 'unknown')).name
        )
        base_metadata = {
            'source': document.get('file_path'),
            'file_name': file_name,
            'total_pages': content.get('metadata', {}).get('pages', 0),
        }

        chunks: list[Chunk] = []
        chunks.extend(self._chunk_text(content.get('text', ''), base_metadata))
        chunks.extend(self._chunk_tables(content.get('tables', []), base_metadata))
        chunks.extend(self._chunk_images(content.get('images', []), base_metadata))

        logger.info("Created %d chunks from %s", len(chunks), file_name)
        return self._embed_chunks(chunks)

    def process_documents(self, documents: list[dict[str, Any]]) -> list[ChunkWithEmbedding]:
        """Process multiple documents."""
        all_chunks: list[ChunkWithEmbedding] = []
        for doc in documents:
            all_chunks.extend(self.process_document(doc))

        logger.info(
            "Processed %d documents into %d chunks", len(documents), len(all_chunks)
        )
        return all_chunks

    def _chunk_text(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk text content."""
        if not text:
            return []
        return self.text_chunker.chunk(text, metadata)

    def _chunk_tables(self, tables: list[dict], metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk table content."""
        chunks: list[Chunk] = []
        for table in tables:
            table_metadata = {**metadata, 'page': table.get('page')}
            chunks.extend(self.table_chunker.chunk(table, table_metadata))
        return chunks

    def _chunk_images(self, images: list[dict], metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk image content."""
        chunks: list[Chunk] = []
        for image in images:
            image_metadata = {**metadata, 'page': image.get('page')}
            chunks.extend(self.image_chunker.chunk(image, image_metadata))
        return chunks

    def _embed_chunks(self, chunks: list[Chunk]) -> list[ChunkWithEmbedding]:
        """Generate embeddings for chunks (dense, and sparse if available)."""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]

        if self._use_sparse and hasattr(self.embedder, 'embed_with_sparse'):
            dense_vecs, sparse_weights = self.embedder.embed_with_sparse(texts)
            return [
                ChunkWithEmbedding(
                    chunk=chunk, embedding=embedding, sparse_embedding=sparse
                )
                for chunk, embedding, sparse in zip(chunks, dense_vecs, sparse_weights)
            ]

        embeddings = self.embedder.embed(texts)
        return [
            ChunkWithEmbedding(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedder.embedding_dimension