"""RAG chatbot - the conversational entrypoint shared by CLI, UI, and API.

Settings-driven and free of module-level shared mutable state: each instance
owns its embedder, store, retriever, and pipeline so the FastAPI service can
create/destroy instances safely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.chunking.pipeline import MultiModalChunkingPipeline
from rag.document_ingestion.enhanced_file_reader import DocumentIngestionPipeline
from rag.qa_generation import RAGPipeline
from rag.retrieval import Retriever
from rag.vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class RAGChatbot:
    """RAG-powered chatbot for query answering."""

    def __init__(
        self,
        settings: Any | None = None,
        store: BaseVectorStore | None = None,
        embedder: Any | None = None,
        memory: Any | None = None,
        blob: Any | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        max_context_tokens: int | None = None,
        top_k: int | None = None,
    ):
        """
        Args:
            settings: Application settings (defaults to process settings).
            store: Pre-built vector store (bypasses store construction).
            embedder: Pre-built embedder (bypasses the factory).
            memory: Pre-built Cosmos memory store (bypasses construction;
                tests inject a fake). Required: configures from settings otherwise.
            blob: Pre-built blob store (bypasses construction; uploads use it
                when provided, otherwise the API builds one per upload).
            api_key: Optional LLM API key override.
            model_name: Optional LLM model override.
            max_context_tokens: Context window budget.
            top_k: Number of chunks to retrieve.
        """
        from rag.chunking.factory import build_embedder
        from rag.core.config import get_settings
        from rag.cosmos import build_cosmos_store

        self.settings = settings or get_settings()

        # Key gating lives inside the embedder constructors: open-source
        # providers (bge_m3) need no API key; OpenAI/Azure raise a clear
        # ValueError when their key is missing.
        self.embedder: Any = embedder or build_embedder(self.settings)

        self.store = store or self._build_store()

        # Cosmos memory is a hard requirement (no fallback/no no-op). Tests
        # inject a fake via ``memory``; production needs COSMOS_* credentials.
        self.memory: Any = memory or build_cosmos_store(self.settings)

        # Optional blob store (raw uploads). When absent the document service
        # builds one lazily per upload from settings.
        self.blob: Any = blob

        if self.store.dimension != self.embedder.embedding_dimension:
            logger.warning(
                "Embedding dimension mismatch: store=%s vs embedder=%s. "
                "Re-index with the configured embedding model.",
                self.store.dimension,
                self.embedder.embedding_dimension,
            )

        retriever = Retriever(
            self.store,
            use_reranker=self.settings.retrieval.use_reranker,
            use_query_expansion=self.settings.retrieval.use_query_expansion,
        )
        self.rag = RAGPipeline(
            retriever=retriever,
            embedder=self.embedder,
            api_key=api_key,
            model_name=model_name,
            max_context_tokens=(
                max_context_tokens or self.settings.llm.max_context_tokens
            ),
            retrieval_strategy=self.settings.retrieval.strategy,
            top_k=top_k or self.settings.retrieval.top_k,
        )

    def _build_store(self) -> BaseVectorStore:
        """Construct the Azure AI Search vector store."""
        return get_vector_store(self.settings, dimension=self.embedder.embedding_dimension)

    def chat(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Answer a question, persisting the turn to Cosmos memory.

        Args:
            question: The user question.
            filters: Optional retrieval metadata filters.
            conversation_id: Required - messages are scoped to this id.
            project_id: Optional - scope retrieval to a project's filters.

        Returns:
            ``{'answer', 'sources', 'success', 'error', 'num_chunks',
            'strategy', 'conversation_id'}``
        """
        from rag.core.errors import AppError

        if not conversation_id:
            raise ValueError("conversation_id is required for chat")

        if filters is not None:
            filters = dict(filters)
        else:
            filters = {}

        if project_id:
            try:
                project = self.memory.get_project(project_id)
            except Exception as e:  # noqa: BLE001 - memory outage is explicit
                raise AppError("Memory store unavailable", status_code=503) from e
            if project is None:
                raise AppError(f"Project not found: {project_id}", status_code=404)
            merged = dict(project.get('filters') or {})
        else:
            merged = {}

        merged.update(filters)

        try:
            history = self.memory.get_conversation(
                conversation_id, limit=self.settings.cosmos.max_history_turns
            )
        except Exception as e:  # noqa: BLE001 - memory outage is explicit
            raise AppError("Memory store unavailable", status_code=503) from e

        result = self.rag.query(question, filters=merged or None, history=history)

        try:
            self.memory.append_message(
                conversation_id, "user", question, project_id=project_id
            )
            if result.get('success') and result.get('answer'):
                self.memory.append_message(
                    conversation_id,
                    "assistant",
                    result['answer'],
                    sources=result.get('sources', []),
                    project_id=project_id,
                )
        except Exception as e:  # noqa: BLE001 - persist as part of the turn
            raise AppError("Memory store unavailable", status_code=503) from e

        return {
            'answer': result['answer'],
            'sources': result['sources'],
            'success': result['success'],
            'error': result.get('error'),
            'num_chunks': result.get('num_chunks_used', 0),
            'strategy': result.get('retrieval_strategy'),
            'conversation_id': conversation_id,
        }

    def close(self) -> None:
        """Release owned resources (HTTP clients, models, store connections)."""
        for obj in (self.embedder, self.store, self.memory, self.blob):
            close = getattr(obj, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("Failed to close %s", type(obj).__name__, exc_info=True)
        retriever = getattr(self.rag, 'retriever', None)
        if retriever is not None:
            expander = getattr(retriever, 'query_expander', None)
            close = getattr(expander, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.warning("Failed to close query expander", exc_info=True)


# ---------------------------------------------------------------------------
# Compatibility helpers (CLI/evaluation). The FastAPI service manages its own
# instances per application lifecycle. The cached singleton is NOT safe for
# concurrent request processing.
# ---------------------------------------------------------------------------

_chatbot: RAGChatbot | None = None


def get_chatbot(**kwargs) -> RAGChatbot:
    """Get or create the process-wide chatbot instance."""
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot(**kwargs)
    return _chatbot


def ask(question: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience wrapper around the cached chatbot."""
    return get_chatbot().chat(question, filters)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(
    document_path: str,
    settings: Any | None = None,
    recreate: bool = True,
    embedder: Any | None = None,
) -> dict[str, Any]:
    """Ingest, chunk, embed, and index a document or folder.

    Args:
        document_path: PDF/DOCX/TXT file or directory.
        settings: Application settings (defaults to process settings).
        recreate: Drop existing points before indexing (full rebuild).
        embedder: Pre-built embedder (bypasses the factory; used by tests).

    Returns:
        Summary of the indexing run.
    """
    from rag.core.config import get_settings

    settings = settings or get_settings()

    ingestion = DocumentIngestionPipeline()
    documents = ingestion.ingest_documents(Path(document_path))
    summary = ingestion.get_content_summary(documents)

    chunking = MultiModalChunkingPipeline(embedder=embedder)
    chunks = chunking.process_documents(documents)
    if not chunks:
        raise ValueError("No chunks were produced; nothing to index.")

    store = get_vector_store(settings, dimension=chunking.embedding_dimension)
    if recreate:
        store.delete_all()
    store.add_chunks([c.to_dict() for c in chunks])

    close = getattr(store, 'close', None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.warning("Failed to close store after indexing", exc_info=True)

    return {
        'documents_processed': summary['successful'],
        'documents_failed': summary['failed'],
        'total_chunks': len(chunks),
        'tables_extracted': summary['total_tables'],
        'images_extracted': summary['total_images'],
        'embedding_dimension': chunking.embedding_dimension,
        'vector_store_backend': store.backend,
    }