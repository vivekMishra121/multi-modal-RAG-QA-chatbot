"""End-to-end RAG pipeline: Query -> Retrieve -> Generate."""

from __future__ import annotations

import logging
from typing import Any, Optional

from .context_manager import ContextManager
from .qa_chain import QAChain

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""

    def __init__(
        self,
        retriever,
        embedder,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        retrieval_strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        context_manager: Optional[ContextManager] = None,
    ):
        """
        Args:
            retriever: Retriever instance.
            embedder: Embedder instance.
            api_key: Optional LLM API key (defaults to settings).
            model_name: Optional LLM model name (defaults to settings).
            max_context_tokens: Context window budget for retrieved chunks.
            retrieval_strategy: 'auto', 'standard', 'mmr', 'hybrid', or 'fusion'.
            top_k: Number of chunks to retrieve.
            context_manager: Optional pre-built context manager.
        """
        from rag.core.config import get_settings

        settings = get_settings()

        self.retriever = retriever
        self.embedder = embedder
        self.retrieval_strategy = retrieval_strategy or settings.retrieval.strategy
        self.top_k = top_k or settings.retrieval.top_k
        self.max_context_tokens = max_context_tokens or settings.llm.max_context_tokens

        self.qa_chain = QAChain(api_key=api_key, model_name=model_name)
        self.context_manager = context_manager or ContextManager(
            model_name=self.qa_chain.model_name,
            max_tokens=self.max_context_tokens,
        )

        logger.info(
            "Initialized RAG pipeline (strategy=%s, top_k=%s, max_tokens=%s)",
            self.retrieval_strategy,
            self.top_k,
            self.max_context_tokens,
        )

    def query(
        self,
        question: str,
        filters: Optional[dict[str, Any]] = None,
        history: Optional[list[dict[str, Any]]] = None,
        **retrieval_kwargs,
    ) -> dict[str, Any]:
        """Process a query end-to-end.

        Args:
            question: User question.
            filters: Optional metadata filters for retrieval.
            history: Optional prior conversation turns (injected in the prompt).

        Returns:
            Dict with answer, sources, and metadata.
        """
        logger.info("Processing query: %s", question)

        # Step 1: Embed query (dense, and sparse for hybrid backends if available).
        query_embedding = self.embedder.embed_query(question)
        query_sparse = None
        if hasattr(self.embedder, 'embed_query_with_sparse'):
            query_sparse = self.embedder.embed_query_with_sparse(question)[1]

        # Step 2: Retrieve relevant chunks.
        chunks_with_scores = self.retriever.retrieve(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.top_k,
            strategy=self.retrieval_strategy,
            filters=filters,
            query_sparse_embedding=query_sparse,
            **retrieval_kwargs,
        )

        if not chunks_with_scores:
            return {
                'answer': "No relevant information found in the documents.",
                'sources': [],
                'success': False,
                'error': 'No chunks retrieved',
            }

        logger.info("Retrieved %d chunks", len(chunks_with_scores))

        # Step 3: Select chunks within the token budget and format context.
        selected_chunks = self.context_manager.select_best_chunks(
            chunks_with_scores, question
        )
        if not selected_chunks:
            return {
                'answer': "No relevant information found in the documents.",
                'sources': [],
                'success': False,
                'error': 'Context budget too small for any chunk',
            }

        context = self.context_manager.format_context(selected_chunks)

        # Step 4: Generate answer.
        result = self.qa_chain.generate_answer(question, context, history=history)

        # Step 5: Add source information.
        sources = self._extract_sources(selected_chunks)

        return {
            'answer': result['answer'],
            'sources': sources,
            'success': result['success'],
            'error': result['error'],
            'num_chunks_used': len(selected_chunks),
            'retrieval_strategy': self.retrieval_strategy,
        }

    def _extract_sources(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract source citations from chunks."""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            content_preview = chunk['content'][:200]
            sources.append({
                'source_id': i,
                'file_name': metadata.get('file_name', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'chunk_type': chunk.get('chunk_type', 'text'),
                'content_preview': content_preview + ('...' if len(chunk['content']) > 200 else ''),
            })
        return sources