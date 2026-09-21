"""Advanced retrieval strategies for RAG.

Backend-agnostic. Dense-only search delegates to the vector store; when the
store implements ``hybrid_search`` (Qdrant sparse, Azure AI Search hybrid),
the auto strategy prefers it. The heavy cross-encoder reranker and the
optional TF-IDF path are loaded lazily so they never block cold start if
unused or unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .query_expansion import QueryExpander

logger = logging.getLogger(__name__)


class Retriever:
    """Advanced retrieval strategies: hybrid, MMR, and cross-encoder reranking."""

    def __init__(
        self,
        vector_store,
        use_reranker: bool = True,
        use_query_expansion: bool = True,
        reranker_model: Optional[str] = None,
        fetch_multiplier: Optional[int] = None,
        query_expander: Optional[Any] = None,
    ):
        """
        Args:
            vector_store: Any store implementing ``search`` (and optionally
                ``hybrid_search``).
            use_reranker: Enable cross-encoder reranking.
            use_query_expansion: Enable LLM query expansion (increase recall).
            reranker_model: Cross-encoder model name (defaults to settings).
            fetch_multiplier: Candidates fetched per final result when
                reranking (defaults to settings).
            query_expander: Optional pre-built QueryExpander instance.
        """
        from rag.core.config import get_settings

        settings = get_settings()

        self.vector_store = vector_store
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.fetch_multiplier = fetch_multiplier or settings.retrieval.fetch_multiplier
        self._reranker_model = reranker_model or settings.retrieval.reranker_model

        self.reranker = None
        self.query_expander = query_expander

        if use_query_expansion:
            try:
                self.query_expander = query_expander or QueryExpander()
                logger.info("Loaded query expander")
            except Exception as e:  # noqa: BLE001 - degraded retrieval is non-fatal
                logger.warning("Failed to load query expander: %s", e)
                self.use_query_expansion = False

        if use_reranker:
            self._load_reranker()

        # TF-IDF is only needed for FAISS-style dense-only stores; built lazily.
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

        logger.info(
            "Initialized retriever (reranker=%s, query_expansion=%s)",
            self.use_reranker,
            self.use_query_expansion,
        )

    # ------------------------------------------------------------------ state

    def _load_reranker(self) -> None:
        """Lazily load the cross-encoder reranker (heavy model, optional dep)."""
        try:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self._reranker_model)
            logger.info("Loaded cross-encoder reranker: %s", self._reranker_model)
        except Exception as e:  # noqa: BLE001 - model download may be unavailable
            logger.warning("Failed to load reranker %s: %s", self._reranker_model, e)
            self.use_reranker = False

    def _ensure_tfidf(self) -> None:
        """Build the TF-IDF index on first use (dense-only hybrid fallback)."""
        if self._tfidf_matrix is not None or not self.vector_store.chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            texts = [chunk['content'] for chunk in self.vector_store.chunks]
            self._tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self._tfidf_matrix = self._tfidf_vectorizer.fit_transform(texts)  # type: ignore[attr-defined]
            logger.info("Built TF-IDF index for hybrid search (%d docs)", len(texts))
        except Exception as e:  # noqa: BLE001 - sklearn optional
            logger.warning("TF-IDF unavailable, hybrid degrades to vector search: %s", e)
            self._tfidf_matrix = None

    # ------------------------------------------------------------- retrieval

    def retrieve(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        strategy: str = 'auto',
        filters: Optional[dict[str, Any]] = None,
        rerank: bool = True,
        query_sparse_embedding: Optional[dict[str, float]] = None,
        **kwargs,
    ) -> list[tuple[dict[str, Any], float]]:
        """Unified retrieval interface with automatic strategy selection.

        Args:
            query_embedding: Query dense vector.
            query_text: Query text.
            top_k: Number of results to return.
            strategy: 'auto', 'standard', 'hybrid', or 'mmr'.
            filters: Optional metadata filters.
            rerank: Apply cross-encoder reranking (default: True).
            query_sparse_embedding: Sparse (BM25-style) query vector for
                hybrid stores.
            **kwargs: Strategy-specific parameters.

        Returns:
            List of ``(chunk, score)`` tuples.
        """
        if strategy == 'auto':
            strategy = self._select_strategy(query_text)
            logger.debug("Auto-selected strategy: %s", strategy)

        fetch_k = (
            top_k * self.fetch_multiplier
            if rerank and self.use_reranker
            else top_k * 2
        )

        hybrid_capable = hasattr(self.vector_store, 'hybrid_search')

        if strategy in ('hybrid', 'fusion') and hybrid_capable:
            results = self.vector_store.hybrid_search(
                dense_embedding=query_embedding,
                sparse_embedding=query_sparse_embedding,
                query_text=query_text,
                top_k=fetch_k,
                filters=filters,
                **{k: v for k, v in kwargs.items() if k not in ('alpha',)},
            )
        elif strategy == 'mmr':
            results = self.mmr_search(
                query_embedding, top_k=fetch_k, filters=filters, **kwargs
            )
        elif strategy == 'hybrid':
            results = self.hybrid_search(
                query_embedding, query_text, top_k=fetch_k, filters=filters, **kwargs
            )
        else:
            results = self.vector_store.search(
                query_embedding, top_k=fetch_k, filters=filters
            )

        if rerank and self.use_reranker and len(results) > 0:
            return self._rerank(query_text, results, top_k)
        return results[:top_k]

    def _select_strategy(self, query: str) -> str:
        """Pick the best strategy for the configured backend."""
        if hasattr(self.vector_store, 'hybrid_search'):
            return 'hybrid'  # native dense + sparse (RRF) fusion
        if self.use_query_expansion and len(query.split()) > 6:
            return 'hybrid'
        return 'standard'

    def mmr_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Maximal Marginal Relevance for diverse results."""
        candidates = self.vector_store.search(query_embedding, top_k=fetch_k, filters=filters)
        if len(candidates) <= top_k:
            return candidates[:top_k]

        query_vec = np.asarray(query_embedding, dtype='float32').reshape(1, -1)
        candidate_embeddings = np.asarray(
            [c[0]['embedding'] for c in candidates], dtype='float32'
        )

        if self.vector_store.distance_metric == 'cosine':
            query_vec = self._l2_normalize(query_vec)
            candidate_embeddings = self._l2_normalize(candidate_embeddings)

        selected_indices: list[int] = []
        remaining_indices = list(range(len(candidates)))

        while len(selected_indices) < top_k and remaining_indices:
            if not selected_indices:
                best_idx = 0
            else:
                relevance = candidate_embeddings @ query_vec.T
                relevance = relevance[:, 0]
                selected_embeddings = candidate_embeddings[selected_indices]
                similarity_to_selected = selected_embeddings @ candidate_embeddings.T
                max_similarity = similarity_to_selected.max(axis=0)
                mmr_scores = (
                    lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                )
                best_idx = remaining_indices[int(np.argmax(mmr_scores[remaining_indices]))]

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [candidates[idx] for idx in selected_indices]

    def hybrid_search(
        self,
        query_embedding: list[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Hybrid search: vector + keyword (TF-IDF) for dense-only stores.

        Falls back to pure vector search when TF-IDF is unavailable.
        """
        self._ensure_tfidf()
        if self._tfidf_matrix is None:
            return self.vector_store.search(query_embedding, top_k, filters)

        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

        vector_results = self.vector_store.search(query_embedding, top_k=top_k * 2, filters=filters)

        query_tfidf = self._tfidf_vectorizer.transform([query_text])
        keyword_scores = sklearn_cosine(query_tfidf, self._tfidf_matrix).flatten()

        combined_scores: dict[int, float] = {}
        for chunk, vec_score in vector_results:
            chunk_idx = self.vector_store.chunks.index(chunk)
            kw_score = keyword_scores[chunk_idx]
            combined_scores[chunk_idx] = alpha * vec_score + (1 - alpha) * kw_score

        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [
            (self.vector_store.chunks[idx], score) for idx, score in ranked[:top_k]
        ]

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[dict[str, Any], float]],
        top_k: int,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rerank candidates using a cross-encoder for better precision.
        The model is loaded on first use to keep cold start cheap.
        """
        if not self.use_reranker or not self.reranker:
            return candidates[:top_k]

        pairs = [(query, chunk['content']) for chunk, _ in candidates]
        rerank_scores = self.reranker.predict(pairs)

        reranked = [
            (candidates[i][0], float(score)) for i, score in enumerate(rerank_scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        logger.debug("Reranked %d candidates to top %d", len(candidates), top_k)
        return reranked[:top_k]

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalization (cosine-equivalent inner products)."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return matrix / norms