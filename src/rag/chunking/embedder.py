"""Embedder implementation using sentence-transformers (config-driven)."""

from __future__ import annotations

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbedderInterface):
    """Embedder using a local sentence-transformers model."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Args:
            model_name: HuggingFace model name (defaults to ``EMBEDDING_MODEL_NAME``).
            batch_size: Batch size for embedding generation.
        """
        from rag.core.config import get_settings

        settings = get_settings().embedding

        self.model_name = model_name or settings.model_name
        self.batch_size = batch_size or settings.batch_size or 32
        self.model = SentenceTransformer(self.model_name)
        logger.info("Loaded embedding model: %s", self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query with instruction."""
        if not query:
            return []

        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(
            query_with_instruction,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        dim = self.model.get_sentence_embedding_dimension()
        return int(dim or 0)