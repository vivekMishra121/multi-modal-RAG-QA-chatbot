"""OpenAI Embedder - Best quality embeddings (config-driven)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbedderInterface):
    """OpenAI embeddings via the ``text-embedding-3-*`` family."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Args:
            model_name: Embedding model (defaults to ``EMBEDDING_MODEL_NAME``).
            api_key: OpenAI API key (defaults to ``OPENAI_API_KEY``).
            dimensions: Output dimensions; 0/None disables explicit sizing.
            batch_size: Number of texts per API call.
            max_retries: SDK retry count on 429/5xx.
            timeout: Per-request timeout in seconds.
        """
        from rag.core.config import get_settings

        settings = get_settings().embedding

        self.model_name = model_name or settings.model_name
        self.api_key = api_key or settings.api_key or os.getenv("OPENAI_API_KEY")
        self.dimensions = settings.dimensions if dimensions is None else dimensions
        self.batch_size = batch_size or settings.batch_size
        self.timeout = timeout or 60.0

        if not self.api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")

        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=max_retries if max_retries is not None else settings.max_retries,
        )
        logger.info(
            "Loaded OpenAI embedding model: %s (dim=%s, batch=%s)",
            self.model_name,
            self.dimensions,
            self.batch_size,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            if self.dimensions and self.dimensions > 0:
                response = self.client.embeddings.create(
                    input=batch, model=self.model_name, dimensions=self.dimensions
                )
            else:
                response = self.client.embeddings.create(
                    input=batch, model=self.model_name
                )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query."""
        if not query:
            return []

        if self.dimensions and self.dimensions > 0:
            response = self.client.embeddings.create(
                input=[query], model=self.model_name, dimensions=self.dimensions
            )
        else:
            response = self.client.embeddings.create(
                input=[query], model=self.model_name
            )
        return response.data[0].embedding

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self.client.close()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimensions