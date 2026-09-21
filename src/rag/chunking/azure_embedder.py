"""Azure OpenAI Embedder (config-driven, retries + timeouts)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import AzureOpenAI

from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedder(EmbedderInterface):
    """Embeddings served from an Azure OpenAI deployment."""

    def __init__(
        self,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Args:
            azure_endpoint: Resource endpoint (defaults to ``AZURE_OPENAI_ENDPOINT``).
            api_key: Azure API key (defaults to ``AZURE_OPENAI_API_KEY``).
            api_version: Azure REST API version (defaults to settings).
            deployment_name: Embedding deployment name. Defaults to
                ``EMBEDDING_DEPLOYMENT_NAME`` then ``AZURE_EMBEDDING_DEPLOYMENT``.
            dimensions: Output dimensions; 0/None disables explicit sizing.
            batch_size: Number of texts per API call.
            max_retries: SDK retry count on 429/5xx.
            timeout: Per-request timeout in seconds.
        """
        from rag.core.config import get_settings

        settings = get_settings()
        azure = settings.azure
        embedding = settings.embedding

        self.azure_endpoint = (
            azure_endpoint or azure.openai_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.api_key = api_key or azure.openai_api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.api_version = api_version or azure.openai_api_version
        self.deployment_name = (
            deployment_name or embedding.deployment_name or azure.embedding_deployment
        )
        self.dimensions = embedding.dimensions if dimensions is None else dimensions
        self.batch_size = batch_size or embedding.batch_size
        self.timeout = timeout or 60.0

        if not self.azure_endpoint or not self.api_key:
            raise ValueError(
                "Azure endpoint and API key required "
                "(set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)"
            )

        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            timeout=self.timeout,
            max_retries=max_retries if max_retries is not None else embedding.max_retries,
        )
        logger.info(
            "Loaded Azure OpenAI embedder: %s (dim=%s, batch=%s, endpoint=%s)",
            self.deployment_name,
            self.dimensions,
            self.batch_size,
            self.azure_endpoint,
        )

    def _embedding_kwargs(self) -> dict:
        """Optional dimensions arg (disabled for non-matryoshka models)."""
        if self.dimensions and self.dimensions > 0:
            return {"dimensions": self.dimensions}
        return {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment_name,
                **self._embedding_kwargs(),
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a single query."""
        if not query:
            return []

        response = self.client.embeddings.create(
            input=[query],
            model=self.deployment_name,
            **self._embedding_kwargs(),
        )
        return response.data[0].embedding

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self.client.close()

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimensions