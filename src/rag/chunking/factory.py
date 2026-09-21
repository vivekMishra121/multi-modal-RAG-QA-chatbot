"""Embedder factory - resolves the configured provider to a concrete embedder."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from .base import EmbedderInterface

if TYPE_CHECKING:
    from rag.core.config import Settings

logger = logging.getLogger(__name__)


def build_embedder(
    settings: Settings,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    provider: Optional[str] = None,
) -> EmbedderInterface:
    """Instantiate the embedder selected by settings.

    Args:
        settings: Application settings.
        model_name: Optional override of the embedding model/deployment name.
        batch_size: Optional override of the embedding batch size.
        provider: Optional override of ``settings.embedding.provider``.
    """
    chosen = provider or settings.embedding.provider

    if chosen == "azure_openai":
        from .azure_embedder import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(
            deployment_name=model_name,
            batch_size=batch_size,
        )

    if chosen == "openai":
        from .openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(
            model_name=model_name,
            batch_size=batch_size,
        )

    if chosen == "sentence_transformers":
        from .embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_name=model_name,
            batch_size=batch_size,
        )

    if chosen == "bge_m3":
        from .bge_m3_embedder import BGEM3Embedder

        return BGEM3Embedder(model_name=model_name or "BAAI/bge-m3")

    raise ValueError(f"Unknown embedding provider: {chosen}")