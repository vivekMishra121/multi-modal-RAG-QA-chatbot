"""Vector store contract and factory — Azure AI Search only."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from rag.core.config import Settings


class BaseVectorStore(ABC):
    """Contract shared by every vector store backend."""

    distance_metric: str
    chunks: list[dict[str, Any]]
    dimension: int = 1536

    @abstractmethod
    def add_chunks(self, chunk_dicts: list[dict[str, Any]]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]: ...

    @abstractmethod
    def delete_all(self) -> None: ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def hybrid_search(
        self,
        dense_embedding: list[float],
        sparse_embedding: Optional[dict[str, float]],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        raise NotImplementedError(f"{type(self).__name__} does not support hybrid search")

    def list_documents(self) -> list[dict[str, Any]]:
        raise NotImplementedError(f"{type(self).__name__} does not support list_documents()")

    def delete_by_document(self, file_name: str) -> int:
        raise NotImplementedError(f"{type(self).__name__} does not support delete_by_document()")


def get_vector_store(
    settings: Settings,
    dimension: Optional[int] = None,
) -> BaseVectorStore:
    """Build the Azure AI Search vector store from settings."""
    from .azure_search_store import AzureAISearchStore

    vs = settings.vector_store
    return AzureAISearchStore(
        endpoint=vs.azure_search.endpoint,
        api_key=vs.azure_search.api_key,
        index_name=vs.azure_search.index_name,
        dimension=dimension or 1536,
        semantic_rank=vs.azure_search.semantic_rank,
        semantic_config_name=vs.azure_search.semantic_config_name,
    )
