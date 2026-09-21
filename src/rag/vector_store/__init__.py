"""Vector store module — Azure AI Search backend."""

from .base import BaseVectorStore, get_vector_store
from .azure_search_store import AzureAISearchStore

__all__ = [
    'BaseVectorStore',
    'AzureAISearchStore',
    'get_vector_store',
]
