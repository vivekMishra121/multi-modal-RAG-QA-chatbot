"""Base classes and interfaces for multi-modal chunking system"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Enumeration of chunk types"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class Chunk:
    """Immutable chunk data structure"""
    content: str
    chunk_type: ChunkType
    metadata: dict[str, Any]
    chunk_id: str
    
    def __post_init__(self):
        """Validate chunk data"""
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if not self.chunk_id:
            raise ValueError("Chunk ID is required")


class ChunkerInterface(ABC):
    """Interface for all chunking strategies"""
    
    @abstractmethod
    def chunk(self, content: Any, metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk content into smaller pieces"""
        pass


class EmbedderInterface(ABC):
    """Interface for embedding generation"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts"""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query"""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Vector dimension produced by this embedder"""
        pass

    def close(self) -> None:  # noqa: B027 - optional hook for HTTP-backed embedders
        """Release any underlying resources (default: no-op)."""
        pass
