"""Base classes and interfaces for multi-modal chunking system"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum


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
    metadata: Dict[str, Any]
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
    def chunk(self, content: Any, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk content into smaller pieces"""
        pass


class EmbedderInterface(ABC):
    """Interface for embedding generation"""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        pass
