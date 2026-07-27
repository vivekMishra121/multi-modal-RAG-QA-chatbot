"""Embedder implementation using sentence-transformers"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer
from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbedderInterface):
    """Embedder using sentence-transformers models"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 32):
        """
        Initialize embedder
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5)
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = self._load_model()
        logger.info(f"Loaded embedding model: {model_name}")
    
    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model"""
        return SentenceTransformer(self.model_name)
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query with instruction"""
        if not query:
            return []
        
        # Add query instruction for BGE models
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.model.encode(
            query_with_instruction,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return embedding.tolist()
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
