"""OpenAI Embedder - Best quality embeddings"""

import logging
import os
from typing import List
from openai import OpenAI
from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class OpenAIEmbedder(EmbedderInterface):
    """OpenAI embeddings - highest quality"""
    
    def __init__(self, model_name: str = "text-embedding-3-large", api_key: str = None, dimensions: int = 1536):
        """
        Initialize OpenAI embedder
        
        Args:
            model_name: OpenAI embedding model
            api_key: OpenAI API key
            dimensions: Output dimensions (1536 for balanced performance/cost)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._dimension = dimensions
        
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Loaded OpenAI embedding model: {model_name} (dim={dimensions})")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []
        
        # Batch process in chunks of 100
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
                dimensions=self._dimension
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        if not query:
            return []
        
        response = self.client.embeddings.create(
            input=[query],
            model=self.model_name,
            dimensions=self._dimension
        )
        
        return response.data[0].embedding
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension
