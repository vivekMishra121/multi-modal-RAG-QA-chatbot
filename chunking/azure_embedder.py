"""Azure OpenAI Embedder"""

import logging
import os
from typing import List
from openai import AzureOpenAI
from .base import EmbedderInterface

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedder(EmbedderInterface):
    """Azure OpenAI embeddings"""
    
    def __init__(
        self,
        azure_endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-02-01",
        deployment_name: str = "text-embedding-3-large",
        dimensions: int = 1536
    ):
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment_name = deployment_name
        self._dimension = dimensions
        
        if not self.azure_endpoint or not self.api_key:
            raise ValueError("Azure endpoint and API key required")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=api_version
        )
        logger.info(f"Loaded Azure OpenAI embedder: {deployment_name}")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.deployment_name,
                dimensions=self._dimension
            )
            all_embeddings.extend([item.embedding for item in response.data])
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        if not query:
            return []
        
        response = self.client.embeddings.create(
            input=[query],
            model=self.deployment_name,
            dimensions=self._dimension
        )
        
        return response.data[0].embedding
    
    @property
    def embedding_dimension(self) -> int:
        return self._dimension
