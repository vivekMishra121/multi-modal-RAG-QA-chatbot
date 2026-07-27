"""Multi-modal chunking and embedding pipeline"""

import logging
import os
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from .base import Chunk, ChunkType
from .chunkers import TextChunker, TableChunker, ImageChunker
from .embedder import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


@dataclass
class ChunkWithEmbedding:
    """Chunk with its embedding vector"""
    chunk: Chunk
    embedding: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chunk_id': self.chunk.chunk_id,
            'content': self.chunk.content,
            'chunk_type': self.chunk.chunk_type.value,
            'metadata': self.chunk.metadata,
            'embedding': self.embedding
        }


class MultiModalChunkingPipeline:
    """Orchestrates chunking and embedding for multi-modal content"""
    
    def __init__(
        self,
        text_chunk_size: int = 1000,
        text_chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-large",
        use_openai: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize pipeline
        
        Args:
            text_chunk_size: Size of text chunks
            text_chunk_overlap: Overlap between text chunks
            embedding_model: Name of embedding model
            use_openai: Use OpenAI embeddings (recommended)
            batch_size: Batch size for embedding generation
        """
        self.text_chunker = TextChunker(text_chunk_size, text_chunk_overlap)
        self.table_chunker = TableChunker()
        self.image_chunker = ImageChunker()
        
        # Use OpenAI embeddings for better quality
        if use_openai:
            from .openai_embedder import OpenAIEmbedder
            from .azure_embedder import AzureOpenAIEmbedder
            
            # Check if Azure or OpenAI
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                self.embedder = AzureOpenAIEmbedder(deployment_name=embedding_model, dimensions=1536)
            else:
                self.embedder = OpenAIEmbedder(model_name=embedding_model, dimensions=1536)
        else:
            self.embedder = SentenceTransformerEmbedder(embedding_model, batch_size)
        
        logger.info(f"Initialized multi-modal chunking pipeline (OpenAI={use_openai})")
    
    def process_document(self, document: Dict[str, Any]) -> List[ChunkWithEmbedding]:
        """
        Process a single document into chunks with embeddings
        
        Args:
            document: Document dict from ingestion pipeline
            
        Returns:
            List of chunks with embeddings
        """
        if 'error' in document:
            logger.warning(f"Skipping document with error: {document.get('file_path')}")
            return []
        
        content = document.get('content', {})
        file_name = content.get('metadata', {}).get('file_name') or Path(document.get('file_path', 'unknown')).name
        base_metadata = {
            'source': document.get('file_path'),
            'file_name': file_name,
            'total_pages': content.get('metadata', {}).get('pages', 0)
        }
        
        # Chunk all content types
        chunks = []
        chunks.extend(self._chunk_text(content.get('text', ''), base_metadata))
        chunks.extend(self._chunk_tables(content.get('tables', []), base_metadata))
        chunks.extend(self._chunk_images(content.get('images', []), base_metadata))
        
        logger.info(f"Created {len(chunks)} chunks from {file_name}")
        
        # Generate embeddings
        return self._embed_chunks(chunks)
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[ChunkWithEmbedding]:
        """
        Process multiple documents
        
        Args:
            documents: List of document dicts from ingestion pipeline
            
        Returns:
            List of all chunks with embeddings
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text content"""
        if not text:
            return []
        return self.text_chunker.chunk(text, metadata)
    
    def _chunk_tables(self, tables: List[Dict], metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk table content"""
        chunks = []
        for table in tables:
            table_metadata = {
                **metadata,
                'page': table.get('page')
            }
            chunks.extend(self.table_chunker.chunk(table, table_metadata))
        return chunks
    
    def _chunk_images(self, images: List[Dict], metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk image content"""
        chunks = []
        for image in images:
            image_metadata = {
                **metadata,
                'page': image.get('page')
            }
            chunks.extend(self.image_chunker.chunk(image, image_metadata))
        return chunks
    
    def _embed_chunks(self, chunks: List[Chunk]) -> List[ChunkWithEmbedding]:
        """Generate embeddings for chunks"""
        if not chunks:
            return []
        
        # Extract content for embedding
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.embedder.embed(texts)
        
        # Combine chunks with embeddings
        return [
            ChunkWithEmbedding(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedder.embedding_dimension
