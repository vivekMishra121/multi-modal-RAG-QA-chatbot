"""FAISS-based vector store for indexing and storage"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for indexing and basic similarity search (Indexing component of RAG)"""
    
    def __init__(self, dimension: int, distance_metric: Literal['l2', 'cosine'] = 'cosine'):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding vector dimension
            distance_metric: 'l2' or 'cosine' (default: cosine for better semantic search)
        """
        self.dimension = dimension
        self.distance_metric = distance_metric
        
        # Use cosine similarity (inner product on normalized vectors) for better semantic search
        if distance_metric == 'cosine':
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.chunks: List[Dict[str, Any]] = []
        logger.info(f"Initialized FAISS vector store (dim={dimension}, metric={distance_metric})")
    
    def add_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]) -> None:
        """
        Add chunks with embeddings to the store
        
        Args:
            chunks_with_embeddings: List of dicts with 'embedding' and chunk data
        """
        if not chunks_with_embeddings:
            return
        
        embeddings = np.array([c['embedding'] for c in chunks_with_embeddings], dtype='float32')
        
        # Normalize for cosine similarity
        if self.distance_metric == 'cosine':
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks_with_embeddings)
        
        logger.info(f"Added {len(chunks_with_embeddings)} chunks (total: {len(self.chunks)})")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Standard similarity search
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of (chunk, score) tuples
        """
        if len(self.chunks) == 0:
            return []
        
        if filters:
            filtered_indices = self._filter_chunks(filters)
            if not filtered_indices:
                return []
            return self._search_filtered(query_embedding, top_k, filtered_indices)
        
        query_vec = np.array([query_embedding], dtype='float32')
        if self.distance_metric == 'cosine':
            faiss.normalize_L2(query_vec)
        
        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))
        
        return [(self.chunks[idx], float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def _filter_chunks(self, filters: Dict[str, Any]) -> List[int]:
        """Filter chunks by metadata"""
        indices = []
        for idx, chunk in enumerate(self.chunks):
            metadata = chunk.get('metadata', {})
            if all(metadata.get(k) == v for k, v in filters.items()):
                indices.append(idx)
        return indices
    
    def _search_filtered(
        self,
        query_embedding: List[float],
        top_k: int,
        valid_indices: List[int]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search within filtered indices"""
        query_vec = np.array([query_embedding], dtype='float32')
        valid_embeddings = np.array([self.chunks[i]['embedding'] for i in valid_indices], dtype='float32')
        
        if self.distance_metric == 'cosine':
            temp_index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(query_vec)
            faiss.normalize_L2(valid_embeddings)
        else:
            temp_index = faiss.IndexFlatL2(self.dimension)
        
        temp_index.add(valid_embeddings)
        scores, indices = temp_index.search(query_vec, min(top_k, len(valid_indices)))
        
        return [(self.chunks[valid_indices[idx]], float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def save(self, path: str) -> None:
        """
        Save vector store to disk
        
        Args:
            path: Directory path to save the store
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        # Save chunks metadata
        with open(save_path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save config
        with open(save_path / "config.pkl", 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'distance_metric': self.distance_metric
            }, f)
        
        logger.info(f"Saved vector store to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """
        Load vector store from disk
        
        Args:
            path: Directory path to load from
            
        Returns:
            Loaded VectorStore instance
        """
        load_path = Path(path)
        
        # Load config
        with open(load_path / "config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        # Create instance
        store = cls(
            dimension=config['dimension'],
            distance_metric=config.get('distance_metric', 'l2')
        )
        
        # Load FAISS index
        store.index = faiss.read_index(str(load_path / "index.faiss"))
        
        # Load chunks
        with open(load_path / "chunks.pkl", 'rb') as f:
            store.chunks = pickle.load(f)
        
        logger.info(f"Loaded vector store from {path} ({len(store.chunks)} chunks)")
        return store
    
    def __len__(self) -> int:
        """Return number of chunks in store"""
        return len(self.chunks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        chunk_types = {}
        for chunk in self.chunks:
            ctype = chunk.get('chunk_type', 'unknown')
            chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'chunk_types': chunk_types
        }
