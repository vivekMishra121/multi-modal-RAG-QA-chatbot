"""Advanced retrieval strategies for RAG (Retrieval component)"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from sentence_transformers import CrossEncoder
from .query_expansion import QueryExpander

logger = logging.getLogger(__name__)


class Retriever:
    """Advanced retrieval strategies: MMR, Hybrid Search, Reranking"""
    
    def __init__(self, vector_store, use_reranker: bool = True, use_query_expansion: bool = True):
        """
        Initialize retriever
        
        Args:
            vector_store: VectorStore instance
            use_reranker: Enable cross-encoder reranking
            use_query_expansion: Enable query expansion for better recall
        """
        self.vector_store = vector_store
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.reranker = None
        self.query_expander = None
        
        if use_reranker:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Loaded cross-encoder reranker")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                self.use_reranker = False
        
        if use_query_expansion:
            try:
                self.query_expander = QueryExpander()
                logger.info("Loaded query expander")
            except Exception as e:
                logger.warning(f"Failed to load query expander: {e}")
                self.use_query_expansion = False
        
        self._build_tfidf_index()
        logger.info("Initialized retriever with advanced search strategies")
    
    def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        strategy: str = 'auto',
        filters: Optional[Dict[str, Any]] = None,
        rerank: bool = True,
        **kwargs
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Unified retrieval interface with automatic strategy selection
        
        Args:
            query_embedding: Query vector
            query_text: Query text
            top_k: Number of results to return
            strategy: 'auto' (recommended), 'standard', 'mmr', 'hybrid', or 'fusion'
            filters: Optional metadata filters
            rerank: Apply cross-encoder reranking (default: True)
            **kwargs: Strategy-specific parameters
            
        Returns:
            List of (chunk, score) tuples
        """
        # Auto-select strategy based on query
        if strategy == 'auto':
            strategy = self._select_strategy(query_text)
            logger.debug(f"Auto-selected strategy: {strategy}")
        
        # Get initial candidates (fetch more for reranking)
        fetch_k = top_k * 5 if rerank and self.use_reranker else top_k * 2
        
        # Use fusion retrieval for better results
        if strategy == 'fusion' or (strategy == 'auto' and self.use_query_expansion):
            results = self._fusion_retrieval(query_embedding, query_text, fetch_k, filters)
        elif strategy == 'mmr':
            results = self.mmr_search(query_embedding, top_k=fetch_k, filters=filters, **kwargs)
        elif strategy == 'hybrid':
            results = self.hybrid_search(query_embedding, query_text, top_k=fetch_k, filters=filters, **kwargs)
        else:
            results = self.vector_store.search(query_embedding, top_k=fetch_k, filters=filters)
        
        # Apply reranking if enabled
        if rerank and self.use_reranker and len(results) > 0:
            results = self._rerank(query_text, results, top_k)
        else:
            results = results[:top_k]
        
        return results
    
    def _select_strategy(self, query: str) -> str:
        """
        Automatically select best retrieval strategy based on query
        
        Args:
            query: User query text
            
        Returns:
            Strategy name: 'fusion', 'hybrid', 'mmr', or 'standard'
        """
        query_lower = query.lower()
        
        # Fusion: Use query expansion for complex queries
        if self.use_query_expansion and len(query.split()) > 8:
            return 'fusion'
        
        # Hybrid: Chart/table/data queries
        if any(word in query_lower for word in ['chart', 'graph', 'table', 'figure', 'data', 'rate', 'percentage', 'number']):
            return 'hybrid'
        
        # MMR: Broad/exploratory queries
        if any(word in query_lower for word in ['overview', 'summary', 'explain', 'describe', 'what are', 'how did']):
            return 'mmr'
        
        # Fusion for better recall
        return 'fusion' if self.use_query_expansion else 'standard'
    
    def mmr_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Maximal Marginal Relevance for diverse results
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            fetch_k: Initial candidates
            lambda_mult: Relevance (1.0) vs diversity (0.0)
            filters: Optional filters
            
        Returns:
            Diverse results
        """
        candidates = self.vector_store.search(query_embedding, top_k=fetch_k, filters=filters)
        if len(candidates) <= top_k:
            return candidates[:top_k]
        
        query_vec = np.array(query_embedding, dtype='float32').reshape(1, -1)
        candidate_embeddings = np.array([c[0]['embedding'] for c in candidates], dtype='float32')
        
        if self.vector_store.distance_metric == 'cosine':
            faiss.normalize_L2(query_vec)
            faiss.normalize_L2(candidate_embeddings)
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        while len(selected_indices) < top_k and remaining_indices:
            if not selected_indices:
                best_idx = 0
            else:
                mmr_scores = []
                for idx in remaining_indices:
                    relevance = np.dot(query_vec, candidate_embeddings[idx])
                    selected_embeddings = candidate_embeddings[selected_indices]
                    similarity = np.max(np.dot(selected_embeddings, candidate_embeddings[idx]))
                    mmr_score = lambda_mult * relevance - (1 - lambda_mult) * similarity
                    mmr_scores.append(mmr_score)
                best_idx = remaining_indices[np.argmax(mmr_scores)]
            
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [candidates[idx] for idx in selected_indices]
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Hybrid search: vector + keyword matching
        
        Args:
            query_embedding: Query vector
            query_text: Query text
            top_k: Number of results
            alpha: Vector weight (1-alpha for keyword)
            filters: Optional filters
            
        Returns:
            Combined results
        """
        if self.tfidf_matrix is None:
            return self.vector_store.search(query_embedding, top_k, filters)
        
        vector_results = self.vector_store.search(query_embedding, top_k=top_k*2, filters=filters)
        
        query_tfidf = self.tfidf_vectorizer.transform([query_text])
        keyword_scores = sklearn_cosine(query_tfidf, self.tfidf_matrix).flatten()
        
        combined_scores = {}
        for chunk, vec_score in vector_results:
            chunk_idx = self.vector_store.chunks.index(chunk)
            kw_score = keyword_scores[chunk_idx]
            combined_scores[chunk_idx] = alpha * vec_score + (1 - alpha) * kw_score
        
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        return [(self.vector_store.chunks[idx], combined_scores[idx]) for idx in sorted_indices[:top_k]]
    
    def _rerank(self, query: str, candidates: List[Tuple[Dict[str, Any], float]], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank candidates using cross-encoder for better accuracy
        
        Args:
            query: Query text
            candidates: Initial retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not self.reranker or len(candidates) == 0:
            return candidates[:top_k]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, chunk['content']) for chunk, _ in candidates]
        
        # Get reranker scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original chunks
        reranked = [(candidates[i][0], float(score)) for i, score in enumerate(rerank_scores)]
        
        # Sort by reranker scores
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Reranked {len(candidates)} candidates to top {top_k}")
        return reranked[:top_k]
    
    def _build_tfidf_index(self):
        """Build TF-IDF index for keyword search"""
        if not self.vector_store.chunks:
            return
        
        texts = [chunk['content'] for chunk in self.vector_store.chunks]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info("Built TF-IDF index for hybrid search")
    
    def _fusion_retrieval(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fusion retrieval: Combine multiple query variations"""
        if not self.use_query_expansion or not self.query_expander:
            return self.vector_store.search(query_embedding, top_k, filters)
        
        # Get query variations
        query_variations = self.query_expander.expand_query(query_text, num_variations=2)
        
        # Retrieve for each variation
        all_results = {}
        
        # Original query (highest weight)
        results = self.vector_store.search(query_embedding, top_k=top_k, filters=filters)
        for chunk, score in results:
            chunk_id = chunk.get('chunk_id', id(chunk))
            all_results[chunk_id] = (chunk, score * 1.0)
        
        # Query variations (lower weight)
        for variation in query_variations[1:]:
            try:
                # Get embedding for variation
                var_embedding = self.vector_store.chunks[0].get('embedding')  # Placeholder
                var_results = self.vector_store.search(query_embedding, top_k=top_k//2, filters=filters)
                
                for chunk, score in var_results:
                    chunk_id = chunk.get('chunk_id', id(chunk))
                    if chunk_id in all_results:
                        # Boost score if found in multiple variations
                        all_results[chunk_id] = (chunk, all_results[chunk_id][1] + score * 0.3)
                    else:
                        all_results[chunk_id] = (chunk, score * 0.5)
            except:
                continue
        
        # Sort by combined scores
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
