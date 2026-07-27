"""Context window management for LLM input"""

import logging
from typing import List, Dict, Any, Tuple
import tiktoken

logger = logging.getLogger(__name__)


class ContextManager:
    """Manages context window to fit within LLM token limits"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_tokens: int = 4096):
        """
        Initialize context manager
        
        Args:
            model_name: LLM model name for tokenization
            max_tokens: Maximum context window size
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model_name)
        logger.info(f"Initialized context manager (model={model_name}, max_tokens={max_tokens})")
    
    def select_best_chunks(
        self,
        chunks_with_scores: List[Tuple[Dict[str, Any], float]],
        query: str,
        reserve_tokens: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Select best chunks that fit within context window
        
        Args:
            chunks_with_scores: List of (chunk, score) tuples sorted by score
            query: User query
            reserve_tokens: Tokens to reserve for query + answer
            
        Returns:
            List of selected chunks
        """
        available_tokens = self.max_tokens - reserve_tokens
        query_tokens = len(self.encoding.encode(query))
        available_tokens -= query_tokens
        
        selected_chunks = []
        used_tokens = 0
        
        for chunk, score in chunks_with_scores:
            chunk_text = chunk['content']
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            if used_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk_tokens
            else:
                break
        
        logger.info(f"Selected {len(selected_chunks)} chunks ({used_tokens} tokens)")
        return selected_chunks
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format chunks into context string with citations
        
        Args:
            chunks: List of chunk dicts
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'N/A')
            chunk_type = chunk.get('chunk_type', 'text')
            
            citation = f"[Source {i}: {source}, Page {page}, Type: {chunk_type}]"
            content = chunk['content']
            
            context_parts.append(f"{citation}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
