"""Context window management for LLM input"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# Models not (yet) known to tiktoken fall back to a generic encoding so that a
# configurable model name (e.g. "gpt-5.1") never crashes the context manager.
_DEFAULT_ENCODING = "cl100k_base"


def _resolve_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(
            "Unknown tokenizer for model '%s'; falling back to %s",
            model_name,
            _DEFAULT_ENCODING,
        )
        return tiktoken.get_encoding(_DEFAULT_ENCODING)


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
        self.encoding = _resolve_encoding(model_name)
        logger.info("Initialized context manager (model=%s, max_tokens=%s)", model_name, max_tokens)
    
    def select_best_chunks(
        self,
        chunks_with_scores: list[tuple[dict[str, Any], float]],
        query: str,
        reserve_tokens: int = 1000
    ) -> list[dict[str, Any]]:
        """
        Select best chunks that fit within context window
        
        Args:
            chunks_with_scores: List of (chunk, score) tuples sorted by score
            query: User query
            reserve_tokens: Tokens to reserve for query + answer
            
        Returns:
            List of selected chunks
        """
        # Never reserve more than the budget minus a small floor so a too-large
        # (or misconfigured) reserve cannot starve retrieval of any context.
        reserve_tokens = max(0, min(reserve_tokens, self.max_tokens - 64))
        available_tokens = self.max_tokens - reserve_tokens
        query_tokens = len(self.encoding.encode(query))
        available_tokens -= query_tokens
        
        selected_chunks = []
        used_tokens = 0
        
        for chunk, _score in chunks_with_scores:
            chunk_text = chunk['content']
            chunk_tokens = len(self.encoding.encode(chunk_text))
            
            if used_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                used_tokens += chunk_tokens
            else:
                break
        
        logger.info(f"Selected {len(selected_chunks)} chunks ({used_tokens} tokens)")
        return selected_chunks
    
    def format_context(self, chunks: list[dict[str, Any]]) -> str:
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
