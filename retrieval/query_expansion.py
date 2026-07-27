"""Query expansion and reformulation for better retrieval"""

import logging
from typing import List
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries using LLM for better retrieval coverage"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
    
    def expand_query(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate query variations for better recall"""
        
        prompt = f"""Generate {num_variations} alternative phrasings of this query for document search.
Keep the same meaning but use different words and structures.

Original query: {query}

Return only the variations, one per line, without numbering."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200
            )
            
            variations = [query]  # Include original
            content = response.choices[0].message.content.strip()
            variations.extend([line.strip() for line in content.split('\n') if line.strip()])
            
            return variations[:num_variations + 1]
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]
    
    def decompose_query(self, query: str) -> List[str]:
        """Break complex queries into sub-queries"""
        
        prompt = f"""Break this query into 2-3 simpler sub-queries that together answer the original question.

Query: {query}

Return only the sub-queries, one per line."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            
            content = response.choices[0].message.content.strip()
            sub_queries = [line.strip() for line in content.split('\n') if line.strip()]
            
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]
