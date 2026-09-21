"""Query expansion and reformulation for better retrieval (provider-agnostic)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expand queries using the configured LLM provider for better recall."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        from rag.core.config import get_settings

        settings = get_settings()
        llm_cfg = settings.llm
        azure = settings.azure

        self.api_key = api_key or llm_cfg.api_key
        self.model_name = model_name or (
            "gpt-4.1-nano"
            if llm_cfg.provider == "azure_openai"
            else llm_cfg.model_name
        )
        self.client: Any

        if llm_cfg.provider == "azure_openai":
            from openai import AzureOpenAI

            if not azure.openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT required for azure_openai provider")
            self.client = AzureOpenAI(
                azure_endpoint=azure.openai_endpoint,
                api_key=self.api_key or azure.openai_api_key,
                api_version=azure.openai_api_version,
                max_retries=llm_cfg.max_retries,
                timeout=llm_cfg.request_timeout,
            )
        else:
            from openai import OpenAI

            if not self.api_key:
                raise ValueError("OpenAI API key required for query expansion")
            self.client = OpenAI(
                api_key=self.api_key,
                max_retries=llm_cfg.max_retries,
                timeout=llm_cfg.request_timeout,
            )

        logger.info("Initialized QueryExpander (model=%s)", self.model_name)

    def expand_query(self, query: str, num_variations: int = 3) -> list[str]:
        """Generate query variations for better recall.

        Always returns a list containing at least ``query`` (the original) even
        when the LLM call fails.
        """
        prompt = f"""Generate {num_variations} alternative phrasings of this query for document search.
Keep the same meaning but use different words and structures.

Original query: {query}

Return only the variations, one per line, without numbering."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            content = response.choices[0].message.content.strip()
            variations = [query]
            variations.extend([line.strip() for line in content.split('\n') if line.strip()])
            return variations[: num_variations + 1]
        except Exception as e:  # noqa: BLE001 - degradation is non-fatal
            logger.warning("Query expansion failed: %s", e)
            return [query]

    def decompose_query(self, query: str) -> list[str]:
        """Break complex queries into sub-queries."""
        prompt = f"""Break this query into 2-3 simpler sub-queries that together answer the original question.

Query: {query}

Return only the sub-queries, one per line."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            sub_queries = [line.strip() for line in content.split('\n') if line.strip()]
            return sub_queries if sub_queries else [query]
        except Exception as e:  # noqa: BLE001 - degradation is non-fatal
            logger.warning("Query decomposition failed: %s", e)
            return [query]

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self.client.close()