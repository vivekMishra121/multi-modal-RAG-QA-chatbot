"""LangChain QA chain for answer generation (provider-agnostic)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class QAChain:
    """Provider-agnostic QA chain configured through application settings."""

    DEFAULT_PROMPT = """You are a helpful AI assistant answering questions based on provided context.

Context:
{context}

{history_section}Question: {question}

Instructions:
1. Answer the question using ONLY the information from the context above
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Include citations in your answer using [Source X] format
4. Be concise and accurate

Answer:"""

    @staticmethod
    def format_history(history: Optional[list[dict[str, Any]]]) -> str:
        """Render prior conversation turns for prompt injection.

        Returns an empty string when there is no history so the prompt simply
        omits the history block.
        """
        if not history:
            return ""
        lines = ["Previous conversation:"]
        for turn in history:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        lines.append("")
        return "\n".join(lines)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        custom_prompt: Optional[str] = None,
    ):
        """
        Builds the LLM from ``Settings`` so the provider (OpenAI, Azure OpenAI,
        Groq) and retries/timeouts are driven entirely by configuration.
        """
        from rag.core.config import get_settings

        settings = get_settings()
        llm_cfg = settings.llm
        azure = settings.azure

        self.api_key = api_key or llm_cfg.api_key
        self.model_name = model_name or (
            llm_cfg.deployment_name or azure.chat_deployment
            if llm_cfg.provider == "azure_openai"
            else llm_cfg.model_name
        )
        self.temperature = llm_cfg.temperature if temperature is None else temperature

        provider = llm_cfg.provider
        if self.api_key is None and provider == "openai":
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY)")

        self.llm: Any

        if provider == "azure_openai":
            from langchain_openai import AzureChatOpenAI

            if not azure.openai_endpoint:
                raise ValueError("AZURE_OPENAI_ENDPOINT required for azure_openai provider")
            self.llm = AzureChatOpenAI(
                azure_endpoint=azure.openai_endpoint,
                azure_deployment=self.model_name,
                api_key=self.api_key or azure.openai_api_key,  # type: ignore[arg-type]
                api_version=azure.openai_api_version,
                max_retries=llm_cfg.max_retries,
                timeout=llm_cfg.request_timeout,
                streaming=llm_cfg.streaming,
            )
        else:
            from langchain_openai import ChatOpenAI

            if provider == "groq" and not llm_cfg.base_url:
                raise ValueError("LLM_GROQ_BASE_URL required for groq provider")
            self.llm = ChatOpenAI(
                api_key=self.api_key,  # type: ignore[arg-type]
                model_name=self.model_name,  # type: ignore[call-arg]
                temperature=self.temperature,
                max_retries=llm_cfg.max_retries,
                timeout=llm_cfg.request_timeout,
                streaming=llm_cfg.streaming,
                base_url=llm_cfg.base_url,
            )

        self.prompt = PromptTemplate(
            template=custom_prompt or self.DEFAULT_PROMPT,
            input_variables=[
                v for v in ("context", "question", "history_section")
                if v in (custom_prompt or self.DEFAULT_PROMPT)
            ],
            validate_template=False,
        )
        logger.info("Initialized QA chain (provider=%s, model=%s)", provider, self.model_name)

    def generate_answer(
        self,
        question: str,
        context: str,
        history: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        """Generate an answer from question + context (+ prior turns).

        Returns:
            ``{'answer': str | None, 'success': bool, 'error': str | None}``.
        """
        try:
            payload: dict[str, Any] = {"question": question, "context": context}
            if "history_section" in self.prompt.input_variables:
                payload["history_section"] = self.format_history(history)
            prompt_text = self.prompt.format(**payload)
            messages = [{"role": "user", "content": prompt_text}]
            response = self.llm.invoke(messages)
            answer = response.content.strip()

            return {'answer': answer, 'success': True, 'error': None}

        except Exception as e:  # noqa: BLE001 - surfaced to the caller
            logger.error("Answer generation failed: %s", e)
            return {'answer': None, 'success': False, 'error': str(e)}