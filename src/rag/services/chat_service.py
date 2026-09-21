"""Chat use case: validate a question and dispatch it to the RAG engine."""

from __future__ import annotations

from typing import Any, Optional

from rag.core.errors import AppError


class ChatService:
    """Thin use-case wrapper around :class:`rag.chatbot.RAGChatbot`."""

    def __init__(self, chatbot: Any) -> None:
        self._chatbot = chatbot

    def answer(
        self,
        question: str,
        filters: Optional[dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Answer a question, returning the engine result dict unchanged."""
        question = (question or "").strip()
        if not question:
            raise AppError("question must not be empty", status_code=422)
        if not conversation_id or not str(conversation_id).strip():
            raise AppError("conversation_id is required", status_code=422)
        return self._chatbot.chat(
            question,
            filters=filters,
            conversation_id=conversation_id,
            project_id=project_id,
        )