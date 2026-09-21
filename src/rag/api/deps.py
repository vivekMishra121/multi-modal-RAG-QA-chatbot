"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Request

from rag.core.errors import ChatUnavailableError


def get_chatbot(request: Request):
    """Resolve the app-scoped chatbot instance (created in the lifespan)."""
    chatbot = request.app.state.chatbot
    if chatbot is None:
        error = request.app.state.chatbot_error or "Chatbot is not initialized"
        raise ChatUnavailableError(error)
    return chatbot