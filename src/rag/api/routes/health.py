"""GET /health handler."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ..schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health(request: Request):
    """Liveness + readiness: report chatbot/store status without failing."""
    app = request.app
    settings = app.state.settings

    chatbot = app.state.chatbot

    backend: str | None = None
    index_size = 0
    if chatbot is not None:
        try:
            backend = chatbot.store.get_stats().get("backend")
            index_size = len(chatbot.store)
        except Exception:  # noqa: BLE001 - health must never raise
            index_size = -1

    return HealthResponse(
        status="ok" if chatbot is not None else "degraded",
        app=settings.app_name,
        version=settings.api.version,
        environment=settings.environment,
        chatbot_ready=chatbot is not None,
        vector_store_backend=backend,
        index_size=index_size,
        memory_backend="cosmos-nosql" if chatbot is not None else None,
        error=app.state.chatbot_error,
    )