"""POST /api/v1/chat handler (thin adapter over ChatService)."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from rag.services.chat_service import ChatService

from ..deps import get_chatbot
from ..schemas import ChatRequest, ChatResponse, Source

logger = logging.getLogger(__name__)

router = APIRouter()

Chatbot = Annotated[Any, Depends(get_chatbot)]


def _to_chat_response(result: dict[str, Any]) -> ChatResponse:
    sources = [Source(**s) for s in result.get("sources", [])]
    return ChatResponse(
        answer=result.get("answer"),
        sources=sources,
        success=result.get("success", False),
        error=result.get("error"),
        num_chunks=result.get("num_chunks", 0),
        strategy=result.get("strategy"),
        conversation_id=result.get("conversation_id"),
    )


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest, chatbot: Chatbot):
    """Answer a question over the indexed documents.

    ``conversation_id`` is required and scopes memory: prior turns are injected
    as conversation history and this turn is persisted.
    """
    service = ChatService(chatbot)
    result = await asyncio.to_thread(
        service.answer,
        request.question,
        request.filters,
        request.conversation_id,
        request.project_id,
    )

    logger.info(
        "Chat done (conversation_id=%s, success=%s, chunks=%s)",
        request.conversation_id,
        result.get("success"),
        result.get("num_chunks"),
    )
    return _to_chat_response(result)