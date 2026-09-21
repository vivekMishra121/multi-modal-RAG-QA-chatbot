"""POST /api/v1/index handler (admin), thin over IndexService."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Request

from rag.services.index_service import IndexService

from ..schemas import IndexRequest, IndexResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/index", response_model=IndexResponse, tags=["index"])
async def index_documents(request: IndexRequest, req: Request):
    """Build/re-index a document path.

    This is an administrative surface (arbitrary local paths). Restrict it to
    trusted callers in production, e.g. via network policy or API auth.
    """
    settings = req.app.state.settings
    service = IndexService(settings)

    summary = await asyncio.to_thread(
        service.build,
        request.document_path,
        request.recreate,
    )

    # Reload the serving chatbot so it sees the fresh index without a restart.
    IndexService.reload_chatbot(req.app, settings)
    return IndexResponse(**summary)