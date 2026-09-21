"""FastAPI application factory for the RAG QA chatbot service.

Run with::
    uvicorn rag.api.main:app --reload

Entrypoints:
    GET  /health            liveness/readiness (+ memory/blob status)
    POST /api/v1/chat       answer a question (JSON, requires conversation_id)
    POST /api/v1/index      build/re-index documents (admin)
    POST /api/v1/documents/upload   upload PDF/DOCX/TXT documents
    GET  /api/v1/documents          list indexed documents
    GET  /api/v1/documents/files    list blob-backed file records
    GET  /api/v1/documents/{file_name}/download  download the original file
    DELETE /api/v1/documents/{file_name}  remove one document's chunks
    GET/DELETE /api/v1/conversations/{conversation_id}  history read/delete
    POST/GET /api/v1/projects        project CRUD
    GET/PATCH/DELETE /api/v1/projects/{project_id}
    GET  /docs              OpenAPI documentation
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rag.api.routes import admin, chat, documents, health, memory
from rag.core.errors import register_exception_handlers
from rag.core.logging import configure_logging

if TYPE_CHECKING:
    from rag.core.config import Settings

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):  # pragma: no cover - infra
    """Assign a request id and log an access record for every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id

        start = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = (time.monotonic() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            extra={"request_id": request_id, "latency_ms": round(duration_ms, 2)},
        )
        return response


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    from rag.core.config import get_settings

    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from rag.chatbot import RAGChatbot

        app.state.settings = settings
        app.state.chatbot = None
        app.state.chatbot_error = None
        try:
            app.state.chatbot = RAGChatbot(settings=settings)
        except Exception as e:  # noqa: BLE001 - report degraded, don't crash
            app.state.chatbot = None
            app.state.chatbot_error = str(e)
            logger.warning("Chatbot initialization deferred/failed: %s", e)
        yield
        if app.state.chatbot is not None:
            try:
                app.state.chatbot.close()
            except Exception:
                logger.warning("Chatbot close failed during shutdown", exc_info=True)

    app = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Outermost middleware: request id + access log.
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)

    # Routers
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(admin.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(memory.router, prefix="/api/v1")
    app.include_router(health.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "app": settings.app_name,
            "version": settings.api.version,
            "docs": "/docs",
        }

    @app.exception_handler(404)
    async def not_found(request: Request, exc):  # pragma: no cover
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = app.state.settings
    uvicorn.run("rag.api.main:app", host=settings.api.host, port=settings.api.port)