"""Application error model and FastAPI exception handlers."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error mapped to an HTTP response."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR

    def __init__(self, message: str, status_code: int | None = None, **details: Any):
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code or self.status_code or HTTPStatus.INTERNAL_SERVER_ERROR)
        self.details: dict[str, Any] = details


class ChatUnavailableError(AppError):
    """Raised when the chatbot cannot serve requests (503)."""

    def __init__(self, message: str = "Chatbot is not available"):
        super().__init__(message, status_code=HTTPStatus.SERVICE_UNAVAILABLE)


def _error_response(exc: AppError, request_id: str | None = None) -> JSONResponse:
    body: dict[str, Any] = {
        "error": exc.message,
        "status_code": exc.status_code,
    }
    if exc.details:
        body["details"] = exc.details
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=exc.status_code, content=body)


def register_exception_handlers(app: FastAPI) -> None:
    """Register handlers for AppError and unexpected exceptions."""

    @app.exception_handler(AppError)
    async def app_error_handler(request, exc: AppError):
        request_id = getattr(request.state, "request_id", None)
        logger.error("App error %s: %s", exc.status_code, exc.message)
        return _error_response(exc, request_id)

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        logger.exception("Unhandled exception: %s", exc)
        return _error_response(
            AppError(
                "Internal server error",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            ),
            request_id,
        )