"""Index administration use case: build/re-index a document path."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag.chatbot import RAGChatbot, build_index
from rag.core.errors import AppError

logger = logging.getLogger(__name__)


class IndexService:
    """Bridges the admin surface to the indexing engine."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    def build(self, document_path: str, recreate: bool = True) -> dict[str, Any]:
        """Ingest ``document_path`` (file or directory) into the index."""
        path = Path(document_path).expanduser()
        if not path.exists():
            raise AppError(f"Path does not exist: {path}", status_code=404)

        summary = build_index(
            str(path),
            store_path=None,
            settings=self._settings,
            recreate=recreate,
        )
        return summary

    @staticmethod
    def reload_chatbot(app: Any, settings: Any) -> None:
        """Swap the serving chatbot so the fresh index is live without a restart."""
        old = getattr(app.state, "chatbot", None)
        if old is not None:
            try:
                old.close()
            except Exception:  # noqa: BLE001
                logger.warning("Failed to close previous chatbot instance")

        app.state.chatbot_error = None
        try:
            app.state.chatbot = RAGChatbot(settings=settings)
        except Exception as e:  # noqa: BLE001
            app.state.chatbot = None
            app.state.chatbot_error = str(e)
            logger.warning("Chatbot reload failed after indexing: %s", e)