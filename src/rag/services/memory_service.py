"""Memory and project management use cases backed by the Cosmos store.

Thin, validated operations over conversation history and project records so the
presentation routes stay small and logic is reusable from other entrypoints.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from rag.core.errors import AppError

# Metadata keys the vector stores can actually filter on.
FILTERABLE_KEYS = ("file_name", "chunk_type", "page")


class MemoryService:
    """Use-case layer over a Cosmos memory store."""

    def __init__(self, store: Any) -> None:
        self._store = store

    @staticmethod
    def _slice_filters(filters: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Keep only filterable metadata keys."""
        if not filters:
            return {}
        return {k: v for k, v in filters.items() if k in FILTERABLE_KEYS}

    # ------------------------------------------------------------ conversations

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        messages = self._store.get_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "total_messages": len(messages),
        }

    def delete_conversation(self, conversation_id: str) -> dict[str, Any]:
        removed = self._store.delete_conversation(conversation_id)
        return {
            "conversation_id": conversation_id,
            "messages_deleted": removed,
        }

    # ----------------------------------------------------------------- projects

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not str(name).strip():
            raise AppError("name must not be empty", status_code=422)
        pid = project_id or "proj-" + uuid.uuid4().hex
        return self._store.create_project(
            project_id=pid,
            name=str(name).strip(),
            description=description,
            tags=tags or [],
            filters=self._slice_filters(filters),
        )

    def list_projects(self) -> dict[str, Any]:
        items = self._store.list_projects()
        return {"projects": items, "total_projects": len(items)}

    def get_project(self, project_id: str) -> dict[str, Any]:
        project = self._store.get_project(project_id)
        if project is None:
            raise AppError(f"Project not found: {project_id}", status_code=404)
        return project

    def update_project(self, project_id: str, **fields: Any) -> dict[str, Any]:
        if "filters" in fields and fields["filters"] is not None:
            fields["filters"] = self._slice_filters(fields["filters"])
        project = self._store.update_project(project_id, **fields)
        if project is None:
            raise AppError(f"Project not found: {project_id}", status_code=404)
        return project

    def delete_project(self, project_id: str) -> dict[str, Any]:
        deleted = self._store.delete_project(project_id)
        if not deleted:
            raise AppError(f"Project not found: {project_id}", status_code=404)
        return {"project_id": project_id, "deleted": True}

    # ------------------------------------------------------------------- files

    def list_files(self, project_id: Optional[str] = None) -> dict[str, Any]:
        items = self._store.list_files(project_id=project_id)
        return {"files": items, "total_files": len(items)}