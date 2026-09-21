"""Azure Cosmos DB (NoSQL API) store for chat memory, project metadata, and
file records.

Holds three containers:

  * ``conversations``  - one item per chat message, keyed by conversation_id.
  * ``projects``       - project registry (metadata + retrieval filters).
  * ``files``          - indexed-document records linking ``file_name`` to the
                         blob that holds the original upload.

Cosmos is a required dependency: constructing the store raises ``ValueError``
when credentials are missing. Tests inject a fake client/containers so no
network or emulator is needed.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MESSAGE = "message"
_PROJECT = "project"
_FILE = "file"


def _message_id() -> str:
    return uuid.uuid4().hex


def _file_id(file_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"file:{file_name}"))


def _now() -> dict[str, float]:
    return {"created_at": time.time(), "updated_at": time.time()}


class CosmosNoSQLStore:
    """Cosmos-backed store implementing the memory/project/file contract."""

    def __init__(
        self,
        settings: Any,
        client: Optional[Any] = None,
        database: Optional[Any] = None,
        containers: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            settings: App settings (reads ``settings.cosmos``).
            client: Optional pre-built CosmosClient (tests).
            database: Optional pre-built database proxy (tests).
            containers: Optional pre-built container proxies keyed by role
                (``conversations``/``projects``/``files``) (tests).
        """
        cfg = settings.cosmos
        if not cfg.is_configured and client is None and database is None:
            raise ValueError(
                "Cosmos DB is not configured (COSMOS_ENDPOINT/COSMOS_KEY). "
                "Memory and project metadata cannot be enabled without it."
            )

        self.settings = settings
        self._owns_client = client is not None and database is None
        self._client = client
        self.database = database

        if containers is not None:
            self.conversations = containers["conversations"]
            self.projects = containers["projects"]
            self.files = containers["files"]
            return

        if self._client is None:
            from azure.cosmos import CosmosClient

            self._client = CosmosClient(cfg.endpoint, credential=cfg.key)

        if self.database is None:
            self.database = self._client.create_database_if_not_exists(cfg.database_name)

        self.conversations = self.database.create_container_if_not_exists(
            id=cfg.conversations_container, partition_key={"path": "/conversation_id"}
        )
        self.projects = self.database.create_container_if_not_exists(
            id=cfg.projects_container, partition_key={"path": "/project_id"}
        )
        self.files = self.database.create_container_if_not_exists(
            id=cfg.files_container, partition_key={"path": "/file_name"}
        )
        logger.info(
            "Initialized Cosmos store (db=%s, history=%s)",
            cfg.database_name,
            cfg.max_history_turns,
        )

    # ----------------------------------------------------------------- memory

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict[str, Any]]] = None,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Persist one conversation turn; returns the stored item."""
        item = {
            "id": _message_id(),
            "type": _MESSAGE,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "project_id": project_id,
            "created_at": time.time(),
        }
        self.conversations.upsert_item(item)
        return item

    def get_conversation(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Return the last ``limit`` turns (oldest-first) for a conversation."""
        cap = limit if limit is not None else self.settings.cosmos.max_history_turns
        # TOP applies after ORDER BY, so request the most recent N (newest
        # first in Cosmos) and reverse into oldest->newest for the prompt.
        query = (
            "SELECT TOP @limit * FROM c "
            "WHERE c.type = @kind ORDER BY c.created_at DESC"
        )
        items = list(
            self.conversations.query_items(
                query=query,
                parameters=[
                    {"name": "@limit", "value": cap},
                    {"name": "@kind", "value": _MESSAGE},
                ],
                partition_key=conversation_id,
            )
        )
        items.reverse()
        return [
            {
                "role": i.get("role", ""),
                "content": i.get("content", ""),
                "sources": i.get("sources") or [],
                "project_id": i.get("project_id"),
                "created_at": i.get("created_at"),
            }
            for i in items
        ]

    def delete_conversation(self, conversation_id: str) -> int:
        """Delete every message in a conversation; returns messages removed."""
        messages = list(
            self.conversations.query_items(
                query="SELECT * FROM c WHERE c.type = @kind",
                parameters=[{"name": "@kind", "value": _MESSAGE}],
                partition_key=conversation_id,
            )
        )
        for item in messages:
            try:
                self.conversations.delete_item(item["id"], partition_key=conversation_id)
            except Exception:  # noqa: BLE001 - best-effort removal
                logger.debug("Failed to delete message %s", item.get("id"))
        return len(messages)

    # ---------------------------------------------------------------- projects

    def create_project(
        self,
        project_id: str,
        name: str,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        filters: Optional[dict[str, Any]] = None,
        file_names: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        item = {
            "id": project_id,
            "type": _PROJECT,
            "project_id": project_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "filters": filters or {},
            "file_names": file_names or [],
            **_now(),
        }
        self.projects.upsert_item(item)
        return item

    def get_project(self, project_id: str) -> Optional[dict[str, Any]]:
        try:
            item = self.projects.read_item(
                item=project_id, partition_key=project_id
            )
        except Exception:  # noqa: BLE001 - 404 for missing project
            return None
        if item.get("type") != _PROJECT:
            return None
        return self._strip_id(item)

    def list_projects(self) -> list[dict[str, Any]]:
        items = list(
            self.projects.query_items(
                query="SELECT * FROM c WHERE c.type = @kind ORDER BY c.name ASC",
                parameters=[{"name": "@kind", "value": _PROJECT}],
            )
        )
        return [self._strip_id(i) for i in items]

    def update_project(
        self, project_id: str, **fields: Any
    ) -> Optional[dict[str, Any]]:
        item = self.get_project_raw(project_id)
        if item is None:
            return None
        allowed = {"name", "description", "tags", "filters", "file_names"}
        for key, value in fields.items():
            if key in allowed and value is not None:
                item[key] = value
        item["updated_at"] = time.time()
        self.projects.upsert_item(item)
        return self._strip_id(item)

    def delete_project(self, project_id: str) -> bool:
        try:
            self.projects.delete_item(item=project_id, partition_key=project_id)
            return True
        except Exception:  # noqa: BLE001 - missing project
            return False

    def get_project_raw(self, project_id: str) -> Optional[dict[str, Any]]:
        try:
            item = self.projects.read_item(
                item=project_id, partition_key=project_id
            )
        except Exception:  # noqa: BLE001 - 404 for missing project
            return None
        return item if item.get("type") == _PROJECT else None

    @staticmethod
    def _strip_id(item: dict[str, Any]) -> dict[str, Any]:
        item = dict(item)
        item.pop("id", None)
        item.pop("_rid", None)
        item.pop("_self", None)
        item.pop("_etag", None)
        item.pop("_attachments", None)
        item.pop("_ts", None)
        return item

    # ------------------------------------------------------------------- files

    def upsert_file(
        self,
        file_name: str,
        blob_path: str,
        content_type: Optional[str] = None,
        size_bytes: Optional[int] = None,
        chunk_count: Optional[int] = None,
        project_id: Optional[str] = None,
    ) -> dict[str, Any]:
        item = {
            "id": _file_id(file_name),
            "type": _FILE,
            "file_name": file_name,
            "project_id": project_id,
            "blob_path": blob_path,
            "content_type": content_type,
            "size_bytes": size_bytes,
            "chunk_count": chunk_count or 0,
            **_now(),
        }
        self.files.upsert_item(item)
        return item

    def get_file(self, file_name: str) -> Optional[dict[str, Any]]:
        try:
            item = self.files.read_item(
                item=_file_id(file_name), partition_key=file_name
            )
        except Exception:  # noqa: BLE001 - missing record
            return None
        return self._strip_id(item)

    def list_files(self, project_id: Optional[str] = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM c WHERE c.type = @kind"
        params = [{"name": "@kind", "value": _FILE}]
        if project_id:
            query += " AND c.project_id = @project"
            params.append({"name": "@project", "value": project_id})
        query += " ORDER BY c.file_name ASC"
        items = list(self.files.query_items(query=query, parameters=params))
        return [self._strip_id(i) for i in items]

    def delete_file(self, file_name: str) -> bool:
        try:
            self.files.delete_item(item=_file_id(file_name), partition_key=file_name)
            return True
        except Exception:  # noqa: BLE001 - missing record
            return False

    # -------------------------------------------------------------- lifecycle

    def close(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None and getattr(client, "close", None) is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to close Cosmos client", exc_info=True)


def build_cosmos_store(settings: Any, **kwargs: Any) -> CosmosNoSQLStore:
    """Construct the Cosmos store, raising loudly when unconfigured."""
    return CosmosNoSQLStore(settings, **kwargs)