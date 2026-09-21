"""CosmosNoSQLStore unit tests against an in-memory fake Cosmos SDK surface."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from rag.cosmos.store import CosmosNoSQLStore


class FakeContainer:
    """Minimal ContainerProxy stand-in supporting the queries the store emits."""

    def __init__(self, pk_field: str, items: Optional[list[dict[str, Any]]] = None):
        self.pk_field = pk_field
        self.items: list[dict[str, Any]] = items or []

    def upsert_item(self, item: dict[str, Any]) -> dict[str, Any]:
        for i, existing in enumerate(self.items):
            if existing.get("id") == item.get("id"):
                self.items[i] = item
                return item
        self.items.append(item)
        return item

    def read_item(self, item: str, partition_key: Any) -> dict[str, Any]:
        for existing in self.items:
            if existing.get("id") == item:
                return existing
        raise KeyError("Item not found")

    def delete_item(self, item: str, partition_key: Any) -> None:
        if not any(i.get("id") == item for i in self.items):
            raise KeyError("Item not found")
        self.items = [i for i in self.items if i.get("id") != item]

    def query_items(
        self, query: str, parameters: Optional[list[dict[str, Any]]] = None,
        partition_key: Any = None,
    ) -> list[dict[str, Any]]:
        params = {p["name"]: p["value"] for p in (parameters or [])}
        if partition_key is not None:
            rows = [i for i in self.items if i.get(self.pk_field) == partition_key]
        else:
            rows = list(self.items)

        if "@kind" in params:
            rows = [i for i in rows if i.get("type") == params["@kind"]]
        if "@project" in params:
            rows = [i for i in rows if i.get("project_id") == params["@project"]]

        low = query.lower()
        if "order by c." in low:
            field = low.split("order by c.")[1].split(" ")[0]
            desc = "desc" in low.split("order by c.")[1]
            rows = sorted(rows, key=lambda r: r.get(field) or 0, reverse=desc)
        if "@limit" in params:
            rows = rows[: params["@limit"]]
        return rows


class FakeDatabase:
    """Minimal DatabaseProxy stand-in."""

    def __init__(self, containers: Optional[dict[str, FakeContainer]] = None):
        self._containers = containers or {}

    def create_container_if_not_exists(self, id: str, partition_key: dict[str, str]):
        if id not in self._containers:
            field = partition_key["path"].lstrip("/")
            self._containers[id] = FakeContainer(field)
        return self._containers[id]


class FakeCosmosClient:
    """Minimal CosmosClient stand-in."""

    def __init__(self, database: Optional[FakeDatabase] = None):
        self.database = database or FakeDatabase()

    def create_database_if_not_exists(self, name):
        return self.database


def _settings(**overrides):
    from rag.core.config import Settings

    settings = Settings()
    for key, value in overrides.items():
        setattr(settings.cosmos, key, value)
    return settings


def test_store_raises_when_unconfigured():
    from rag.core.config import Settings

    settings = Settings()  # cosmos unconfigured
    with pytest.raises(ValueError, match="Cosmos DB is not configured"):
        CosmosNoSQLStore(settings)


def test_conversation_lifecycle():
    settings = _settings(endpoint="https://x.documents.azure.com:443/", key="k")
    store = CosmosNoSQLStore(
        settings,
        containers={
            "conversations": FakeContainer("conversation_id"),
            "projects": FakeContainer("project_id"),
            "files": FakeContainer("file_name"),
        },
    )
    store.append_message("cv-1", "user", "hello")
    store.append_message("cv-1", "assistant", "hi", sources=[{"source_id": 1}])
    store.append_message("cv-2", "user", "other")

    history = store.get_conversation("cv-1")
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert history[1]["sources"] == [{"source_id": 1}]

    limited = store.get_conversation("cv-1", limit=1)
    assert len(limited) == 1 and limited[0]["role"] == "assistant"

    assert store.delete_conversation("cv-1") == 2
    assert store.get_conversation("cv-1") == []


def test_project_crud():
    settings = _settings(endpoint="https://x.documents.azure.com:443/", key="k")
    store = CosmosNoSQLStore(
        settings,
        containers={
            "conversations": FakeContainer("conversation_id"),
            "projects": FakeContainer("project_id"),
            "files": FakeContainer("file_name"),
        },
    )
    store.create_project(
        "p-1", "Acme", description="Hr", tags=["hr"], filters={"file_name": "a.pdf"}
    )
    project = store.get_project("p-1")
    assert project["name"] == "Acme"
    assert project["filters"] == {"file_name": "a.pdf"}
    assert "id" not in project and "_rid" not in project

    updated = store.update_project("p-1", tags=["hr", "2026"])
    assert updated["tags"] == ["hr", "2026"]

    listed = store.list_projects()
    assert [p["project_id"] for p in listed] == ["p-1"]

    assert store.delete_project("p-1") is True
    assert store.get_project("p-1") is None
    assert store.delete_project("p-1") is False


def test_file_records():
    settings = _settings(endpoint="https://x.documents.azure.com:443/", key="k")
    files = FakeContainer("file_name")
    store = CosmosNoSQLStore(
        settings,
        containers={
            "conversations": FakeContainer("conversation_id"),
            "projects": FakeContainer("project_id"),
            "files": files,
        },
    )
    store.upsert_file("a.txt", "rag-uploads/a.txt", size_bytes=10, chunk_count=2, project_id="p-1")
    store.upsert_file("b.txt", "rag-uploads/b.txt", project_id="p-2")

    assert store.list_files(project_id="p-1")[0]["file_name"] == "a.txt"
    assert len(store.list_files()) == 2
    assert store.get_file("a.txt")["blob_path"] == "rag-uploads/a.txt"
    assert store.delete_file("a.txt") is True
    assert store.get_file("a.txt") is None


def test_provisioning_path_builds_containers():
    settings = _settings(endpoint="https://x.documents.azure.com:443/", key="k")
    store = CosmosNoSQLStore(
        settings,
        database=FakeDatabase(),
        client=FakeCosmosClient(FakeDatabase()),
    )
    assert store.conversations.pk_field == "conversation_id"
    assert store.projects.pk_field == "project_id"
    assert store.files.pk_field == "file_name"
    store.close()  # fake client has no close -> must not raise