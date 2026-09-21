"""BlobStore unit tests against an in-memory fake container surface."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

import pytest

from rag.storage.blob_store import BlobStore


class FakeBlobClient:
    def __init__(self, namespace: dict[str, dict[str, Any]], name: str):
        self._namespace = namespace
        self.name = name

    def upload_blob(self, data, overwrite=True):
        if hasattr(data, "read"):
            data = data.read()
        self._namespace[self.name] = {"data": data, "content_type": None}

    def set_http_headers(self, headers: dict[str, Any]):
        entry = self._namespace.setdefault(self.name, {"data": b"", "content_type": None})
        if "content_type" in headers:
            entry["content_type"] = headers["content_type"]

    def exists(self) -> bool:
        return self.name in self._namespace

    def delete_blob(self):
        self._namespace.pop(self.name, None)


class FakeBlobContainer:
    def __init__(self, settings: Any):
        self.settings = settings
        self.namespace: dict[str, dict[str, Any]] = {}

    def get_blob_client(self, name: str) -> FakeBlobClient:
        return FakeBlobClient(self.namespace, name)

    def delete_blob(self, name: str) -> None:
        if name not in self.namespace:
            raise KeyError(name)
        self.namespace.pop(name, None)

    def download_blob(self, name: str):
        entry = self.namespace.get(name)
        if entry is None:
            raise KeyError(name)

        class Downloader:
            def __init__(self, data: bytes):
                self._data = data

            def readall(self) -> bytes:
                return self._data

        return Downloader(entry["data"])

    def list_blobs(self, name_starts_with: str = ""):
        class Stub:
            pass

        blobs = []
        for name, entry in self.namespace.items():
            if name.startswith(name_starts_with):
                stub = Stub()
                stub.name = name
                stub.size = len(entry["data"])
                blobs.append(stub)
        return blobs

    def close(self) -> None:
        pass


def _store(container: Optional[FakeBlobContainer] = None) -> tuple[BlobStore, FakeBlobContainer]:
    from rag.core.config import Settings

    settings = Settings()
    container = container or FakeBlobContainer(settings)
    return BlobStore(settings, container=container), container


def test_store_raises_when_unconfigured():
    from rag.core.config import Settings

    with pytest.raises(ValueError, match="Blob Storage is not configured"):
        BlobStore(Settings())


def test_upload_download_delete_full_cycle():
    store, container = _store()

    blob_path = store.upload_file(
        "default/report.txt", BytesIO(b"hello world"), content_type="text/plain"
    )
    assert blob_path == "rag-uploads/default/report.txt"
    assert store.download("default/report.txt") == b"hello world"
    assert store.exists("default/report.txt") is True

    listed = store.list_blobs(prefix="default/")
    assert listed == [{"name": "default/report.txt", "size_bytes": 11}]

    assert store.delete("default/report.txt") is True
    assert store.exists("default/report.txt") is False
    assert store.delete("default/report.txt") is False


def test_upload_blob_bytes_and_content_type():
    store, container = _store()
    store.upload_blob("a.txt", b"\x00\x01", content_type="application/octet-stream")
    entry = container.namespace["a.txt"]
    assert entry["data"] == b"\x00\x01"
    assert entry["content_type"] == "application/octet-stream"

    with pytest.raises(KeyError):
        store.download("missing.txt")