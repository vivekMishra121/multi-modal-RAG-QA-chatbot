"""Azure Blob Storage store for raw document persistence.

Uploaded originals are the source of truth after ingestion. Chunks derived
from them live in the vector store; the Cosmos ``files`` container records the
blob path, so documents can be re-downloaded or re-processed.

Blob credentials are required: constructing the store (or using it without an
injected container) raises when unconfigured. Tests inject a fake container so
no network is required.
"""

from __future__ import annotations

import logging
from typing import Any, BinaryIO, Optional

logger = logging.getLogger(__name__)


class BlobStore:
    """Blob container backed store for raw uploads."""

    def __init__(self, settings: Any, container: Optional[Any] = None) -> None:
        """
        Args:
            settings: App settings (reads ``settings.blob``).
            container: Optional pre-built ``ContainerClient`` (tests).
        """
        cfg = settings.blob
        if container is None and not cfg.is_configured:
            raise ValueError(
                "Azure Blob Storage is not configured (AZURE_BLOB_CONNECTION_STRING). "
                "Raw document persistence cannot be enabled without it."
            )

        self.settings = settings
        self.container_name = cfg.container_name
        self._owns_client = container is None
        self._container: Any = container

        if container is None:
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(cfg.connection_string)
            self._container = service.get_container_client(cfg.container_name)
            if not self._container.exists():
                self._container.create_container()

    @property
    def container(self) -> Any:
        return self._container

    def upload_blob(
        self,
        name: str,
        data: bytes,
        content_type: Optional[str] = None,
        overwrite: bool = True,
    ) -> str:
        """Upload a blob and return its path (``container/name``)."""
        blob_client = self._container.get_blob_client(name)
        blob_client.upload_blob(data, overwrite=overwrite)
        if content_type:
            blob_client.set_http_headers({"content_type": content_type})
        logger.info("Uploaded blob %s/%s", self.container_name, name)
        return f"{self.container_name}/{name}"

    def download(self, name: str) -> bytes:
        """Download a blob's content as bytes."""
        stream = self._container.download_blob(name)
        return stream.readall()

    def exists(self, name: str) -> bool:
        return bool(self._container.get_blob_client(name).exists())

    def delete(self, name: str) -> bool:
        try:
            self._container.delete_blob(name)
            return True
        except Exception:  # noqa: BLE001 - missing blob
            return False

    def list_blobs(self, prefix: str = "") -> list[dict[str, Any]]:
        """List blobs under ``prefix`` with name and size metadata."""
        entries = []
        for blob in self._container.list_blobs(name_starts_with=prefix):
            entries.append({"name": blob.name, "size_bytes": blob.size})
        return entries

    def upload_file(
        self,
        name: str,
        fileobj: BinaryIO,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload from a file object, returning the blob path."""
        blob_client = self._container.get_blob_client(name)
        blob_client.upload_blob(fileobj, overwrite=True)
        if content_type:
            blob_client.set_http_headers({"content_type": content_type})
        return f"{self.container_name}/{name}"

    def close(self) -> None:
        if self._owns_client and getattr(self._container, "close", None) is not None:
            try:
                self._container.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Failed to close blob container", exc_info=True)


def build_blob_store(settings: Any, **kwargs: Any) -> BlobStore:
    """Construct the Blob store, raising loudly when unconfigured."""
    return BlobStore(settings, **kwargs)