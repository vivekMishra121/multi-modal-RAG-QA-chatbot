"""Azure Blob Storage integration for raw document persistence."""

from .blob_store import BlobStore, build_blob_store

__all__ = ["BlobStore", "build_blob_store"]