"""Pydantic request/response schemas for the API.

Note: no ``from __future__ import annotations`` here - annotations are real
types so pydantic evaluates them eagerly (Python 3.9 compatible).
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Body of ``POST /api/v1/chat``."""

    question: str = Field(min_length=1, max_length=4000)
    conversation_id: str = Field(min_length=1)
    filters: Optional[dict[str, Any]] = None
    project_id: Optional[str] = None


class Source(BaseModel):
    """A retrieved source citation."""

    source_id: int
    file_name: str
    page: Any = None
    chunk_type: str
    content_preview: str


class ChatResponse(BaseModel):
    """Response of ``POST /api/v1/chat``."""

    answer: Optional[str] = None
    sources: list[Source] = Field(default_factory=list)
    success: bool
    error: Optional[str] = None
    num_chunks: int = 0
    strategy: Optional[str] = None
    conversation_id: Optional[str] = None


class Message(BaseModel):
    """A single stored conversation turn."""

    role: str
    content: str
    sources: list[Source] = Field(default_factory=list)
    project_id: Optional[str] = None
    created_at: Optional[float] = None


class ConversationResponse(BaseModel):
    """Response of ``GET /api/v1/conversations/{conversation_id}``."""

    conversation_id: str
    messages: list[Message] = Field(default_factory=list)
    total_messages: int = 0


class DeleteConversationResponse(BaseModel):
    """Response of ``DELETE /api/v1/conversations/{conversation_id}``."""

    conversation_id: str
    messages_deleted: int = 0


class ProjectCreate(BaseModel):
    """Body of ``POST /api/v1/projects``."""

    name: str = Field(min_length=1, max_length=200)
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    filters: Optional[dict[str, Any]] = None
    project_id: Optional[str] = None


class ProjectUpdate(BaseModel):
    """Body of ``PATCH /api/v1/projects/{project_id}``."""

    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    filters: Optional[dict[str, Any]] = None
    file_names: Optional[list[str]] = None


class ProjectResponse(BaseModel):
    """A project record."""

    project_id: str
    name: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    file_names: list[str] = Field(default_factory=list)
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class ProjectListResponse(BaseModel):
    """Response of ``GET /api/v1/projects``."""

    projects: list[ProjectResponse] = Field(default_factory=list)
    total_projects: int = 0


class FileRecord(BaseModel):
    """A stored record linking an indexed document to its source blob."""

    file_name: str
    project_id: Optional[str] = None
    blob_path: Optional[str] = None
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    chunk_count: int = 0
    created_at: Optional[float] = None
    updated_at: Optional[float] = None


class FileListResponse(BaseModel):
    """Response of ``GET /api/v1/documents/files`` (blob-backed records)."""

    files: list[FileRecord] = Field(default_factory=list)
    total_files: int = 0


class IndexRequest(BaseModel):
    """Body of ``POST /api/v1/index``."""

    document_path: str = Field(min_length=1)
    recreate: bool = True


class IndexResponse(BaseModel):
    """Response of ``POST /api/v1/index``."""

    documents_processed: int
    documents_failed: int = 0
    total_chunks: int
    tables_extracted: int = 0
    images_extracted: int = 0
    embedding_dimension: int
    vector_store_backend: str


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str
    app: str
    version: str
    environment: str
    chatbot_ready: bool
    vector_store_backend: Optional[str] = None
    index_size: int = 0
    memory_backend: Optional[str] = None
    error: Optional[str] = None


class UploadResponse(BaseModel):
    """Response of ``POST /api/v1/documents/upload``."""

    files_uploaded: int
    documents_processed: int
    documents_failed: int = 0
    total_chunks: int
    tables_extracted: int = 0
    images_extracted: int = 0
    embedding_dimension: int
    vector_store_backend: str
    blob_paths: list[str] = Field(default_factory=list)


class DocumentItem(BaseModel):
    """A single indexed document (name + chunk count)."""

    file_name: str = "unknown"
    chunks: int = 0
    blob_path: Optional[str] = None
    size_bytes: Optional[int] = None


class DocumentsResponse(BaseModel):
    """Response of ``GET /api/v1/documents``."""

    documents: list[DocumentItem] = Field(default_factory=list)
    total_documents: int = 0
    total_chunks: int = 0


class DeleteDocumentsResponse(BaseModel):
    """Response of ``DELETE /api/v1/documents/{file_name}``."""

    file_name: str
    deleted_chunks: int = 0