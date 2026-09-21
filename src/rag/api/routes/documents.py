"""Document upload and management endpoints (thin over DocumentService)."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile

from rag.services.document_service import DocumentService

from ..deps import get_chatbot
from ..schemas import DeleteDocumentsResponse, DocumentsResponse, UploadResponse

router = APIRouter()

Chatbot = Annotated[Any, Depends(get_chatbot)]


@router.post("/documents/upload", response_model=UploadResponse, tags=["documents"])
async def upload_documents(
    request: Request,
    files: Annotated[list[UploadFile], File(...)],
    project_id: Annotated[Optional[str], Form()] = None,
    chatbot: Chatbot = None,
):
    """Extract, chunk, embed, index, and persist originals to blob storage.

    Accepts ``multipart/form-data`` with one or more ``files`` parts
    (PDF, DOCX, TXT) and an optional ``project_id`` form field. Content is
    appended to the existing index; re-uploading the same file updates its
    chunks in place.
    """
    settings = request.app.state.settings
    result = await DocumentService(chatbot).upload(files, settings, project_id=project_id)
    return UploadResponse(**result)


@router.get("/documents", response_model=DocumentsResponse, tags=["documents"])
async def list_documents(chatbot: Chatbot = None):
    """List the documents currently indexed, with their chunk counts."""
    result = await asyncio.to_thread(DocumentService(chatbot).list)
    return DocumentsResponse(**result)


@router.get("/documents/{file_name}/download", tags=["documents"])
async def download_document(
    request: Request,
    file_name: str,
    chatbot: Chatbot = None,
):
    """Download the originally uploaded file for an indexed document."""
    settings = request.app.state.settings
    service = DocumentService(chatbot)
    data, content_type = await asyncio.to_thread(service.download, file_name, settings)
    return Response(
        content=data,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
    )


@router.delete("/documents/{file_name}", response_model=DeleteDocumentsResponse, tags=["documents"])
async def delete_document(
    request: Request,
    file_name: str,
    chatbot: Chatbot = None,
):
    """Remove every chunk belonging to ``file_name`` from the index."""
    settings = request.app.state.settings
    service = DocumentService(chatbot)
    result = await asyncio.to_thread(service.delete, file_name, settings)
    return DeleteDocumentsResponse(**result)