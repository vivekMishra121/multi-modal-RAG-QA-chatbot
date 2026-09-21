"""Conversation history and project management endpoints (thin over MemoryService)."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from rag.services.memory_service import MemoryService

from ..deps import get_chatbot
from ..schemas import (
    ConversationResponse,
    DeleteConversationResponse,
    FileListResponse,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()

Chatbot = Annotated[Any, Depends(get_chatbot)]


def _service(chatbot: Any) -> MemoryService:
    return MemoryService(chatbot.memory)


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
    tags=["memory"],
)
async def get_conversation(conversation_id: str, chatbot: Chatbot = None):
    """Return the stored message history for a conversation."""
    return await asyncio.to_thread(
        _service(chatbot).get_conversation, conversation_id
    )


@router.delete(
    "/conversations/{conversation_id}",
    response_model=DeleteConversationResponse,
    tags=["memory"],
)
async def delete_conversation(conversation_id: str, chatbot: Chatbot = None):
    """Delete every message in a conversation."""
    return await asyncio.to_thread(
        _service(chatbot).delete_conversation, conversation_id
    )


@router.post("/projects", response_model=ProjectResponse, tags=["projects"])
async def create_project(request: ProjectCreate, chatbot: Chatbot = None):
    """Create a project (metadata + retrieval filters)."""
    service = _service(chatbot)
    return await asyncio.to_thread(
        service.create_project,
        request.name,
        request.description,
        request.tags,
        request.filters,
        request.project_id,
    )


@router.get("/projects", response_model=ProjectListResponse, tags=["projects"])
async def list_projects(chatbot: Chatbot = None):
    """List all projects."""
    return await asyncio.to_thread(_service(chatbot).list_projects)


@router.get("/projects/{project_id}", response_model=ProjectResponse, tags=["projects"])
async def get_project(project_id: str, chatbot: Chatbot = None):
    """Fetch a single project."""
    return await asyncio.to_thread(_service(chatbot).get_project, project_id)


@router.patch("/projects/{project_id}", response_model=ProjectResponse, tags=["projects"])
async def update_project(project_id: str, request: ProjectUpdate, chatbot: Chatbot = None):
    """Update fields of a project."""
    service = _service(chatbot)
    return await asyncio.to_thread(
        service.update_project,
        project_id,
        **request.model_dump(exclude_none=True),
    )


@router.delete("/projects/{project_id}", tags=["projects"])
async def delete_project(project_id: str, chatbot: Chatbot = None):
    """Delete a project."""
    return await asyncio.to_thread(_service(chatbot).delete_project, project_id)


@router.get("/documents/files", response_model=FileListResponse, tags=["documents"])
async def list_file_records(request: Request, chatbot: Chatbot = None):
    """List blob-backed file records (raw uploads persisted in blob storage)."""
    return await asyncio.to_thread(_service(chatbot).list_files)