"""Shared fixtures: deterministic fakes that avoid all external services."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from rag.vector_store.base import BaseVectorStore


class FakeEmbedder:
    """Deterministic embedder - no model downloads or network calls."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def embedding_dimension(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 10 + 1)] * self._dim for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query) + 1)] * self._dim

    def close(self) -> None:
        pass


def make_chunks(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            'chunk_id': f'c{i}',
            'content': f'Sample content number {i} about topic {chr(97 + i)}.',
            'chunk_type': 'text',
            'metadata': {'file_name': f'doc{i}.txt', 'page': i + 1},
            'embedding': [float(i + 1)] * 4,
        }
        for i in range(n)
    ]


class FakeStore(BaseVectorStore):
    """In-memory dense store implementing the BaseVectorStore contract."""

    distance_metric = 'cosine'
    dimension = 4

    def __init__(self, chunks: Optional[list[dict[str, Any]]] = None):
        self.chunks: list[dict[str, Any]] = chunks if chunks is not None else make_chunks()

    def add_chunks(self, chunk_dicts: list[dict[str, Any]]) -> None:
        self.chunks.extend(chunk_dicts)

    def search(self, query_embedding, top_k=5, filters=None) -> list[tuple[dict[str, Any], float]]:
        scored = []
        for chunk in self.chunks:
            score = sum(a * b for a, b in zip(query_embedding, chunk['embedding']))
            scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete_all(self) -> None:
        self.chunks = []

    def list_documents(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for c in self.chunks:
            fn = (c.get('metadata') or {}).get('file_name') or 'unknown'
            counts[fn] = counts.get(fn, 0) + 1
        return [{'file_name': fn, 'chunks': n} for fn, n in sorted(counts.items())]

    def delete_by_document(self, file_name: str) -> int:
        keep = [
            c for c in self.chunks if (c.get('metadata') or {}).get('file_name') != file_name
        ]
        deleted = len(self.chunks) - len(keep)
        self.chunks = keep
        return deleted

    def get_stats(self) -> dict[str, Any]:
        return {'backend': 'fake', 'total_chunks': len(self.chunks)}

    def __len__(self) -> int:
        return len(self.chunks)


class FakeMemoryStore:
    """In-memory Cosmos stand-in implementing the memory/project/file contract."""

    def __init__(self):
        self.messages: list[dict[str, Any]] = []
        self.projects: dict[str, dict[str, Any]] = {}
        self.files: list[dict[str, Any]] = []

    def append_message(self, conversation_id, role, content, sources=None, project_id=None):
        item = {
            'id': f'm{len(self.messages)}',
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'sources': sources or [],
            'project_id': project_id,
            'created_at': len(self.messages),
        }
        self.messages.append(item)
        return item

    def get_conversation(self, conversation_id, limit=None):
        turns = [m for m in self.messages if m['conversation_id'] == conversation_id]
        return turns[-limit:] if limit is not None else turns

    def delete_conversation(self, conversation_id):
        total = len(self.messages)
        self.messages = [m for m in self.messages if m['conversation_id'] != conversation_id]
        return total - len(self.messages)

    def create_project(self, project_id, name, description=None, tags=None,
                       filters=None, file_names=None):
        item = {
            'project_id': project_id,
            'name': name,
            'description': description or '',
            'tags': tags or [],
            'filters': filters or {},
            'file_names': file_names or [],
            'created_at': 1,
            'updated_at': 1,
        }
        self.projects[project_id] = item
        return dict(item)

    def get_project(self, project_id):
        item = self.projects.get(project_id)
        return dict(item) if item is not None else None

    def list_projects(self):
        return [dict(p) for p in self.projects.values()]

    def update_project(self, project_id, **fields):
        item = self.projects.get(project_id)
        if item is None:
            return None
        for key, value in fields.items():
            if value is not None:
                item[key] = value
        return dict(item)

    def delete_project(self, project_id):
        return self.projects.pop(project_id, None) is not None

    def upsert_file(self, file_name, blob_path, content_type=None, size_bytes=None,
                    chunk_count=None, project_id=None):
        item = {
            'file_name': file_name,
            'project_id': project_id,
            'blob_path': blob_path,
            'content_type': content_type,
            'size_bytes': size_bytes,
            'chunk_count': chunk_count or 0,
            'created_at': 1,
            'updated_at': 1,
        }
        self.files = [f for f in self.files if f['file_name'] != file_name]
        self.files.append(item)
        return item

    def get_file(self, file_name):
        return next((dict(f) for f in self.files if f['file_name'] == file_name), None)

    def list_files(self, project_id=None):
        if project_id is None:
            return [dict(f) for f in self.files]
        return [dict(f) for f in self.files if f.get('project_id') == project_id]

    def delete_file(self, file_name):
        before = len(self.files)
        self.files = [f for f in self.files if f['file_name'] != file_name]
        return len(self.files) != before

    def close(self) -> None:
        pass


class FakeBlobStore:
    """In-memory blob store implementing the BlobStore contract."""

    def __init__(self):
        self.namespace: dict[str, bytes] = {}
        self.content_types: dict[str, str] = {}

    def _name(self, name: str) -> str:
        return f"rag-uploads/{name}"

    def upload_blob(self, name, data, content_type=None, overwrite=True):
        self.namespace[self._name(name)] = data
        if content_type:
            self.content_types[self._name(name)] = content_type
        return self._name(name)

    def upload_file(self, name, fileobj, content_type=None):
        return self.upload_blob(name, fileobj.read(), content_type=content_type)

    def download(self, name):
        return self.namespace[self._name(name)]

    def exists(self, name):
        return self._name(name) in self.namespace

    def delete(self, name):
        return self.namespace.pop(self._name(name), None) is not None

    def list_blobs(self, prefix=""):
        return [
            {'name': name, 'size_bytes': len(data)}
            for name, data in self.namespace.items()
        ]

    def close(self) -> None:
        pass


class FakeChatbot:
    """Minimal chatbot stand-in for API-level tests."""

    def __init__(self):
        self.store = FakeStore()
        self.embedder = FakeEmbedder()
        self.memory = FakeMemoryStore()
        self.blob = FakeBlobStore()

    def chat(self, question: str, filters=None, conversation_id=None, project_id=None):
        return {
            'answer': 'Fake answer',
            'sources': [],
            'success': True,
            'error': None,
            'num_chunks': 0,
            'strategy': 'vector',
            'conversation_id': conversation_id or 'fake-conversation',
        }

    def close(self) -> None:
        pass


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_memory() -> FakeMemoryStore:
    return FakeMemoryStore()


@pytest.fixture
def fake_blob() -> FakeBlobStore:
    return FakeBlobStore()


@pytest.fixture
def off_network_settings():
    """Settings with retrieval reranking/expansion disabled (no model downloads).

    Forces the Qdrant ``:memory:`` backend so nothing touches Azure AI Search
    or the network in offline tests. Tests inject fake memory/blob stores.
    """
    from rag.core.config import Settings

    settings = Settings()
    settings.retrieval.use_reranker = False
    settings.retrieval.use_query_expansion = False
    settings.retrieval.strategy = 'vector'
    settings.vector_store.backend = 'qdrant'
    return settings