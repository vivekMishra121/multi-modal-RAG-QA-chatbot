"""Qdrant (in-memory) vector store behaviour: identity, search, hybrid, snapshot."""

from __future__ import annotations

import pytest

from rag.vector_store.qdrant_store import QdrantVectorStore


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(dimension=4, location=":memory:", collection_name="test_chunks")


def _chunks():
    return [
        {
            'chunk_id': f'c{i}',
            'content': f'content {i}',
            'chunk_type': 'text',
            'metadata': {'file_name': f'doc{i}.txt', 'page': i},
            'embedding': [float(i + 1)] * 4,
        }
        for i in range(3)
    ]


def test_upsert_is_idempotent(store):
    store.add_chunks(_chunks())
    assert len(store) == 3

    updated = _chunks()
    updated[0]['content'] = 'changed'
    store.add_chunks(updated)
    assert len(store) == 3


def test_search_returns_sorted(store):
    store.add_chunks(_chunks())
    # query closest to c2 (embedding [3,3,3,3]).
    results = store.search([3.0, 3.0, 3.0, 3.0], top_k=2)
    assert len(results) == 2
    assert results[0][0]['chunk_id'] == 'c2'


def test_search_respects_filters(store):
    store.add_chunks(_chunks())
    results = store.search([1.0, 1.0, 1.0, 1.0], top_k=5, filters={'page': 1})
    assert results
    assert all(r[0]['metadata']['page'] == 1 for r in results)


def test_hybrid_search_with_sparse(store):
    store.add_chunks(_chunks())
    results = store.hybrid_search(
        dense_embedding=[3.0, 3.0, 3.0, 3.0],
        sparse_embedding={str(i): 1.0 for i in range(3)},
        top_k=2,
    )
    assert results
    assert results[0][0]['chunk_id'] == 'c2'


def test_delete_all(store):
    store.add_chunks(_chunks())
    assert len(store) == 3
    store.delete_all()
    assert len(store) == 0


def test_save_and_load_roundtrip(store, tmp_path):
    store.add_chunks(_chunks())
    store.save(str(tmp_path))

    restored = QdrantVectorStore.load(str(tmp_path), location=":memory:")
    assert len(restored) == 3
    hits = restored.search([3.0, 3.0, 3.0, 3.0], top_k=1)
    assert hits[0][0]['chunk_id'] == 'c2'
    store.close()
    restored.close()