"""RAGChatbot happy-path, memory, and indexing tests using deterministic fakes."""

from __future__ import annotations

import pytest

from .conftest import FakeBlobStore, FakeEmbedder, FakeMemoryStore, FakeStore


def _stub_answer(self, question: str, context: str, history=None):
    return {'answer': 'Stubbed answer', 'success': True, 'error': None}


def _make_chatbot(settings, monkeypatch, store=None):
    from rag.chatbot import RAGChatbot
    from rag.qa_generation.qa_chain import QAChain

    monkeypatch.setattr(QAChain, "generate_answer", _stub_answer)
    return RAGChatbot(
        settings=settings,
        store=store or FakeStore(),
        embedder=FakeEmbedder(),
        memory=FakeMemoryStore(),
        blob=FakeBlobStore(),
        api_key="sk-test",
    )


def test_chatbot_chat_requires_conversation_id(monkeypatch, off_network_settings):
    chatbot = _make_chatbot(off_network_settings, monkeypatch)
    with pytest.raises(ValueError):
        chatbot.chat("hello")
    chatbot.close()


def test_chatbot_chat_persists_memory(monkeypatch, off_network_settings):
    chatbot = _make_chatbot(off_network_settings, monkeypatch)

    result = chatbot.chat("hello", conversation_id="cv-1")
    assert result['success'] is True
    assert result['conversation_id'] == "cv-1"

    turns = chatbot.memory.get_conversation("cv-1")
    assert [t['role'] for t in turns] == ["user", "assistant"]
    assert turns[0]['content'] == "hello"
    assert turns[1]['content'] == "Stubbed answer"
    assert chatbot.memory.get_conversation("other") == []
    chatbot.close()


def test_chatbot_follow_up_uses_history(monkeypatch, off_network_settings):
    seen = {}

    def _stub(self, question: str, context: str, history=None):
        seen['history'] = history
        return {'answer': 'A', 'success': True, 'error': None}

    from rag.chatbot import RAGChatbot
    from rag.qa_generation.qa_chain import QAChain

    monkeypatch.setattr(QAChain, "generate_answer", _stub)
    chatbot = RAGChatbot(
        settings=off_network_settings,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        memory=FakeMemoryStore(),
        blob=FakeBlobStore(),
        api_key="sk-test",
    )
    chatbot.chat("first", conversation_id="cv-2")
    chatbot.chat("second", conversation_id="cv-2")

    assert seen['history'] and seen['history'][0]['content'] == "first"
    assert seen['history'][0]['role'] == "user"
    chatbot.close()


def test_chatbot_applies_project_filters(monkeypatch, off_network_settings):
    captured = {}

    def _stub(self, question: str, context: str, history=None):
        return {'answer': 'A', 'success': True, 'error': None}

    from rag.chatbot import RAGChatbot
    from rag.qa_generation.qa_chain import QAChain

    monkeypatch.setattr(QAChain, "generate_answer", _stub)
    chatbot = RAGChatbot(
        settings=off_network_settings,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        memory=FakeMemoryStore(),
        blob=FakeBlobStore(),
        api_key="sk-test",
    )
    chatbot.memory.create_project(
        "proj-1", "Acme", filters={"file_name": "acme.pdf"}
    )

    orig = chatbot.rag.query

    def spy(question, filters=None, history=None, **kwargs):
        captured['filters'] = filters
        return orig(question, filters=filters, history=history, **kwargs)

    chatbot.rag.query = spy  # type: ignore[method-assign]
    chatbot.chat("q", conversation_id="cv-3", project_id="proj-1")
    assert captured['filters'] == {"file_name": "acme.pdf"}

    # A per-request filter overrides the project filter.
    chatbot.chat(
        "q2",
        conversation_id="cv-3",
        project_id="proj-1",
        filters={"file_name": "other.pdf"},
    )
    assert captured['filters'] == {"file_name": "other.pdf"}
    chatbot.close()


def test_chatbot_unknown_project_404(monkeypatch, off_network_settings):
    from rag.core.errors import AppError

    chatbot = _make_chatbot(off_network_settings, monkeypatch)
    with pytest.raises(AppError) as exc:
        chatbot.chat("q", conversation_id="cv-9", project_id="does-not-exist")
    assert exc.value.status_code == 404
    chatbot.close()


def test_chatbot_chat_returns_formatted_response(monkeypatch, off_network_settings):
    from rag.chatbot import RAGChatbot
    from rag.qa_generation.qa_chain import QAChain

    monkeypatch.setattr(QAChain, "generate_answer", _stub_answer)

    chatbot = RAGChatbot(
        settings=off_network_settings,
        store=FakeStore(),
        embedder=FakeEmbedder(),
        memory=FakeMemoryStore(),
        blob=FakeBlobStore(),
        api_key="sk-test",
    )

    result = chatbot.chat("hello", conversation_id="cv-4")
    assert result['success'] is True
    assert result['answer'] == 'Stubbed answer'
    assert result['num_chunks'] >= 1
    assert result['strategy'] == 'vector'
    assert result['sources']
    chatbot.close()


def test_build_index_on_txt(monkeypatch, off_network_settings, tmp_path):
    """Index a tiny .txt file end-to-end with a fake embedder."""
    from rag.chatbot import build_index
    from rag.qa_generation.qa_chain import QAChain

    from .conftest import FakeEmbedder

    doc = tmp_path / "sample.txt"
    doc.write_text("First page line about economics.\n## Page 2\nMore economics content here.\n", encoding="utf-8")

    monkeypatch.setattr(QAChain, "generate_answer", _stub_answer)

    # Force tiny chunk sizes so multiple chunks are produced.
    off_network_settings.chunking.text_chunk_size = 40
    off_network_settings.chunking.text_chunk_overlap = 5
    off_network_settings.vector_store.path = str(tmp_path / "store")

    summary = build_index(
        str(doc),
        store_path=str(tmp_path / "store"),
        settings=off_network_settings,
        embedder=FakeEmbedder(),
    )
    assert summary['documents_processed'] == 1
    assert summary['total_chunks'] >= 1
    assert summary['embedding_dimension'] == 4

    # The snapshot must be queryable.
    from rag.vector_store.qdrant_store import QdrantVectorStore

    store = QdrantVectorStore.load(str(tmp_path / "store"), location=":memory:")
    assert len(store) == summary['total_chunks']
    store.close()