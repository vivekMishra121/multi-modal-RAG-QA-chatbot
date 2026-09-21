"""FastAPI endpoint tests (degraded start + healthy path with a fake chatbot)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .conftest import FakeChatbot


@pytest.fixture
def api_settings(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from rag.core.config import Settings

    settings = Settings()
    settings.retrieval.use_reranker = False
    settings.vector_store.path = str(tmp_path / "store")
    return settings


def test_health_and_chat_degraded_without_key(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "degraded"
        assert body["chatbot_ready"] is False
        assert "API key" in body["error"]

        chat = client.post("/api/v1/chat", json={"question": "hello"})
        assert chat.status_code == 503
        assert "request_id" in chat.json()


def test_chat_validation(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        r = client.post("/api/v1/chat", json={"question": ""})
        assert r.status_code in (422, 503)


def test_healthy_chat_and_health(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["chatbot_ready"] is True
        assert body["vector_store_backend"] == "fake"
        assert body["index_size"] == 3

        chat = client.post(
            "/api/v1/chat",
            json={"question": "what is this?", "conversation_id": "conv-1"},
        )
        assert chat.status_code == 200
        payload = chat.json()
        assert payload["success"] is True
        assert payload["answer"] == "Fake answer"
        assert payload["conversation_id"] == "conv-1"


def test_chat_requires_conversation_id(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        r = client.post("/api/v1/chat", json={"question": "hello"})
        assert r.status_code == 422
        assert "conversation_id" in r.text


def test_index_missing_path_returns_404(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        r = client.post("/api/v1/index", json={"document_path": "/does/not/exist.pdf"})
        assert r.status_code == 404


def test_openapi_docs_available(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/docs").status_code == 200


def _pdf_upload():
    return ("files", ("report.txt", b"Revenue rose 12 percent to $2.1B this quarter.", "text/plain"))


def test_upload_document_ingests_txt(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        r = client.post("/api/v1/documents/upload", files=[_pdf_upload()])
        assert r.status_code == 200
        body = r.json()
        assert body["files_uploaded"] == 1
        assert body["documents_processed"] == 1
        assert body["total_chunks"] >= 1
        assert body["embedding_dimension"] == 4
        assert len(client.app.state.chatbot.store) >= 1


def test_upload_multiple_files(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        files = [
            ("files", ("a.txt", b"Alpha content about widgets.", "text/plain")),
            ("files", ("b.txt", b"Beta content about gadgets.", "text/plain")),
        ]
        r = client.post("/api/v1/documents/upload", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["files_uploaded"] == 2
        assert body["documents_processed"] == 2
        assert body["total_chunks"] >= 2


def test_upload_rejects_unsupported_type(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        r = client.post(
            "/api/v1/documents/upload",
            files=[("files", ("notes.md", b"# markdown", "text/markdown"))],
        )
        assert r.status_code == 415


def test_upload_requires_healthy_chatbot(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        r = client.post("/api/v1/documents/upload", files=[_pdf_upload()])
        assert r.status_code == 503


def test_list_and_delete_documents(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        client.app.state.chatbot = FakeChatbot()
        client.app.state.chatbot_error = None

        listed = client.get("/api/v1/documents")
        assert listed.status_code == 200
        body = listed.json()
        assert body["total_documents"] == 3
        assert body["total_chunks"] == 3
        assert {d["file_name"] for d in body["documents"]} == {
            "doc0.txt",
            "doc1.txt",
            "doc2.txt",
        }

        deleted = client.delete("/api/v1/documents/doc0.txt")
        assert deleted.status_code == 200
        assert deleted.json() == {"file_name": "doc0.txt", "deleted_chunks": 1}
        assert len(client.app.state.chatbot.store) == 2


def test_document_endpoints_require_healthy_chatbot(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        assert client.get("/api/v1/documents").status_code == 503
        assert client.delete("/api/v1/documents/x.txt").status_code == 503


def _healthy(client):
    client.app.state.chatbot = FakeChatbot()
    client.app.state.chatbot_error = None


def test_upload_persists_blob_and_file_record(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        _healthy(client)
        r = client.post("/api/v1/documents/upload", files=[_pdf_upload()])
        assert r.status_code == 200
        body = r.json()
        assert body["blob_paths"] == ["rag-uploads/default/report.txt"]
        assert client.app.state.chatbot.memory.get_file("report.txt")["blob_path"].endswith(
            "report.txt"
        )


def test_download_original_file(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        _healthy(client)
        client.app.state.chatbot.blob.upload_blob(
            "report.txt", b"Revenue rose 12 percent.", content_type="text/plain"
        )
        client.app.state.chatbot.memory.upsert_file(
            "report.txt", "rag-uploads/report.txt", content_type="text/plain", size_bytes=27
        )

        r = client.get("/api/v1/documents/report.txt/download")
        assert r.status_code == 200
        assert r.content == b"Revenue rose 12 percent."
        assert r.headers["content-type"].startswith("text/plain")


def test_conversation_read_and_delete(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        _healthy(client)
        mem = client.app.state.chatbot.memory
        mem.append_message("cv", "user", "hi")
        mem.append_message("cv", "assistant", "hello!")

        listed = client.get("/api/v1/conversations/cv")
        assert listed.status_code == 200
        body = listed.json()
        assert body["total_messages"] == 2
        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]

        deleted = client.delete("/api/v1/conversations/cv")
        assert deleted.status_code == 200
        assert deleted.json()["messages_deleted"] == 2


def test_project_crud(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        _healthy(client)

        created = client.post(
            "/api/v1/projects",
            json={
                "name": "Acme HR",
                "description": "HR handbook",
                "tags": ["hr"],
                "filters": {"file_name": "acme.pdf"},
            },
        )
        assert created.status_code == 200
        pid = created.json()["project_id"]
        assert pid

        fetched = client.get(f"/api/v1/projects/{pid}")
        assert fetched.status_code == 200
        assert fetched.json()["name"] == "Acme HR"

        updated = client.patch(f"/api/v1/projects/{pid}", json={"tags": ["hr", "2026"]})
        assert updated.status_code == 200
        assert updated.json()["tags"] == ["hr", "2026"]

        listed = client.get("/api/v1/projects")
        assert listed.status_code == 200
        assert listed.json()["total_projects"] == 1

        removed = client.delete(f"/api/v1/projects/{pid}")
        assert removed.status_code == 200
        assert client.get(f"/api/v1/projects/{pid}").status_code == 404


def test_file_records_listing(api_settings):
    from rag.api.main import create_app

    client = TestClient(create_app(settings=api_settings))
    with client:
        _healthy(client)
        client.app.state.chatbot.memory.upsert_file(
            "a.txt", "rag-uploads/default/a.txt", size_bytes=10, chunk_count=2
        )
        r = client.get("/api/v1/documents/files")
        assert r.status_code == 200
        body = r.json()
        assert body["total_files"] == 1
        assert body["files"][0]["file_name"] == "a.txt"