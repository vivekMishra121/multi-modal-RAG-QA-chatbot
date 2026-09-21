# Multi-Modal RAG QA Chatbot

Retrieval-Augmented Generation over multi-modal documents (text, tables, images,
charts) backed by a modular FastAPI service, a Streamlit web client, and a
command-line interface.

## Architecture

```
PDF/DOCX/TXT (and images/tables) -- or upload via the API
   │
   ├──► storage (Blob)  ──►  raw upload bytes + file records (Cosmos)
   ▼
document_ingestion  ──►  text + tables + images + charts
   │
   ▼
chunking            ──►  typed chunks (text/table/chart) → embeddings
   │
   ▼
vector_store        ──►  provider layer (Azure AI Search default; Qdrant /
   │                       FAISS for offline dev)
   ▼
retrieval           ──►  standard / hybrid / MMR strategies + optional
   │                       cross-encoder reranking + query expansion
   ▼
qa_generation       ──►  token-budgeted context → LLM answer + citations
   │                       (conversation history injected from Cosmos)
   ▼
rag.api (FastAPI) / scripts.cli (CLI) / streamlit_app.py (web UI)
```

Conversation memory and projects live in **Azure Cosmos DB (NoSQL/Core SQL API)**
(`rag.cosmos`): chat turns are persisted per `conversation_id`, and projects
store metadata + retrieval filters applied to every query. Raw uploaded files
are kept in **Azure Blob Storage** (`rag.storage`).

## Layout

``src/rag`` is the application package (src-layout, instalable via ``pip install .``):

```
src/rag/
  core/            configuration (pydantic-settings), error model, logging
  services/        application use cases: chat, document management, indexing
  api/             FastAPI presentation layer: main factory, routes, schemas, deps
  chatbot.py       RAGChatbot orchestrator + build_index/get_chatbot helpers
  chunking/        smart chunking + embedding providers (OpenAI/Azure/BGE)
  retrieval/       Retriever (standard/hybrid/MMR), query expansion, reranking
  qa_generation/   qa_chain, pipeline, token-budgeted context manager
  vector_store/    provider layer: azure_search_store (default), qdrant_store, store (FAISS)
  cosmos/          Cosmos DB NoSQL store: conversation memory, projects, file records
  storage/         Azure Blob Storage wrapper for raw uploaded documents
  document_ingestion/  multi-modal extraction (pdfplumber, PyMuPDF, python-docx, OCR, tables)

tools/evaluation/   RAG evaluation metrics + test cases (scripts/evaluate.py runner)
scripts/cli.py      CLI: build / chat / evaluate
streamlit_app.py    Streamlit web client (HTTP client of the API)
tests/              pytest suite with in-memory Qdrant + fakes (no network/keys needed)
```

Each engine module stays decoupled from the API: routes are thin adapters over
``rag.services``, and ``rag.services`` orchestrate the engine plus shared policies
(validation, snapshotting, degraded-mode responses).

## Prerequisites

- Python 3.9
- Tesseract OCR (for image/OCR-based extraction)
- Ghostscript (runtime dependency of `camelot-py` for table extraction)

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt     # dev: includes pytest/ruff/mypy
pip install -r requirements/ui.txt      # UI add-on: streamlit
pip install -e .                        # src-layout: installs the `rag` package
```

## Configuration

Create `.env` from the template (`cp .env.example .env`) and set at least one
LLM provider. Everything is namespaced with env prefixes — see `.env.example`
for the full reference.

Quick reference (all optional):

```
LLM_PROVIDER=openai               # openai | azure_openai | groq
OPENAI_API_KEY=...                # OpenAI API key
AZURE_OPENAI_ENDPOINT=...         # required when provider=azure_openai
AZURE_OPENAI_API_KEY=...
EMBEDDING_PROVIDER=openai         # openai | azure_openai | sentence_transformers | bge_m3
RETRIEVAL_TOP_K=5
VECTOR_STORE_BACKEND=azure_search # azure_search (default) | qdrant | faiss
RETRIEVAL_STRATEGY=auto           # auto | vector | hybrid | mmr
RETRIEVAL_USE_RERANKER=false      # optional cross-encoder reranking
COSMOS_ENDPOINT=...               # Azure Cosmos DB (NoSQL API) URI (required)
COSMOS_KEY=...                    # Cosmos DB key (required)
AZURE_BLOB_CONNECTION_STRING=...  # Azure Blob Storage connection string
```

Notes:
- **Cosmos DB is required for chat**: it backs conversation memory and the
  project registry, and there is no fallback. Missing credentials start the
  service in degraded mode; chat returns `503` until configured. Database and
  container names (`rag-chat` / `conversations` / `projects` / `files`) and
  `COSMOS_MAX_HISTORY_TURNS=6` default in code.
- **Blob Storage** persists raw uploads under `{container}/{project_id|default}/{file}`
  (`AZURE_BLOB_CONTAINER` defaults to `rag-uploads`).
- The default vector store is **Azure AI Search**. Set ``AZURE_SEARCH_ENDPOINT``
  and ``AZURE_SEARCH_API_KEY`` to index/query documents, or switch
  ``VECTOR_STORE_BACKEND`` to ``qdrant`` for zero-config local development.
- The service boots in **degraded mode** when required keys are missing: health
  checks pass (``200``), document/chat endpoints return ``503`` with a ``request_id``.
- Embedding/retrieval dimension mismatches are guarded: if an existing index has
  a different dimension than the configured embedder, you are told to rebuild.
- pydantic-settings reads environment variables once per process; change `.env`
  and restart.

## Usage

### API service

```bash
uvicorn rag.api.main:app --reload --port 8000
```

- ``GET /health`` — liveness/readiness with dependency status
- ``POST /api/v1/chat`` — ``{"question": "...", "conversation_id": "...", "project_id": "...", "strategy": "vector", "top_k": 5}``
- ``POST /api/v1/documents/upload`` — upload PDF/DOCX/TXT files (multipart; optional ``project_id`` form field)
- ``GET  /api/v1/documents`` — list indexed documents and chunk counts
- ``GET  /api/v1/documents/{file_name}/download`` — stream the stored raw file
- ``DELETE /api/v1/documents/{file_name}`` — remove one document's chunks, blob, and record
- ``GET  /api/v1/conversations/{conversation_id}`` — fetch stored chat memory
- ``DELETE /api/v1/conversations/{conversation_id}`` — clear a conversation
- ``POST /api/v1/projects`` / ``GET /api/v1/projects`` — create/list projects
- ``GET/PATCH/DELETE /api/v1/projects/{project_id}`` — manage a project
- ``GET  /api/v1/documents/files`` — list stored file records

### CLI

```bash
python scripts/cli.py build <path/to/document.pdf> --store-path vector_store_data
python scripts/cli.py chat                        --store-path vector_store_data
python scripts/cli.py evaluate
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

It is a thin HTTP client of the API (set `RAG_API_URL` if the API is not on
`http://localhost:8000`). It talks to the model/vector store through the service,
never directly.

## Docker

```bash
cp .env.example .env    # fill in keys
docker compose up --build
```

- API on `http://localhost:8000`, UI on `http://localhost:8501`.
- Vector store data persists in a named volume.

## CI / Quality

```bash
ruff check .                       # lint
mypy src/ tools/ scripts/ streamlit_app.py
pytest tests/                      # 55 tests, in-memory, offline
```

GitHub Actions runs all three on push/PR (`.github/workflows/ci.yml`).

## Roadmap

- [x] Modular package structure + pyproject/requirements split
- [x] Enterprise src-layout: `core` / `services` / `api` / engine subpackages
- [x] FastAPI service with degraded-mode startup, middleware, structured logging
- [x] pydantic-settings configuration (openai / azure_openai / groq)
- [x] Provider-aware vector store: Azure AI Search (default), Qdrant, FAISS
- [x] Document upload, list, and delete API endpoints
- [x] Conversation memory + project registry (Azure Cosmos DB NoSQL)
- [x] Raw document persistence + download (Azure Blob Storage)
- [x] Streamlit web client + CLI
- [x] Docker + docker-compose + CI
- [ ] Azure OpenAI embeddings/generation wiring polish (eval with real keys)
- [ ] Monitoring dashboard

## Tech Stack

FastAPI, Streamlit, Qdrant, Azure AI Search, Azure Cosmos DB, Azure Blob Storage,
OpenAI/Azure OpenAI, pydantic-settings, PyMuPDF, pdfplumber, python-docx,
camelot-py, Tesseract, sentence-transformers, tiktoken.