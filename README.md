# Multi-Modal RAG QA System

A production-ready Retrieval-Augmented Generation system for multi-modal documents (text, tables, images) with industry-standard evaluation.

[![Grade: A](https://img.shields.io/badge/Grade-A-brightgreen)]()
[![Accuracy: 80-85%](https://img.shields.io/badge/Accuracy-80--85%25-blue)]()
[![Latency: <3s](https://img.shields.io/badge/Latency-<3s-orange)]()
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-yellow)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

---

## Overview

This system answers questions from complex documents by combining multi-modal extraction (text, tables, charts), semantic search, and LLM-based generation — all with automatic retrieval strategy selection and cross-encoder reranking.

---

## Features

| Feature | Details |
|---------|---------|
| Multi-Modal Extraction | Text, tables, images, charts (PDF/DOCX/TXT) |
| Smart Chunking | 512-char chunks with page tracking |
| Embeddings | `all-MiniLM-L6-v2` (384-dim) |
| Vector Store | FAISS with cosine similarity |
| Retrieval Strategies | Standard / MMR / Hybrid (auto-selected) |
| Reranking | Cross-encoder `ms-marco-MiniLM` (+15-20% accuracy) |
| Chart Detection | Bar/line/pie classification with confidence scoring |
| LLM | GPT-3.5-turbo with citation tracking |
| Evaluation | Industry-standard metrics (Precision, Recall, MRR, NDCG, Faithfulness) |

---

## Architecture

```
PDF/DOCX
   │
   ▼
Document Ingestion ──► Text + Tables + Images + Chart Detection
   │
   ▼
Chunking & Embedding ──► 512-char chunks → 384-dim vectors
   │
   ▼
FAISS Vector Store ──► Cosine similarity index + metadata
   │
   ▼
Retrieval ──► Auto strategy (Standard/MMR/Hybrid) → Cross-encoder reranking
   │
   ▼
QA Generation ──► LLM answer + citations
   │
   ▼
Evaluation ──► Retrieval + Generation + Multi-Modal + Latency metrics
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- Tesseract OCR

```bash
# Windows (Chocolatey)
choco install tesseract -y

# Or download: https://github.com/UB-Mannheim/tesseract/wiki
```

### Installation

```bash
git clone https://github.com/vivekMishra121/multi-modal-RAG-QA-chatbot.git
cd multi-modal-RAG-QA-chatbot
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
GROQ_API_KEY=your_groq_key_here       # optional
MODEL_NAME=gpt-3.5-turbo              # optional
MAX_CONTEXT_TOKENS=4096               # optional
TOP_K=5                               # optional
```

### Build Index

```bash
python main.py build "path/to/your/document.pdf"
```

### Run

```bash
# Streamlit UI
streamlit run chatbot.py

# CLI
python main.py
```

---

## Usage

### Basic

```python
from main import get_chatbot

chatbot = get_chatbot()
result = chatbot.chat("What is Qatar's GDP?")

print(result['answer'])
print(f"Sources: {len(result['sources'])} chunks")
```

### With Filters

```python
# Filter by page
result = chatbot.chat("What is inflation?", filters={'page': 5})

# Filter by content type
result = chatbot.chat("Show GDP data", filters={'chunk_type': 'table'})
result = chatbot.chat("Show growth trends", filters={'is_chart': True})
```

### Advanced Retrieval

```python
from retrieval import Retriever
from vector_store import VectorStore

store = VectorStore.load('./vector_store_data')
retriever = Retriever(store, use_reranker=True)

results = retriever.retrieve(
    query_embedding,
    query_text,
    top_k=5,
    strategy='auto',
    rerank=True
)
```

---

## Evaluation

```bash
python run_evaluation.py
```

### Metrics

| Category | Metrics | Target |
|----------|---------|--------|
| Retrieval (30%) | Precision@5, Recall@5, MRR, NDCG | >0.70 |
| Generation (30%) | Semantic Similarity, Faithfulness | >0.70 |
| Multi-Modal (25%) | Coverage, Table/Chart Accuracy | >0.70 |
| Latency (15%) | Response Time | <3.5s |

### Sample Output

```
📊 RETRIEVAL METRICS
  Precision@5: 0.850 ✅   Recall@5: 0.720 ✅
  MRR:         0.680 ✅   NDCG@5:   0.750 ✅

📊 GENERATION METRICS
  Semantic Similarity: 0.780 ✅
  Faithfulness:        0.880 ✅

📊 MULTI-MODAL METRICS
  Modality Coverage: 0.800 ✅
  Table Accuracy:    0.850 ✅
  Chart Detection:   0.820 ✅

📊 LATENCY
  Avg: 2.8s ✅

🏆 OVERALL: 0.825 — Grade A
```

---

## Performance

| Operation | Time |
|-----------|------|
| Indexing (78-page PDF) | ~45s (one-time) |
| Query end-to-end | 2.5–3.0s |
| Retrieval only | ~0.3s |
| Generation only | ~2.0s |

---

## Project Structure

```
multi-modal-RAG-QA-chatbot/
├── document_ingestion/     # Multi-modal extraction + chart detection
├── chunking/               # Smart chunking + embeddings
├── vector_store/           # FAISS index + metadata
├── retrieval/              # Auto strategy selection + reranking
├── qa_generation/          # LLM integration + citations
├── evaluation/             # Evaluation metrics
├── main.py                 # Core API
├── chatbot.py              # Streamlit UI
├── run_evaluation.py       # Evaluation runner
└── requirements.txt
```

---

## Configuration Reference

```python
# chunking/pipeline.py
text_chunk_size=512       # characters per chunk
text_chunk_overlap=100    # overlap between chunks

# main.py
retrieval_strategy='auto' # auto | standard | mmr | hybrid
use_reranker=True
top_k=5
```

---

## Roadmap

- [x] Document ingestion (PDF/DOCX/TXT)
- [x] Multi-modal extraction (text, tables, images)
- [x] Chart detection & classification
- [x] Smart chunking with page tracking
- [x] FAISS vector store
- [x] Auto retrieval strategy selection
- [x] Cross-encoder reranking
- [x] QA generation with citations
- [x] Industry-standard evaluation
- [x] Streamlit UI
- [ ] FastAPI deployment
- [ ] Docker container
- [ ] Monitoring dashboard

---

## Tech Stack

- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenAI](https://platform.openai.com/)
- [Streamlit](https://streamlit.io/)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

---
