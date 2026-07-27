# RAG System Architecture

## Component Separation (Industry Standard)

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. INGESTION (document_ingestion/)                    │
│     └─ Extract text, tables, images from PDFs          │
│                                                        │
│  2. CHUNKING (chunking/)                               │
│     └─ Split content + Generate embeddings             │
│                                                        │
│  3. INDEXING (vector_store/)                           │
│     └─ Store embeddings in FAISS                       │
│     └─ Basic similarity search                         │
│                                                        │
│  4. RETRIEVAL (retrieval/)                             │
│     └─ Advanced search strategies:                     │
│        • Standard (vector similarity)                  │
│        • MMR (diverse results)                         │
│        • Hybrid (vector + keyword)                     │
│                                                        │
│  5. GENERATION (coming next)                           │
│     └─ LLM integration for answers                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Why This Separation?

### **Indexing (vector_store/)**
- **Purpose**: Store and organize embeddings
- **Responsibility**: Fast vector similarity search
- **Methods**: `add_chunks()`, `search()`, `save()`, `load()`
- **Industry examples**: Pinecone, Weaviate, Chroma

### **Retrieval (retrieval/)**
- **Purpose**: Advanced search strategies
- **Responsibility**: Query processing, reranking, diversity
- **Methods**: `mmr_search()`, `hybrid_search()`, `rerank()`
- **Industry examples**: LangChain Retrievers, LlamaIndex

## Current Implementation

### Indexing (vector_store/)
```python
store = VectorStore(dimension=384, distance_metric='cosine')
store.add_chunks(chunks)
results = store.search(query_embedding, top_k=5)
```

### Retrieval (retrieval/)
```python
retriever = Retriever(store)

# Standard
results = retriever.retrieve(query_emb, query_text, strategy='standard')

# MMR (diverse)
results = retriever.retrieve(query_emb, query_text, strategy='mmr', lambda_mult=0.5)

# Hybrid (vector + keyword)
results = retriever.retrieve(query_emb, query_text, strategy='hybrid', alpha=0.7)
```

## Industry Alignment

| Component | Our Module | LangChain | LlamaIndex |
|-----------|------------|-----------|------------|
| Indexing | `vector_store/` | `VectorStore` | `VectorStoreIndex` |
| Retrieval | `retrieval/` | `Retriever` | `QueryEngine` |
| Generation | (next) | `Chain` | `ResponseSynthesizer` |

This separation follows industry best practices and makes the system modular and maintainable.
