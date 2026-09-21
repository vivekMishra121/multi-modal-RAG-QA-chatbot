"""RAG QA chatbot application package (src-layout).

Layers
------
``rag.core``      configuration, error model, logging
``rag.rag``       the RAG engine: document ingestion, chunking, retrieval,
                  QA generation, vector stores, and the ``RAGChatbot``
``rag.services``  application use cases (chat, documents, indexing)
``rag.api``       FastAPI presentation layer (application factory, routes)
"""

__version__ = "1.1.0"