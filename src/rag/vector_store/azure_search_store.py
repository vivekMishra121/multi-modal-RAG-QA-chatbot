"""Azure AI Search vector store backend (dense + keyword hybrid retrieval).

Implements the same ``BaseVectorStore`` contract as Qdrant/FAISS so the
retrieval, pipeline, and API layers are fully portable. Key behaviours:

  * The Azure AI Search index is provisioned on first use (HNSW profile over
    the ``embedding`` field), then reused across restarts.
  * Document ids are deterministic UUIDv5 values derived from ``chunk_id`` so
    re-uploading a chunk merge-updates instead of duplicating it.
  * ``search`` is a pure vector query; ``hybrid_search`` combines keyword text
    with the vector query (RRF-scored by the service) and optionally uses the
    service's semantic ranker when ``semantic_rank`` is enabled.
  * ``metadata`` is persisted as JSON; ``file_name``/``chunk_type``/``page``
    are additionally stored as filterable fields.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from azure.core.credentials import AzureKeyCredential

try:  # pragma: no cover - branch exercised by the SDK import order
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        HnswParameters,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchAlgorithmMetric,
        VectorSearchProfile,
    )
    from azure.search.documents.models import VectorizedQuery
except Exception:  # pragma: no cover - azure search packages are optional deps
    SearchClient = None  # type: ignore[misc, assignment]
    SearchIndexClient = None  # type: ignore[misc, assignment]
    VectorizedQuery = None  # type: ignore[misc, assignment]

from .base import BaseVectorStore

logger = logging.getLogger(__name__)

_PROFILE = "hnsw-profile"
_ALGORITHM = "hnsw-config"
_DEFAULT_SEMANTIC_CONFIG = "rag-semantic-config"

# Fields searchable/filterable at query time. Unknown metadata keys cannot be
# indexed dynamically in Azure AI Search, so they are kept in ``metadata`` only.
_FILTERABLE_FIELDS = ("file_name", "chunk_type", "page")


def _escape(value: str) -> str:
    """Escape a string for use inside an OData string literal."""
    return str(value).replace("'", "''")


def _document_id(chunk_id: str, index_name: str) -> str:
    """Deterministic document id for a chunk (idempotent upserts)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{index_name}:{chunk_id}"))


def _build_index_schema(
    index_name: str, dimension: int, semantic_rank: bool
) -> SearchIndex:
    """Schema used to provision the search index (idempotent shape)."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="chunk_id", type="Edm.String", filterable=True),
        SearchableField(name="content", type="Edm.String"),
        SearchableField(name="chunk_type", type="Edm.String", filterable=True, facetable=True),
        SearchableField(name="file_name", type="Edm.String", filterable=True, facetable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, nullable=True, filterable=True),
        SimpleField(name="metadata", type=SearchFieldDataType.String),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimension,
            vector_search_profile_name=_PROFILE,
        ),
    ]

    vector_search = VectorSearch(
        profiles=[
            VectorSearchProfile(name=_PROFILE, algorithm_configuration_name=_ALGORITHM),
        ],
        algorithms=[
            HnswAlgorithmConfiguration(
                name=_ALGORITHM,
                parameters=HnswParameters(
                    m=16,
                    ef_construction=400,
                    ef_search=500,
                    metric=VectorSearchAlgorithmMetric.COSINE,
                ),
            ),
        ],
    )
    kwargs: dict[str, Any] = {"vector_search": vector_search}

    if semantic_rank:
        kwargs["semantic_search"] = SemanticSearch(
            default_configuration_name=_DEFAULT_SEMANTIC_CONFIG,
            configurations=[
                SemanticConfiguration(
                    name=_DEFAULT_SEMANTIC_CONFIG,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="file_name"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ],
        )

    return SearchIndex(name=index_name, fields=fields, **kwargs)


def _metadata_json(chunk: dict[str, Any]) -> str:
    metadata = chunk.get('metadata') or {}
    return json.dumps(metadata, ensure_ascii=False)


class AzureAISearchStore(BaseVectorStore):
    """Azure AI Search-backed vector store."""

    distance_metric = 'cosine'

    def __init__(
        self,
        endpoint: Optional[str],
        api_key: Optional[str],
        index_name: str,
        dimension: int,
        semantic_rank: bool = False,
        semantic_config_name: str = _DEFAULT_SEMANTIC_CONFIG,
        client: Optional[Any] = None,
        index_client: Optional[Any] = None,
    ):
        """
        Args:
            endpoint: Search service URL (https://<name>.search.windows.net).
            api_key: Admin or query key.
            index_name: Index (collection) name.
            dimension: Dense embedding dimension.
            semantic_rank: Enable the service's semantic ranker on hybrid search.
            semantic_config_name: Name of the semantic configuration.
            client: Optional pre-built SearchClient (tests).
            index_client: Optional pre-built SearchIndexClient (tests).
        """
        if not endpoint or not api_key:
            raise ValueError(
                "Azure AI Search backend requires AZURE_SEARCH_ENDPOINT and "
                "AZURE_SEARCH_API_KEY. Add them to your .env or switch "
                "VECTOR_STORE_BACKEND to another backend."
            )
        if SearchClient is None or SearchIndexClient is None or VectorizedQuery is None:
            raise ValueError(
                "Azure AI Search packages are not installed. "
                "Run: pip install azure-search-documents"
            )

        self.endpoint = endpoint
        self.index_name = index_name
        self.dimension = dimension
        self.semantic_rank = semantic_rank
        self.semantic_config_name = semantic_config_name

        # Lightweight in-process chunk cache (stats, MMR embedding reuse).
        self.chunks: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}

        self.index_client = index_client or SearchIndexClient(
            endpoint, credential=AzureKeyCredential(api_key)
        )
        self.client = client or self.index_client.get_search_client(index_name)

        self._ensure_index()
        logger.info(
            "Initialized Azure AI Search store (index=%s, dim=%s, semantic=%s)",
            index_name,
            dimension,
            semantic_rank,
        )

    # ------------------------------------------------------------- lifecycle

    def _ensure_index(self) -> None:
        """Provision the index on first use; reuse it afterwards."""
        try:
            self.index_client.get_index(self.index_name)
            logger.info("Reusing existing Azure AI Search index '%s'", self.index_name)
            return
        except Exception as e:  # noqa: BLE001 - index missing (404) -> create it
            logger.debug("Index '%s' not found (%s); provisioning", self.index_name, e)
        self.index_client.create_index(
            _build_index_schema(self.index_name, self.dimension, self.semantic_rank)
        )
        logger.info("Provisioned Azure AI Search index '%s'", self.index_name)

    def close(self) -> None:
        """Release resources (SDK clients hold no long-lived sockets)."""
        for obj in (self.client, self.index_client):
            close = getattr(obj, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("Error closing %s: %s", type(obj).__name__, e)

    # ---------------------------------------------------------------- writes

    def _chunk_to_document(self, chunk: dict[str, Any]) -> dict[str, Any]:
        """Map a chunk dict to a search document."""
        metadata = chunk.get('metadata') or {}
        cid = chunk.get('chunk_id') or uuid.uuid4().hex
        return {
            'id': _document_id(cid, self.index_name),
            'chunk_id': cid,
            'content': chunk.get('content', ''),
            'chunk_type': chunk.get('chunk_type', 'text'),
            'file_name': metadata.get('file_name'),
            'page': metadata.get('page'),
            'metadata': _metadata_json(chunk),
            'embedding': chunk.get('embedding', []),
        }

    def add_chunks(self, chunk_dicts: list[dict[str, Any]]) -> None:
        """Merge-or-upload chunks into the search index (idempotent)."""
        if not chunk_dicts:
            return
        documents = [self._chunk_to_document(c) for c in chunk_dicts]
        self.client.merge_or_upload_documents(documents)

        self.chunks.extend(chunk_dicts)
        for chunk in chunk_dicts:
            cid = chunk.get('chunk_id')
            if cid:
                self._by_id[cid] = chunk
        logger.info("Upserted %d chunks to Azure AI Search (%d total)", len(chunk_dicts), len(self))

    def delete_all(self) -> None:
        """Delete every document from the index."""
        while True:
            ids = [
                r['id']
                for r in self.client.search(search_text="*", select=["id"], include_total_count=True)
            ]
            if not ids:
                break
            self.client.delete_documents([{'id': i} for i in ids])
        self.chunks = []
        self._by_id = {}

    def delete_by_document(self, file_name: str) -> int:
        """Delete all chunks attributed to ``file_name``; returns count removed."""
        odata = f"file_name eq '{_escape(file_name)}'"
        ids = [
            r['id']
            for r in self.client.search(
                search_text="*", filter=odata, select=["id"], include_total_count=True
            )
        ]
        deleted = len(ids)
        if ids:
            self.client.delete_documents([{'id': i} for i in ids])
        self.chunks = [
            c for c in self.chunks if (c.get('metadata') or {}).get('file_name') != file_name
        ]
        self._by_id = {c['chunk_id']: c for c in self.chunks if c.get('chunk_id')}
        logger.info("Deleted %d chunks for document '%s'", deleted, file_name)
        return deleted

    # ---------------------------------------------------------------- reads

    @staticmethod
    def _filters_to_odata(filters: Optional[dict[str, Any]]) -> Optional[str]:
        """Convert known metadata filters to an OData filter expression."""
        if not filters:
            return None
        clauses = []
        for key, value in filters.items():
            if key not in _FILTERABLE_FIELDS:
                logger.warning(
                    "Filter key '%s' is not filterable in Azure AI Search; ignoring", key
                )
                continue
            if isinstance(value, bool) or isinstance(value, (int, float)):
                clauses.append(f"{key} eq {value}")
            else:
                clauses.append(f"{key} eq '{_escape(value)}'")
        return " and ".join(clauses) if clauses else None

    def _payload_to_chunk(self, item) -> tuple[dict[str, Any], float]:
        """Reconstruct a chunk dict from a search response item."""
        raw_metadata = item.get('metadata') or '{}'
        try:
            metadata = json.loads(raw_metadata)
        except (TypeError, ValueError):
            metadata = {}
        chunk: dict[str, Any] = {
            'chunk_id': item.get('chunk_id'),
            'content': item.get('content', ''),
            'chunk_type': item.get('chunk_type', 'text'),
            'metadata': metadata,
        }
        cid = chunk.get('chunk_id')
        cached = self._by_id.get(cid) if isinstance(cid, str) else None
        if cached is not None:
            if 'embedding' in cached:
                chunk['embedding'] = cached['embedding']
            if 'sparse_embedding' in cached:
                chunk['sparse_embedding'] = cached['sparse_embedding']
        return chunk, float(item.get('@search.score') or 0.0)

    def _run_query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[dict[str, Any]],
        query_text: Optional[str] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Shared vector(+keyword, optional semantic) search."""
        odata = self._filters_to_odata(filters)
        vector_query: Any = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )
        kwargs: dict[str, Any] = {
            'search_text': query_text or None,
            'filter': odata,
            'vector_queries': [vector_query],
            'top': top_k,
            'select': [
                'id', 'chunk_id', 'content', 'chunk_type', 'file_name', 'page', 'metadata',
            ],
            'include_total_count': True,
        }
        if self.semantic_rank and query_text:
            kwargs['query_type'] = 'semantic'
            kwargs['semantic_configuration_name'] = self.semantic_config_name
            kwargs['query_language'] = 'en-us'

        results = self.client.search(**kwargs)
        return [self._payload_to_chunk(item) for item in results]

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Dense vector similarity search."""
        if len(self) == 0:
            return []
        return self._run_query(query_embedding, top_k, filters)

    def hybrid_search(
        self,
        dense_embedding: list[float],
        sparse_embedding: Optional[dict[str, float]],
        query_text: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Hybrid search: keyword (RRF with the vector query) + optional semantic."""
        if len(self) == 0:
            return []
        return self._run_query(dense_embedding, top_k, filters, query_text=query_text)

    # --------------------------------------------------------------- metadata

    def list_documents(self) -> list[dict[str, Any]]:
        """Summarize indexed documents by ``file_name``."""
        counts: dict[str, int] = {}
        for item in self.client.search(
            search_text="*", select=["file_name", "chunk_id"], include_total_count=True
        ):
            fn = item.get('file_name') or 'unknown'
            counts[fn] = counts.get(fn, 0) + 1
        return [{'file_name': fn, 'chunks': n} for fn, n in sorted(counts.items())]

    def __len__(self) -> int:
        try:
            return int(self.client.get_document_count())
        except Exception:  # noqa: BLE001
            return len(self.chunks)

    def get_stats(self) -> dict[str, Any]:
        chunk_types: dict[str, int] = {}
        for chunk in self.chunks:
            ctype = chunk.get('chunk_type', 'unknown')
            chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
        try:
            stats = self.index_client.get_index_statistics(self.index_name)
            total = int(stats.get('document_count', len(self)))
        except Exception:  # noqa: BLE001
            total = len(self)
        return {
            'total_chunks': total,
            'dimension': self.dimension,
            'chunk_types': chunk_types,
            'backend': 'azure_search',
        }