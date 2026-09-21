"""AzureAISearchStore tests against a fake Search service client (no network)."""

from __future__ import annotations

from typing import Any, Optional

from rag.vector_store.azure_search_store import AzureAISearchStore


class FakeSearchClient:
    """In-memory stand-in for azure ``SearchClient``."""

    def __init__(self):
        self._docs: dict[str, dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------ data helpers

    def clear(self) -> None:
        self.calls = []
        self._docs = {}

    def _matches(self, doc: dict[str, Any], filter_expr: Optional[str]) -> bool:
        if not filter_expr:
            return True
        for clause in filter_expr.split(" and "):
            key, _, rest = clause.strip().partition(" eq ")
            value = rest
            if value.startswith("'") and value.endswith("'"):
                value = value[1:-1].replace("''", "'")
            else:
                try:
                    value = int(value)
                except ValueError:
                    pass
            if doc.get(key) != value:
                return False
        return True

    def _cosine(self, a, b) -> float:
        denom = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0

    # ---------------------------------------------------------------- SDK API

    def merge_or_upload_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            self._docs[doc['id']] = doc

    def delete_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            self._docs.pop(doc.get('id'), None)

    def get_document_count(self) -> int:
        return len(self._docs)

    def search(self, search_text=None, filter=None, top=None, select=None, **kwargs):  # noqa: A002 - SDK kwargs
        self.calls.append(
            {
                'search_text': search_text,
                'filter': filter,
                'top': top,
                'select': select,
                'vector_queries': kwargs.get('vector_queries'),
                'query_type': kwargs.get('query_type'),
                'semantic_configuration_name': kwargs.get('semantic_configuration_name'),
            }
        )
        candidates = [d for d in self._docs.values() if self._matches(d, filter)]

        scored: list[tuple[dict[str, Any], float]] = []
        for doc in candidates:
            score = 0.0
            if kwargs.get('vector_queries'):
                vec = kwargs['vector_queries'][0]
                score = self._cosine(doc.get('embedding', []), vec.vector)
            if search_text and search_text.strip() and search_text != '*':
                lex = sum(1 for w in search_text.lower().split() if w in (doc.get('content', '') or '').lower())
                score += 0.01 * lex
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        items = []
        for doc, score in scored:
            item: dict[str, Any] = dict(doc)
            if select:
                item = {k: v for k, v in item.items() if k in list(select) or k == 'id'}
            item['@search.score'] = score
            items.append(item)
        return items[: (top if top else len(items))]

    def close(self) -> None:
        pass


class FakeIndexClient:
    """In-memory stand-in for azure ``SearchIndexClient``."""

    def __init__(self, search_client: FakeSearchClient):
        self.search_client = search_client
        self.indexes: dict[str, Any] = {}

    def get_index(self, name: str) -> None:
        if name not in self.indexes:
            raise RuntimeError("missing")
        return self.indexes[name]

    def create_index(self, index) -> None:
        self.indexes[index.name] = index

    def get_index_statistics(self, name: str) -> dict[str, Any]:
        return {'document_count': self.search_client.get_document_count()}

    def close(self) -> None:
        pass


def make_chunks(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            'chunk_id': f'az{i}',
            'content': f'Quarterly revenue grew to $1{i}0 million for region {chr(97 + i)}.',
            'chunk_type': 'text',
            'metadata': {'file_name': f'report{i}.pdf', 'page': i + 1},
            'embedding': [float(i + 1), 1.0, 2.0, 3.0],
        }
        for i in range(n)
    ]


def _build_store(semantic_rank: bool = False) -> tuple[AzureAISearchStore, FakeSearchClient, FakeIndexClient]:
    search = FakeSearchClient()
    index = FakeIndexClient(search)
    store = AzureAISearchStore(
        endpoint='https://fake.search.windows.net',
        api_key='fake-key',
        index_name='test-idx',
        dimension=4,
        semantic_rank=semantic_rank,
        client=search,
        index_client=index,
    )
    return store, search, index


def test_missing_configuration_raises_clear_error():
    try:
        AzureAISearchStore(endpoint=None, api_key=None, index_name='x', dimension=4)
    except ValueError as e:
        assert 'AZURE_SEARCH_ENDPOINT' in str(e)
    else:  # pragma: no cover
        raise AssertionError('expected ValueError')


def test_index_provisioned_with_vector_profile():
    store, _, index = _build_store()
    by_name = {f.name: f for f in index.indexes['test-idx'].fields}

    assert by_name['id'].key is True
    assert by_name['embedding'].vector_search_dimensions == 4
    assert by_name['embedding'].vector_search_profile_name == 'hnsw-profile'
    assert by_name['file_name'].filterable is True
    assert by_name['page'].filterable is True
    assert index.indexes['test-idx'].vector_search is not None
    assert index.indexes['test-idx'].semantic_search is None
    assert len(store) == 0
    store.close()


def test_semantic_rank_adds_semantic_configuration():
    _, _, index = _build_store(semantic_rank=True)
    semantic = index.indexes['test-idx'].semantic_search
    assert semantic is not None
    assert semantic.configurations[0].name == 'rag-semantic-config'


def test_add_search_and_idempotency():
    store, search, _ = _build_store()
    chunks = make_chunks(3)
    store.add_chunks(chunks)
    assert len(store) == 3

    # Re-adding the same chunks must not duplicate.
    store.add_chunks(chunks)
    assert len(store) == 3

    query = [9.0, 1.0, 2.0, 3.0]
    results = store.search(query, top_k=2)
    assert len(results) == 2
    # Highest cosine against chunk 'az2' (embedding [3, 1, 2, 3]).
    assert results[0][0]['chunk_id'] == 'az2'
    assert results[0][0]['metadata']['file_name'] == 'report2.pdf'
    assert results[0][1] > results[1][1]
    store.close()


def test_search_filters_by_supported_fields():
    store, _, _ = _build_store()
    store.add_chunks(make_chunks(3))

    results = store.search([1.0, 1.0, 1.0, 1.0], top_k=5, filters={'file_name': 'report1.pdf'})
    assert {c['chunk_id'] for c, _ in results} == {'az1'}
    store.close()


def test_hybrid_search_uses_keyword_and_vector():
    store, search, _ = _build_store()
    store.add_chunks(make_chunks(2))
    results = store.hybrid_search(
        dense_embedding=[2.0, 2.0, 2.0, 2.0],
        sparse_embedding=None,
        query_text='revenue grew 110',
        top_k=2,
    )
    assert len(results) == 2
    recorded = search.calls[-1]
    assert recorded['search_text'] == 'revenue grew 110'
    assert recorded['vector_queries'] is not None
    assert recorded['query_type'] is None  # semantic ranker disabled by default

    semantic_store, semantic_search, _ = _build_store(semantic_rank=True)
    semantic_store.add_chunks(make_chunks(2))
    semantic_store.hybrid_search([2.0, 2.0, 2.0, 2.0], None, 'revenue', top_k=2)
    assert semantic_search.calls[-1]['query_type'] == 'semantic'
    assert semantic_search.calls[-1]['semantic_configuration_name'] == 'rag-semantic-config'
    semantic_store.close()
    store.close()


def test_list_documents_and_delete_by_document():
    store, _, _ = _build_store()
    store.add_chunks(make_chunks(4) + make_chunks(4))
    documents = store.list_documents()
    assert {d['file_name'] for d in documents} == {'report0.pdf', 'report1.pdf', 'report2.pdf', 'report3.pdf'}
    assert sum(d['chunks'] for d in documents) == 4

    deleted = store.delete_by_document('report0.pdf')
    assert deleted == 1
    assert len(store) == 3
    assert all(
        (c.get('metadata') or {}).get('file_name') != 'report0.pdf'
        for c in store.list_documents()
    )
    store.close()


def test_delete_all_and_stats():
    store, _, _ = _build_store()
    store.add_chunks(make_chunks(3))
    stats = store.get_stats()
    assert stats['backend'] == 'azure_search'
    assert stats['total_chunks'] == 3
    assert stats['dimension'] == 4

    store.delete_all()
    assert len(store) == 0
    store.close()