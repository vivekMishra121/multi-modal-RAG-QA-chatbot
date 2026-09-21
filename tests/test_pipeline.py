"""RAGPipeline end-to-end behaviour with deterministic fakes (no network)."""

from __future__ import annotations

from typing import Any

from rag.qa_generation.pipeline import RAGPipeline
from rag.qa_generation.qa_chain import QAChain
from rag.retrieval.retriever import Retriever

from .conftest import FakeEmbedder, FakeStore


def _stub_generate_answer(self, question: str, context: str, history=None) -> dict[str, Any]:
    assert question
    assert context
    return {'answer': 'Stubbed answer', 'success': True, 'error': None}


def test_pipeline_query_returns_sources_and_strategy(
    monkeypatch, off_network_settings, tmp_path
):
    monkeypatch.setattr(QAChain, "generate_answer", _stub_generate_answer)

    store = FakeStore()
    retriever = Retriever(
        store,
        use_reranker=False,
        use_query_expansion=False,
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        embedder=FakeEmbedder(),
        api_key="sk-test",
        max_context_tokens=500,
        retrieval_strategy="vector",
        top_k=2,
    )

    result = pipeline.query("what is topic c?")

    assert result['success'] is True
    assert result['answer'] == 'Stubbed answer'
    assert result['num_chunks_used'] == 2
    assert result['retrieval_strategy'] == 'vector'
    assert len(result['sources']) == 2
    # FakeEmbedder ranks the FAKE chunk with the largest numeric embedding on top.
    assert result['sources'][0]['file_name'] == 'doc2.txt'


def test_pipeline_empty_retrieval_handled(off_network_settings):
    store = FakeStore(chunks=[])
    retriever = Retriever(store, use_reranker=False, use_query_expansion=False)
    pipeline = RAGPipeline(
        retriever=retriever,
        embedder=FakeEmbedder(),
        api_key="sk-test",
        retrieval_strategy="vector",
    )
    result = pipeline.query("anything")
    assert result['success'] is False
    assert result['answer'] == "No relevant information found in the documents."


def test_pipeline_passes_history_to_qa(monkeypatch, off_network_settings):
    captured = {}

    def _capture(self, question, context, history=None):
        captured['history'] = history
        return {'answer': 'A', 'success': True, 'error': None}

    monkeypatch.setattr(QAChain, "generate_answer", _capture)

    pipeline = RAGPipeline(
        retriever=Retriever(FakeStore(), use_reranker=False, use_query_expansion=False),
        embedder=FakeEmbedder(),
        api_key="sk-test",
        retrieval_strategy="vector",
        top_k=2,
    )
    history = [{"role": "user", "content": "prior q"}, {"role": "assistant", "content": "prior a"}]
    pipeline.query("now?", history=history)
    assert captured['history'] == history


def test_format_history_and_prompt_include_history(monkeypatch, off_network_settings):
    rendered = {}

    def _capture(self, question, context, history=None):
        rendered['prompt'] = self.prompt.format(
            question=question,
            context=context,
            history_section=QAChain.format_history(history),
        )
        return {'answer': 'A', 'success': True, 'error': None}

    monkeypatch.setattr(QAChain, "generate_answer", _capture)

    pipeline = RAGPipeline(
        retriever=Retriever(FakeStore(), use_reranker=False, use_query_expansion=False),
        embedder=FakeEmbedder(),
        api_key="sk-test",
        retrieval_strategy="vector",
        top_k=1,
    )
    history = [{"role": "user", "content": "from before"}]
    pipeline.query("q2", history=history)

    assert "Previous conversation:" in rendered['prompt']
    assert "User: from before" in rendered['prompt']
    assert "Question: q2" in rendered['prompt']