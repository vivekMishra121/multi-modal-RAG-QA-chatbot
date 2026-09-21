"""Configuration tests - defaults, Azure switchability, env aliases.

Note: pydantic-settings bakes environment values into the model validator on the
first instantiation of a class in a process, so mid-process ``setenv`` is NOT
reflected in later ``Settings()`` instances. Tests that exercise environment
reading therefore run a fresh subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rag.core.config import (
    AzureSearchSettings,
    AzureSettings,
    EmbeddingSettings,
    LLMSettings,
    Settings,
    VectorStoreSettings,
)

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _probe_env(**extra: str) -> dict[str, str]:
    """Subprocess env with ``src`` on PYTHONPATH (src-layout)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_SRC, env.get("PYTHONPATH", "")])
    env.update(extra)
    return env


def _run_settings_probe(env: dict[str, str]) -> str:
    """Run ``Settings().llm.provider`` in a fresh process with given env."""
    code = (
        f"import os; os.environ.update({env!r}); "
        "from rag.core.config import Settings; "
        "print(Settings().llm.provider, Settings().embedding.api_key, Settings().retrieval.top_k)"
    )
    return subprocess.check_output(  # noqa: S603 - fixed static snippet
        [sys.executable, "-c", code],
        text=True,
        env=_probe_env(),
    ).strip()


def test_defaults():
    s = Settings()
    assert s.llm.provider == "openai"
    assert s.embedding.provider == "openai"
    assert s.vector_store.backend == "azure_search"
    assert not s.is_production
    assert not s.is_azure_llm


def test_azure_provider_switch():
    """All three Azure switches are pure configuration."""
    s = Settings(
        llm=LLMSettings(provider="azure_openai", model_name="gpt-4o"),
        embedding=EmbeddingSettings(provider="azure_openai", deployment_name="text-embedding-3-large"),
        azure=AzureSettings(
            openai_endpoint="https://example.openai.azure.com",
            openai_api_key="k",
        ),
        vector_store=VectorStoreSettings(
            backend="azure_search",
            azure_search=AzureSearchSettings(endpoint="https://x.search.windows.net", api_key="k2"),
        ),
    )
    assert s.is_azure_llm
    assert s.is_azure_embedding
    assert s.is_azure_search
    assert s.azure.openai_endpoint == "https://example.openai.azure.com"
    assert s.embedding.deployment_name == "text-embedding-3-large"


def test_legacy_env_aliases_in_fresh_process():
    out = _run_settings_probe({"OPENAI_API_KEY": "legacy-key", "TOP_K": "7"})
    provider, api_key, top_k = out.split()
    assert api_key == "legacy-key"
    assert top_k == "7"
    assert provider == "openai"


def test_azure_env_prefixes_in_fresh_process():
    env = {
        "LLM_PROVIDER": "azure_openai",
        "EMBEDDING_PROVIDER": "azure_openai",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "VECTOR_STORE_BACKEND": "azure_search",
    }
    code = (
        f"import os; os.environ.update({env!r}); "
        "from rag.core.config import Settings; "
        "s = Settings(); "
        "print(s.llm.provider, s.embedding.provider, s.is_azure_search)"
    )
    out = subprocess.check_output(  # noqa: S603 - fixed static snippet
        [sys.executable, "-c", code],
        text=True,
        env=_probe_env(),
    ).strip()
    assert out == "azure_openai azure_openai True"


def test_nested_model_nullable_annotations_are_py39_safe():
    """Regression: pydantic must evaluate nullable annotations on 3.9."""
    from rag.core.config import QdrantSettings

    q = QdrantSettings()
    assert q.location == ":memory:"
    assert q.path is None