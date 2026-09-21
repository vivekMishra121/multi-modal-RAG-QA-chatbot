"""Central application configuration — Azure OpenAI + Azure AI Search stack."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env once — all sub-settings classes read from os.environ after this
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    provider: Literal["openai", "azure_openai", "groq"] = "azure_openai"
    model_name: str = "gpt-4o-mini"
    deployment_name: Optional[str] = None
    temperature: float = 0.0
    max_context_tokens: int = 4096
    request_timeout: float = 60.0
    max_retries: int = 3
    streaming: bool = True
    api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"),
    )


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", extra="ignore")

    provider: Literal["openai", "azure_openai", "sentence_transformers", "bge_m3"] = "azure_openai"
    model_name: str = "text-embedding-3-small"
    deployment_name: Optional[str] = None
    dimensions: int = 1536
    batch_size: int = 100
    max_retries: int = 3


class AzureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_", extra="ignore")

    openai_endpoint: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_api_version: str = "2024-02-01"
    embedding_deployment: str = "text-embedding-3-small"
    chat_deployment: str = "gpt-5.6-terra"


class AzureSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_SEARCH_", extra="ignore")

    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    index_name: str = "rag-chunks"
    semantic_rank: bool = False
    semantic_config_name: str = "rag-semantic-config"

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)


class CosmosSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COSMOS_", extra="ignore")

    endpoint: Optional[str] = None
    key: Optional[str] = None
    database_name: str = "rag-chat"
    conversations_container: str = "conversations"
    projects_container: str = "projects"
    files_container: str = "files"
    max_history_turns: int = 6

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.key)


class BlobSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_BLOB_", extra="ignore")

    connection_string: Optional[str] = None
    container_name: str = "rag-uploads"

    @property
    def is_configured(self) -> bool:
        return bool(self.connection_string)


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", extra="ignore")

    backend: Literal["azure_search"] = "azure_search"
    azure_search: AzureSearchSettings = AzureSearchSettings()


class DocIntelligenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DOC_INTEL_", extra="ignore")

    endpoint: str = ""
    key: str = ""
    model_id: str = "prebuilt-layout"
    enabled: bool = False


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNK_", extra="ignore")

    text_chunk_size: int = 1000
    text_chunk_overlap: int = 200


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", extra="ignore")

    top_k: int = Field(default=5, validation_alias=AliasChoices("RETRIEVAL_TOP_K", "TOP_K"))
    strategy: Literal["auto", "standard", "mmr", "hybrid", "fusion"] = "auto"
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_query_expansion: bool = True
    fetch_multiplier: int = 5


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", extra="ignore")

    level: str = "INFO"
    json_format: bool = False


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    title: str = "RAG QA Chatbot API"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    request_timeout: float = 120.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    environment: str = "development"
    app_name: str = "RAG QA Chatbot"

    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    azure: AzureSettings = AzureSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    doc_intel: DocIntelligenceSettings = DocIntelligenceSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    cosmos: CosmosSettings = CosmosSettings()
    blob: BlobSettings = BlobSettings()
    logging: LoggingSettings = LoggingSettings()
    api: APISettings = APISettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
