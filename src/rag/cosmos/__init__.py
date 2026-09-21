"""Azure Cosmos DB (NoSQL API) integration for memory and project metadata."""

from .store import CosmosNoSQLStore, build_cosmos_store

__all__ = ["CosmosNoSQLStore", "build_cosmos_store"]