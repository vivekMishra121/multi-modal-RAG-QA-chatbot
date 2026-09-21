"""BGE-M3 embedder providing dense + sparse (BM25-style) embeddings.

BGE-M3 is an open-source, multilingual embedding model that returns three
kinds of vectors:
  * dense_vecs      -> semantic similarity vectors (1024-dim)
  * lexical_weights -> sparse, token-level weights (BM25-style lexical match)
  * colbert_vecs     -> multi-vector (optional, unused here)

We use the dense + sparse combination for Qdrant hybrid search
(dense semantic + sparse keyword/BM25 fusion).
"""

import logging
from typing import Optional

from .base import EmbedderInterface

logger = logging.getLogger(__name__)

# Default model: excellent open-source multilingual with native sparse support
DEFAULT_MODEL = "BAAI/bge-m3"


class BGEM3Embedder(EmbedderInterface):
    """Dense + sparse embedding using BAAI/bge-m3 (open source)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_fp16: bool = False,
        max_length: int = 1024,
        use_faiss: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-m3)
            device: 'cuda', 'cpu', or None for auto-detection
            use_fp16: Use half precision (GPU only)
            max_length: Max sequence length for encoding
            use_faiss: Reuse FAISS dense index internally (optional)
        """
        self.model_name = model_name
        self._dimension = 1024
        self.max_length = max_length

        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
            use_faiss=use_faiss,
        )
        # Real dimension may differ for custom models; probe on first encode
        self._probing = True
        logger.info(f"Loaded open-source embedding model: {model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning only dense vectors."""
        if not texts:
            return []
        outputs = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            max_length=self.max_length,
        )
        return outputs["dense_vecs"].tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query, returning the dense vector."""
        return self.embed([query])[0]

    def embed_query_with_sparse(self, query: str) -> tuple[list[float], dict[str, float]]:
        """Embed a single query, returning (dense_vector, sparse_weights)."""
        dense, sparse = self.embed_with_sparse([query])
        return dense[0], sparse[0]

    def embed_with_sparse(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[str, float]]]:
        """Embed a batch of texts, returning (dense_vecs, sparse_weights).

        sparse_weights is a list of dicts mapping token ids -> lexical weight
        (BM25-style), compatible with Qdrant sparse vectors.
        """
        if not texts:
            return [], []
        outputs = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
            max_length=self.max_length,
        )
        dense = outputs["dense_vecs"].tolist()
        sparse = outputs["lexical_weights"]
        # lexical_weights are already list[dict[int, float]]
        if self._probing and dense:
            self._dimension = len(dense[0])
            self._probing = False
        return dense, sparse

    @property
    def embedding_dimension(self) -> int:
        return self._dimension
