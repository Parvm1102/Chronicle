"""Centralised configuration for the novel_rag vector-DB layer.

Driven entirely by environment variables / .env so it behaves identically on
local Docker, a managed Qdrant cluster, or CI. Mirrors the pattern used by
novel_parser.config.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Single source of truth for all novel_rag configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Feature flag ---
    # Pass 3 (vector indexing) only runs after Pass 2 when this is true.
    enable_rag_indexing: bool = False

    # --- Qdrant connection (local Docker or managed cloud) ---
    # Local default matches the docker-compose `qdrant` service. For a managed
    # cluster, set QDRANT_URL=https://<cluster>.qdrant.io and QDRANT_API_KEY.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_prefer_grpc: bool = False
    rag_collection: str = "novel_rag"

    # --- Models (all ONNX via FastEmbed — no torch required) ---
    rag_embed_model: str = "BAAI/bge-base-en-v1.5"   # 768-dim dense vectors
    rag_embed_dim: int = 768
    rag_sparse_model: str = "Qdrant/bm25"            # keyword / lexical vectors
    rag_rerank_model: str = "BAAI/bge-reranker-base"  # cross-encoder reranker

    # --- GPU acceleration (optional) ---
    # When true, dense embedding + cross-encoder reranking run on a torch /
    # sentence-transformers backend on ``rag_device`` (e.g. CUDA) instead of the
    # CPU-only FastEmbed ONNX path. Requires torch + sentence-transformers +
    # llama-index-embeddings-huggingface to be installed. The sparse (BM25)
    # vectors always stay on FastEmbed/CPU — they are trivial to compute.
    rag_use_gpu: bool = False
    rag_device: str = "cuda"

    # --- Chunking ---
    rag_chunk_size: int = 512      # target tokens per chunk
    rag_chunk_overlap: int = 64    # token overlap between adjacent chunks

    # --- Retrieval ---
    rag_retrieve_k: int = 40       # candidates pulled from hybrid search
    rag_rerank_k: int = 20         # kept after cross-encoder rerank
    rag_top_k: int = 8             # final results returned to the caller

    # --- Temporal proximity re-weighting ---
    # final = rerank_score * (1 + alpha * exp(-distance / decay)) where
    # distance is the chapter gap between a chunk and the current chapter.
    # Keep alpha modest so proximity nudges rather than dominates relevance.
    rag_temporal_alpha: float = 0.15
    rag_temporal_decay: float = 5.0

    @property
    def qdrant_api_key_or_none(self) -> Optional[str]:
        return self.qdrant_api_key or None


_settings: Optional[RagSettings] = None


def get_rag_settings() -> RagSettings:
    """Return the global RagSettings singleton, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = RagSettings()
    return _settings
