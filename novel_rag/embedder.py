"""Backend selection for dense embedding and cross-encoder reranking.

Two interchangeable backends share one collection layout (named ``text-dense`` /
``text-sparse`` vectors):

* **CPU (default)** — FastEmbed ONNX. No torch required.
* **GPU (``rag_use_gpu``)** — torch / sentence-transformers on ``rag_device``,
  reusing the standard HuggingFace ``bge`` weights. Much faster for bulk
  indexing on a CUDA device.

Both paths use the *same* model names (``rag_embed_model`` /
``rag_rerank_model``), so dense vectors stay dimension-compatible and a novel
indexed on GPU can be queried on CPU and vice-versa.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Optional

from .config import RagSettings, get_rag_settings

logger = logging.getLogger(__name__)

# hf_xet can spuriously re-download large weights that are already cached via the
# classic LFS path, throttled through the XET CAS (very slow). Disabling it makes
# huggingface_hub recognise the existing blob and load instantly. Honour an
# explicit user override if one is already set.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Recommended retrieval instruction for the bge-*-en family. Applied to queries
# only (not passages) so query and passage spaces align as the model authors
# intend.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def build_dense_embedding(settings: Optional[RagSettings] = None) -> Any:
    """Return a LlamaIndex embedding model for dense vectors.

    GPU path uses ``HuggingFaceEmbedding`` on ``rag_device``; CPU path uses
    ``FastEmbedEmbedding`` (ONNX).
    """
    settings = settings or get_rag_settings()
    if settings.rag_use_gpu:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        logger.info(
            "RAG: dense embedder = HuggingFace %s on %s (GPU path)",
            settings.rag_embed_model,
            settings.rag_device,
        )
        return HuggingFaceEmbedding(
            model_name=settings.rag_embed_model,
            device=settings.rag_device,
            query_instruction=_BGE_QUERY_INSTRUCTION,
        )

    from llama_index.embeddings.fastembed import FastEmbedEmbedding

    logger.info("RAG: dense embedder = FastEmbed %s (CPU path)", settings.rag_embed_model)
    return FastEmbedEmbedding(model_name=settings.rag_embed_model)


class _CrossEncoderReranker:
    """Adapts a sentence-transformers ``CrossEncoder`` to the ``.rerank`` API."""

    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, texts: list[str]) -> Iterable[float]:
        pairs = [[query, text] for text in texts]
        return [float(s) for s in self._model.predict(pairs)]


def build_reranker(settings: Optional[RagSettings] = None) -> Any:
    """Return a reranker exposing ``rerank(query, texts) -> Iterable[float]``.

    GPU path uses a torch ``CrossEncoder`` on ``rag_device``; CPU path uses
    FastEmbed's ONNX ``TextCrossEncoder``.
    """
    settings = settings or get_rag_settings()
    if settings.rag_use_gpu:
        logger.info(
            "RAG: reranker = CrossEncoder %s on %s (GPU path)",
            settings.rag_rerank_model,
            settings.rag_device,
        )
        return _CrossEncoderReranker(settings.rag_rerank_model, settings.rag_device)

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    logger.info("RAG: reranker = FastEmbed %s (CPU path)", settings.rag_rerank_model)
    return TextCrossEncoder(model_name=settings.rag_rerank_model)
