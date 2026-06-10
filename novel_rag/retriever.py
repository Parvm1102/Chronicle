"""Temporal-aware, spoiler-safe hybrid retriever (foundation for chat).

This is the query side of the RAG foundation. It does NOT generate answers — it
returns ranked, spoiler-safe chunks (with their linked event ids) that a future
chat layer can feed to an LLM.

Flow:
1. Build a Qdrant filter enforcing the temporal rule — never return content past
   the reader's current chapter (and never from a later book in the series).
2. Hybrid retrieve: dense (bge) + sparse (lexical) prefetch fused with RRF.
3. Rerank candidates with a cross-encoder.
4. Apply a gentle temporal-proximity boost so chunks near the current chapter
   rank slightly higher — a nudge, not a takeover.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from qdrant_client import models

from .client import get_client
from .config import RagSettings, get_rag_settings

logger = logging.getLogger(__name__)

# Vector names created by llama-index QdrantVectorStore in hybrid mode. The dense
# name is stable; the sparse name has changed across versions ("text-sparse" ->
# "text-sparse-new"), so it is resolved from the live collection at runtime with
# this as the fallback default.
_DENSE_VECTOR = "text-dense"
_SPARSE_VECTOR_DEFAULT = "text-sparse-new"

# Chapter-equivalent penalty applied per earlier book, so same-book proximity
# always dominates cross-book proximity in the temporal weighting.
_BOOK_DISTANCE = 1000


@dataclass
class RetrievedChunk:
    """A spoiler-safe chunk returned to the caller."""

    text: str
    score: float
    novel_uuid: str
    novel_number: int
    chapter_index: int
    chapter_title: str
    chunk_serial: int
    speakers: list[str] = field(default_factory=list)
    associated_characters: list[str] = field(default_factory=list)
    event_ids: list[int] = field(default_factory=list)
    max_spoiler_level: int = 0


class RagRetriever:
    """Hybrid + temporal retriever over the shared Qdrant collection."""

    def __init__(self, settings: Optional[RagSettings] = None) -> None:
        self._settings = settings or get_rag_settings()
        self._sparse_vector_name: Optional[str] = None
        self._dense = None
        self._sparse = None
        self._reranker = None

    def retrieve(
        self,
        query: str,
        *,
        novel_uuid: str,
        current_chapter: int,
        novel_number: int = 1,
        series_key: Optional[str] = None,
        include_series: bool = True,
        speaker: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Return spoiler-safe, ranked chunks for ``query``.

        Args:
            query: Natural-language query.
            novel_uuid: The novel the reader is currently in.
            current_chapter: Reader's current chapter (``section_index``).
            novel_number: Book order of the current novel within its series.
            series_key: Series id (first book's uuid). Enables cross-book recall.
            include_series: If True and ``series_key`` is set, also retrieve from
                earlier books in the series. Otherwise restrict to this novel.
            speaker: Optional speaker name to restrict results to (for
                character-wise chat).
        """
        flt = self._build_filter(
            novel_uuid=novel_uuid,
            current_chapter=current_chapter,
            novel_number=novel_number,
            series_key=series_key,
            include_series=include_series,
            speaker=speaker,
        )
        candidates = self._hybrid_search(query, flt)
        if not candidates:
            return []
        ranked = self._rerank(query, candidates)
        boosted = self._apply_temporal_boost(ranked, current_chapter, novel_number)
        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted[: self._settings.rag_top_k]

    # ── filter ───────────────────────────────────────────────────────────

    def _build_filter(
        self,
        *,
        novel_uuid: str,
        current_chapter: int,
        novel_number: int,
        series_key: Optional[str],
        include_series: bool,
        speaker: Optional[str],
    ) -> models.Filter:
        # Tenancy: whole series (cross-book) or just this novel.
        if include_series and series_key:
            tenancy = models.FieldCondition(
                key="series_key", match=models.MatchValue(value=series_key)
            )
        else:
            tenancy = models.FieldCondition(
                key="novel_uuid", match=models.MatchValue(value=novel_uuid)
            )

        # Temporal: earlier book, OR same book up to the current chapter.
        earlier_book = models.FieldCondition(
            key="novel_number", range=models.Range(lt=novel_number)
        )
        same_book_so_far = models.Filter(
            must=[
                models.FieldCondition(
                    key="novel_number", match=models.MatchValue(value=novel_number)
                ),
                models.FieldCondition(
                    key="chapter_index", range=models.Range(lte=current_chapter)
                ),
            ]
        )
        temporal = models.Filter(should=[earlier_book, same_book_so_far])

        must: list[Any] = [tenancy, temporal]
        if speaker:
            must.append(
                models.FieldCondition(key="speakers", match=models.MatchValue(value=speaker))
            )
        return models.Filter(must=must)

    # ── hybrid search ─────────────────────────────────────────────────────

    def _resolve_sparse_vector_name(self, client: Any) -> str:
        """Return the collection's sparse vector name, detected once and cached."""
        if self._sparse_vector_name is None:
            self._sparse_vector_name = _SPARSE_VECTOR_DEFAULT
            try:
                info = client.get_collection(self._settings.rag_collection)
                names = list((info.config.params.sparse_vectors or {}).keys())
                if names:
                    self._sparse_vector_name = names[0]
            except Exception as exc:
                logger.debug("Could not resolve sparse vector name (%s) — using default", exc)
        return self._sparse_vector_name

    def _hybrid_search(self, query: str, flt: models.Filter) -> list[Any]:
        dense_vec = self._dense_embed(query)
        sparse_vec = self._sparse_embed(query)
        retrieve_k = self._settings.rag_retrieve_k

        client = get_client(self._settings)
        try:
            if not client.collection_exists(self._settings.rag_collection):
                return []
            sparse_name = self._resolve_sparse_vector_name(client)
            result = client.query_points(
                collection_name=self._settings.rag_collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec, using=_DENSE_VECTOR,
                        filter=flt, limit=retrieve_k,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                        using=sparse_name,
                        filter=flt, limit=retrieve_k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=retrieve_k,
                with_payload=True,
            )
            return list(result.points)
        finally:
            client.close()

    # ── rerank ────────────────────────────────────────────────────────────

    def _rerank(self, query: str, points: list[Any]) -> list[RetrievedChunk]:
        texts = [self._point_text(p) for p in points]
        reranker = self._get_reranker()
        try:
            scores = list(reranker.rerank(query, texts))
        except Exception as exc:
            logger.warning("RAG rerank failed (%s) — using fusion scores", exc)
            scores = [float(getattr(p, "score", 0.0) or 0.0) for p in points]

        chunks = [self._to_chunk(p, score) for p, score in zip(points, scores)]
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks[: self._settings.rag_rerank_k]

    def _apply_temporal_boost(
        self, chunks: list[RetrievedChunk], current_chapter: int, current_novel: int
    ) -> list[RetrievedChunk]:
        alpha = self._settings.rag_temporal_alpha
        decay = max(self._settings.rag_temporal_decay, 1e-6)
        for chunk in chunks:
            distance = (
                (current_novel - chunk.novel_number) * _BOOK_DISTANCE
                + (current_chapter - chunk.chapter_index)
            )
            distance = max(distance, 0)
            chunk.score = chunk.score * (1.0 + alpha * math.exp(-distance / decay))
        return chunks

    # ── embedding / model helpers (lazy, cached) ──────────────────────────

    def _dense_embed(self, query: str):
        if self._dense is None:
            from .embedder import build_dense_embedding

            self._dense = build_dense_embedding(self._settings)
        return self._dense.get_query_embedding(query)

    def _sparse_embed(self, query: str):
        if self._sparse is None:
            from fastembed import SparseTextEmbedding

            self._sparse = SparseTextEmbedding(model_name=self._settings.rag_sparse_model)
        return list(self._sparse.query_embed(query))[0]

    def _get_reranker(self):
        if self._reranker is None:
            from .embedder import build_reranker

            self._reranker = build_reranker(self._settings)
        return self._reranker

    # ── payload helpers ───────────────────────────────────────────────────

    @staticmethod
    def _point_text(point: Any) -> str:
        payload = point.payload or {}
        # A plain "text" field wins if present. Otherwise llama-index stores the
        # node as a JSON blob under "_node_content" whose "text" key holds the
        # actual chunk text — extract it rather than returning raw JSON.
        if payload.get("text"):
            return payload["text"]
        node_content = payload.get("_node_content")
        if node_content:
            try:
                return json.loads(node_content).get("text", "") or ""
            except (json.JSONDecodeError, TypeError):
                return node_content
        return ""

    @staticmethod
    def _to_chunk(point: Any, score: float) -> RetrievedChunk:
        p = point.payload or {}
        return RetrievedChunk(
            text=RagRetriever._point_text(point),
            score=float(score),
            novel_uuid=p.get("novel_uuid", ""),
            novel_number=int(p.get("novel_number", 1)),
            chapter_index=int(p.get("chapter_index", 0)),
            chapter_title=p.get("chapter_title", ""),
            chunk_serial=int(p.get("chunk_serial", 0)),
            speakers=list(p.get("speakers") or []),
            associated_characters=list(p.get("associated_characters") or []),
            event_ids=list(p.get("event_ids") or []),
            max_spoiler_level=int(p.get("max_spoiler_level", 0)),
        )
