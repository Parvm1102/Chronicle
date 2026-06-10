"""Pass 3 — build LlamaIndex nodes from chunks and upsert into Qdrant.

Produces a hybrid (dense + sparse) index. Dense vectors come from a FastEmbed
ONNX model (``bge-base-en-v1.5`` by default); sparse vectors come from a
FastEmbed lexical model for keyword matching. Point ids are deterministic
(``uuid5(novel_uuid + chunk_serial)``) so re-indexing is idempotent.

Only structural text is embedded — all the rich metadata (series/novel/chapter
serials, speakers, characters, event ids, spoiler level) is stored as payload
for filtering and downstream chat, never folded into the embedding text.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from .chunker import Chunk, chunk_sections
from .client import delete_novel_points, ensure_payload_indexes, get_client
from .config import RagSettings, get_rag_settings
from .metadata import enrich_chunks

logger = logging.getLogger(__name__)

# Stable namespace for deterministic point ids.
_POINT_NAMESPACE = uuid.UUID("b6f1e3a2-0c4d-4e6f-8a1b-2c3d4e5f6a7b")

# Metadata keys that must never be embedded or shown to the LLM as context text.
_NON_TEXT_METADATA = [
    "series_key", "series_name", "novel_uuid", "novel_number",
    "chapter_index", "chapter_title", "chunk_serial", "chunk_index_in_chapter",
    "char_start", "char_end", "speakers", "associated_characters",
    "event_ids", "max_spoiler_level", "parent_id",
]


class RagIndexer:
    """Index a novel's full text into the shared Qdrant collection."""

    def __init__(self, settings: Optional[RagSettings] = None) -> None:
        self._settings = settings or get_rag_settings()

    def run(
        self,
        novel_uuid: str,
        sections: list[dict[str, Any]],
        *,
        series_key: str,
        series_name: str = "",
        novel_number: int = 1,
        db: Any = None,
        novel_meta_id: Optional[int] = None,
    ) -> int:
        """Chunk, enrich, and upsert a novel. Returns the number of chunks indexed.

        Args:
            novel_uuid: SQLite novel uuid (per-novel tenancy key).
            sections: Section dicts (``section_index``, ``title``, ``text``).
            series_key: Stable series id — the first book's uuid (== novel_uuid
                for a standalone book).
            series_name: Human-readable series name (payload only).
            novel_number: Book order within the series (1-based).
            db: Optional parser ``DatabaseManager`` for metadata enrichment.
            novel_meta_id: Parser-side novel id (required for enrichment).
        """
        logger.info("RAG: chunking %d sections for novel %s", len(sections), novel_uuid)
        chunks = chunk_sections(
            sections,
            chunk_size=self._settings.rag_chunk_size,
            chunk_overlap=self._settings.rag_chunk_overlap,
        )
        if not chunks:
            logger.info("RAG: no chunks for novel %s — skipping", novel_uuid)
            return 0
        logger.info("RAG: produced %d chunks for novel %s", len(chunks), novel_uuid)

        if db is not None and novel_meta_id is not None:
            logger.info("RAG: enriching %d chunks with parser metadata", len(chunks))
            try:
                enrich_chunks(db, novel_meta_id, sections, chunks)
            except Exception as exc:
                logger.warning("RAG: metadata enrichment failed (%s) — indexing without it", exc)

        nodes = [
            self._to_node(chunk, novel_uuid, series_key, series_name, novel_number)
            for chunk in chunks
        ]

        logger.info("RAG: connecting to Qdrant at %s", self._settings.qdrant_url)
        client = get_client(self._settings)
        try:
            # Wipe any prior points for this novel so re-index is clean.
            logger.info("RAG: clearing prior points for novel %s", novel_uuid)
            delete_novel_points(novel_uuid, client=client, settings=self._settings)
            logger.info(
                "RAG: embedding + upserting %d chunks (loading models on first run — "
                "this can take a while)",
                len(nodes),
            )
            self._upsert(client, nodes)
            ensure_payload_indexes(client, self._settings)
        finally:
            client.close()

        logger.info("RAG: indexed %d chunks for novel %s", len(nodes), novel_uuid)
        return len(nodes)

    # ── internals ──────────────────────────────────────────────────────────

    def _to_node(
        self,
        chunk: Chunk,
        novel_uuid: str,
        series_key: str,
        series_name: str,
        novel_number: int,
    ) -> Any:
        from llama_index.core.schema import TextNode

        metadata = {
            "series_key": series_key,
            "series_name": series_name,
            "novel_uuid": novel_uuid,
            "novel_number": int(novel_number),
            "chapter_index": chunk.chapter_index,
            "chapter_title": chunk.chapter_title,
            "chunk_serial": chunk.chunk_serial,
            "chunk_index_in_chapter": chunk.chunk_index_in_chapter,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "speakers": chunk.speakers,
            "associated_characters": chunk.associated_characters,
            "event_ids": chunk.event_ids,
            "max_spoiler_level": chunk.max_spoiler_level,
            "parent_id": None,  # reserved for future hierarchical chunking
        }
        node_id = str(uuid.uuid5(_POINT_NAMESPACE, f"{novel_uuid}:{chunk.chunk_serial}"))
        return TextNode(
            id_=node_id,
            text=chunk.text,
            metadata=metadata,
            excluded_embed_metadata_keys=list(_NON_TEXT_METADATA),
            excluded_llm_metadata_keys=list(_NON_TEXT_METADATA),
        )

    def _upsert(self, client: Any, nodes: list[Any]) -> None:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        from .embedder import build_dense_embedding

        vector_store = QdrantVectorStore(
            collection_name=self._settings.rag_collection,
            client=client,
            enable_hybrid=True,
            fastembed_sparse_model=self._settings.rag_sparse_model,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = build_dense_embedding(self._settings)
        VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
        )
