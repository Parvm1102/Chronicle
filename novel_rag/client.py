"""Qdrant client factory, collection bootstrap, and per-novel cleanup.

A single collection (``rag_collection``) holds chunks for every novel. Tenancy
is by payload: ``novel_uuid`` isolates a book and ``series_key`` links books in
a series. Temporal filtering relies on payload indexes over ``novel_number`` and
``chapter_index`` so the "never beyond current chapter" rule is enforced
server-side.

The collection itself (named dense + sparse vectors) is created by LlamaIndex's
``QdrantVectorStore`` on first write; this module only adds the payload indexes
and provides deletion helpers.
"""

from __future__ import annotations

import logging
from typing import Optional

from qdrant_client import QdrantClient, models

from .config import RagSettings, get_rag_settings

logger = logging.getLogger(__name__)

# Payload fields we filter on. Keyword for exact-match tenancy, integer for the
# temporal range comparisons (novel_number, chapter_index).
_KEYWORD_INDEX_FIELDS = ("novel_uuid", "series_key")
_INTEGER_INDEX_FIELDS = ("novel_number", "chapter_index", "chunk_serial", "max_spoiler_level")


def get_client(settings: Optional[RagSettings] = None) -> QdrantClient:
    """Create a Qdrant client for the configured local or cloud instance."""
    settings = settings or get_rag_settings()
    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key_or_none,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )


def ensure_payload_indexes(
    client: QdrantClient,
    settings: Optional[RagSettings] = None,
) -> None:
    """Create payload indexes used for tenancy + temporal filtering.

    Idempotent: skips fields that are already indexed. Safe to call after every
    index run. No-op if the collection does not exist yet.
    """
    settings = settings or get_rag_settings()
    collection = settings.rag_collection
    if not client.collection_exists(collection):
        return

    for field in _KEYWORD_INDEX_FIELDS:
        _try_create_index(client, collection, field, models.PayloadSchemaType.KEYWORD)
    for field in _INTEGER_INDEX_FIELDS:
        _try_create_index(client, collection, field, models.PayloadSchemaType.INTEGER)


def _try_create_index(
    client: QdrantClient,
    collection: str,
    field: str,
    schema: "models.PayloadSchemaType",
) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=schema,
        )
    except Exception as exc:  # already exists or transient — log and continue
        logger.debug("Payload index for %s skipped: %s", field, exc)


def delete_novel_points(
    novel_uuid: str,
    client: Optional[QdrantClient] = None,
    settings: Optional[RagSettings] = None,
) -> None:
    """Delete all chunk points belonging to a single novel.

    Used on re-index (wipe-before-write) and on novel deletion. Silently
    no-ops if the collection or points are absent.
    """
    settings = settings or get_rag_settings()
    own_client = client is None
    client = client or get_client(settings)
    try:
        if not client.collection_exists(settings.rag_collection):
            return
        client.delete(
            collection_name=settings.rag_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="novel_uuid",
                            match=models.MatchValue(value=novel_uuid),
                        )
                    ]
                )
            ),
        )
        logger.info("Deleted Qdrant points for novel %s", novel_uuid)
    except Exception as exc:
        logger.warning("Failed to delete Qdrant points for %s: %s", novel_uuid, exc)
    finally:
        if own_client:
            client.close()
