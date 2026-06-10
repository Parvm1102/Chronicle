"""Back-fill: index novels that were parsed before the RAG layer existed.

Usage::

    python -m novel_rag.backfill            # index every completed novel
    python -m novel_rag.backfill <uuid> ... # index specific novels by uuid

Reads chapter text from the SQLite library store and enrichment data from the
parser's PostgreSQL store, then runs the same Pass 3 indexing used by the
pipeline. Idempotent — re-running re-indexes cleanly.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from .config import get_rag_settings
from .indexer import RagIndexer

logger = logging.getLogger(__name__)


def _resolve_series(db: Any, novel_meta_id: int, novel_uuid: str) -> tuple[str, str, int]:
    """Return (series_key, series_name, novel_number) for a parsed novel."""
    series_key, series_name, novel_number = novel_uuid, "", 1
    info = db.get_series_for_novel(novel_meta_id)
    if info:
        series_name = info.get("name", "")
        novel_number = int(info.get("book_order", 1))
        books = db.get_series_novels(info["id"])
        if books:
            series_key = books[0].get("novel_uuid", novel_uuid)
    return series_key, series_name, novel_number


def backfill(uuids: Optional[list[str]] = None) -> None:
    """Index the given novel uuids, or all parse-complete novels if None."""
    from novel_parser.config import get_settings as get_parser_settings
    from novel_parser.database import DatabaseManager
    from novel_reader.storage import LibraryStore

    store = LibraryStore()
    db = DatabaseManager(get_parser_settings())
    db.init_schema()
    indexer = RagIndexer(get_rag_settings())

    novels = store.list_novels(include_archived=True)
    if uuids:
        wanted = set(uuids)
        novels = [n for n in novels if n["uuid"] in wanted]

    logger.info("Back-fill starting — %d novel(s) to consider", len(novels))
    indexed = 0
    try:
        for novel in novels:
            novel_uuid = novel["uuid"]
            logger.info("Processing %s (%s)", novel.get("title", "?"), novel_uuid)
            meta = db.get_novel_meta(novel_uuid)
            if not meta or meta.get("parse_status") != "complete":
                logger.info("Skipping %s — not parse-complete", novel_uuid)
                continue

            sections = store.list_sections_full(novel["id"])
            if not sections:
                logger.info("Skipping %s — no sections", novel_uuid)
                continue
            logger.info("Loaded %d sections for %s", len(sections), novel_uuid)

            series_key, series_name, novel_number = _resolve_series(
                db, meta["id"], novel_uuid
            )
            count = indexer.run(
                novel_uuid,
                sections,
                series_key=series_key,
                series_name=series_name,
                novel_number=novel_number,
                db=db,
                novel_meta_id=meta["id"],
            )
            indexed += 1
            logger.info("Back-filled %s — %d chunks", novel["title"], count)
    finally:
        db.close()

    logger.info("Back-fill complete — %d novels indexed", indexed)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:]
    backfill(args or None)


if __name__ == "__main__":
    main()
