"""Series manager — CRUD for novel series and cross-book context loading.

A series is a user-defined ordered group of novels.  When parsing book N in a
series, the parser can load characters and profiles from books 1..N-1 to
maintain continuity and avoid spoilers from book N+1 onwards.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class SeriesManager:
    """Manage novel series and provide cross-book context."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    # ── CRUD ───────────────────────────────────────────────────────────────

    def create_series(self, name: str, description: str = "") -> int:
        """Create a new series, return its id."""
        series_id = self._db.create_series(name, description)
        logger.info("Created series '%s' (id=%d)", name, series_id)
        return series_id

    def add_book(
        self, series_id: int, novel_uuid: str, book_order: int
    ) -> None:
        """Add a novel to a series at the given position.

        The novel must already exist in novels_meta.
        """
        meta = self._db.get_novel_meta(novel_uuid)
        if not meta:
            raise ValueError(
                f"Novel with uuid '{novel_uuid}' not found in novels_meta. "
                "Ingest the novel first."
            )
        self._db.add_novel_to_series(series_id, meta["id"], book_order)
        logger.info(
            "Added novel '%s' to series %d at position %d",
            meta["novel_title"], series_id, book_order,
        )

    def get_series_for_novel(
        self, novel_meta_id: int
    ) -> Optional[dict[str, Any]]:
        """Return series info for a novel, or None if standalone."""
        return self._db.get_series_for_novel(novel_meta_id)

    def list_series_books(self, series_id: int) -> list[dict[str, Any]]:
        """All books in a series, ordered."""
        return self._db.get_series_novels(series_id)

    # ── cross-book context ─────────────────────────────────────────────────

    def get_prior_characters(
        self, series_id: int, current_book_order: int
    ) -> list[dict[str, Any]]:
        """Characters from books before the current one (spoiler-safe).

        Returns characters from books 1..current_book_order-1.
        """
        if current_book_order <= 1:
            return []
        return self._db.get_characters_up_to_book(
            series_id, current_book_order - 1
        )

    def get_prior_profiles(
        self, series_id: int, current_book_order: int
    ) -> list[dict[str, Any]]:
        """Latest character profiles from books before the current one."""
        if current_book_order <= 1:
            return []
        return self._db.get_latest_profiles_for_series(
            series_id, current_book_order - 1
        )

    def get_series_id_for_novel(self, novel_meta_id: int) -> Optional[int]:
        """Return the series_id for a novel, or None."""
        info = self._db.get_series_for_novel(novel_meta_id)
        return info["id"] if info else None

    def get_book_order(self, novel_meta_id: int) -> int:
        """Return the book_order for a novel in its series, or 1 if standalone."""
        info = self._db.get_series_for_novel(novel_meta_id)
        return info["book_order"] if info else 1
