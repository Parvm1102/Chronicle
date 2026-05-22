from __future__ import annotations

import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DATABASE_PATH, NOVELS_DIR, ensure_app_dirs
from .models import ParsedBook


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LibraryStore:
    def __init__(self, db_path: Path = DATABASE_PATH) -> None:
        ensure_app_dirs()
        self.db_path = db_path
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS novels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT 'Unknown author',
                    file_format TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    rag_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    status_message TEXT NOT NULL DEFAULT '',
                    archived INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    novel_id INTEGER NOT NULL REFERENCES novels(id) ON DELETE CASCADE,
                    section_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    html TEXT NOT NULL DEFAULT '',
                    source_locator TEXT NOT NULL,
                    UNIQUE(novel_id, section_index)
                );

                CREATE TABLE IF NOT EXISTS progress (
                    novel_id INTEGER PRIMARY KEY REFERENCES novels(id) ON DELETE CASCADE,
                    section_index INTEGER NOT NULL DEFAULT 0,
                    scroll_hint REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    novel_id INTEGER NOT NULL REFERENCES novels(id) ON DELETE CASCADE,
                    section_index INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS highlights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    novel_id INTEGER NOT NULL REFERENCES novels(id) ON DELETE CASCADE,
                    section_index INTEGER NOT NULL,
                    quote TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    color TEXT NOT NULL DEFAULT 'gold',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dictionary_lookups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    novel_id INTEGER REFERENCES novels(id) ON DELETE SET NULL,
                    section_index INTEGER NOT NULL DEFAULT 0,
                    query TEXT NOT NULL,
                    search_url TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                """
            )
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(sections)")}
            if "html" not in columns:
                conn.execute("ALTER TABLE sections ADD COLUMN html TEXT NOT NULL DEFAULT ''")
            novel_columns = {row["name"] for row in conn.execute("PRAGMA table_info(novels)")}
            if "archived" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            if "completed_at" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN completed_at TEXT")
            if "cover_image" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN cover_image TEXT NOT NULL DEFAULT ''")
            if "series" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN series TEXT NOT NULL DEFAULT ''")
            if "genres" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN genres TEXT NOT NULL DEFAULT ''")
            if "file_size" not in novel_columns:
                conn.execute("ALTER TABLE novels ADD COLUMN file_size INTEGER NOT NULL DEFAULT 0")

    def create_novel_record(self, source_path: Path) -> int:
        book_uuid = uuid.uuid4().hex
        novel_dir = NOVELS_DIR / book_uuid
        uploads_dir = novel_dir / "source"
        rag_dir = novel_dir / "rag"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        rag_dir.mkdir(parents=True, exist_ok=True)
        stored_path = uploads_dir / source_path.name
        shutil.copy2(source_path, stored_path)
        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO novels
                    (uuid, title, author, file_format, original_filename, stored_path, rag_path, status, status_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 'Waiting to parse book', ?, ?)
                """,
                (
                    book_uuid,
                    source_path.stem,
                    "Unknown author",
                    source_path.suffix.lower().lstrip("."),
                    source_path.name,
                    str(stored_path),
                    str(rag_dir),
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def set_status(self, novel_id: int, status: str, message: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE novels SET status = ?, status_message = ?, updated_at = ? WHERE id = ?",
                (status, message, utc_now(), novel_id),
            )

    def save_parsed_book(self, novel_id: int, parsed: ParsedBook) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute("DELETE FROM sections WHERE novel_id = ?", (novel_id,))
            conn.executemany(
                """
                INSERT INTO sections (novel_id, section_index, title, text, html, source_locator)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (novel_id, section.index, section.title, section.text, section.html, section.source_locator)
                    for section in parsed.sections
                ],
            )
            conn.execute(
                """
                UPDATE novels
                SET title = ?, author = ?, file_format = ?, status = 'complete',
                    status_message = ?, updated_at = ?
                WHERE id = ?
                """,
                (parsed.title, parsed.author, parsed.file_format, f"Ready: {len(parsed.sections)} sections", now, novel_id),
            )
            conn.execute(
                """
                INSERT INTO progress (novel_id, section_index, updated_at)
                VALUES (?, 0, ?)
                ON CONFLICT(novel_id) DO NOTHING
                """,
                (novel_id, now),
            )

    def list_novels(self, include_archived: bool = False) -> list[dict[str, Any]]:
        where = "" if include_archived else "WHERE n.archived = 0"
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT n.*, COALESCE(p.section_index, 0) AS progress_section,
                       (SELECT COUNT(*) FROM sections s WHERE s.novel_id = n.id) AS section_count
                FROM novels n
                LEFT JOIN progress p ON p.novel_id = n.id
                {where}
                ORDER BY n.updated_at DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def get_novel(self, novel_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT n.*, COALESCE(p.section_index, 0) AS progress_section
                FROM novels n
                LEFT JOIN progress p ON p.novel_id = n.id
                WHERE n.id = ?
                """,
                (novel_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_section(self, novel_id: int, section_index: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT s.*, n.title AS novel_title, n.author, n.status
                FROM sections s
                JOIN novels n ON n.id = s.novel_id
                WHERE s.novel_id = ? AND s.section_index = ?
                """,
                (novel_id, section_index),
            ).fetchone()
            return dict(row) if row else None

    def list_sections(self, novel_id: int) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT section_index, title
                FROM sections
                WHERE novel_id = ?
                ORDER BY section_index
                """,
                (novel_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def section_count(self, novel_id: int) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM sections WHERE novel_id = ?", (novel_id,)).fetchone()[0])

    def first_section_text(self, novel_id: int) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT text FROM sections WHERE novel_id = ? ORDER BY section_index LIMIT 1",
                (novel_id,),
            ).fetchone()
            return str(row["text"]) if row else ""

    def save_cover_image(self, novel_id: int, cover_b64: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE novels SET cover_image = ?, updated_at = ? WHERE id = ?",
                (cover_b64, utc_now(), novel_id),
            )

    def save_extra_metadata(self, novel_id: int, series: str = "", genres: str = "", file_size: int = 0) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE novels SET series = ?, genres = ?, file_size = ?, updated_at = ? WHERE id = ?",
                (series, genres, file_size, utc_now(), novel_id),
            )

    def archive_novel(self, novel_id: int) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                "UPDATE novels SET archived = 1, completed_at = ?, updated_at = ? WHERE id = ?",
                (now, now, novel_id),
            )

    def delete_novel(self, novel_id: int) -> None:
        novel = self.get_novel(novel_id)
        with self.connect() as conn:
            conn.execute("DELETE FROM dictionary_lookups WHERE novel_id = ?", (novel_id,))
            conn.execute("DELETE FROM highlights WHERE novel_id = ?", (novel_id,))
            conn.execute("DELETE FROM bookmarks WHERE novel_id = ?", (novel_id,))
            conn.execute("DELETE FROM progress WHERE novel_id = ?", (novel_id,))
            conn.execute("DELETE FROM sections WHERE novel_id = ?", (novel_id,))
            conn.execute("DELETE FROM novels WHERE id = ?", (novel_id,))
        if novel:
            novel_dir = Path(str(novel["stored_path"])).parent.parent
            if novel_dir.exists() and novel_dir.parent == NOVELS_DIR:
                shutil.rmtree(novel_dir)

    def update_progress(self, novel_id: int, section_index: int) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO progress (novel_id, section_index, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(novel_id) DO UPDATE SET section_index = excluded.section_index, updated_at = excluded.updated_at
                """,
                (novel_id, section_index, now),
            )

    def add_bookmark(self, novel_id: int, section_index: int, label: str, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO bookmarks (novel_id, section_index, label, note, created_at) VALUES (?, ?, ?, ?, ?)",
                (novel_id, section_index, label, note, utc_now()),
            )

    def add_highlight(self, novel_id: int, section_index: int, quote: str, note: str = "", color: str = "gold") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO highlights (novel_id, section_index, quote, note, color, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (novel_id, section_index, quote, note, color, utc_now()),
            )

    def add_dictionary_lookup(self, novel_id: int | None, section_index: int, query: str, search_url: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO dictionary_lookups (novel_id, section_index, query, search_url, created_at) VALUES (?, ?, ?, ?, ?)",
                (novel_id, section_index, query, search_url, utc_now()),
            )

    def sidebar_items(self, novel_id: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if not novel_id:
            return [], [], []
        with self.connect() as conn:
            bookmarks = [dict(row) for row in conn.execute(
                "SELECT * FROM bookmarks WHERE novel_id = ? ORDER BY created_at DESC LIMIT 20", (novel_id,)
            )]
            highlights = [dict(row) for row in conn.execute(
                "SELECT * FROM highlights WHERE novel_id = ? ORDER BY created_at DESC LIMIT 20", (novel_id,)
            )]
            lookups = [dict(row) for row in conn.execute(
                "SELECT * FROM dictionary_lookups WHERE novel_id = ? ORDER BY created_at DESC LIMIT 20", (novel_id,)
            )]
            return bookmarks, highlights, lookups
