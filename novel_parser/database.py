"""PostgreSQL connection management and schema initialisation.

Supports both local Docker PostgreSQL and Supabase (cloud).
Uses psycopg 3 with connection pooling.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Schema DDL
# ───────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- Series management
CREATE TABLE IF NOT EXISTS series (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Links to existing SQLite novel via UUID
CREATE TABLE IF NOT EXISTS novels_meta (
    id            SERIAL PRIMARY KEY,
    novel_uuid    TEXT NOT NULL UNIQUE,
    novel_title   TEXT NOT NULL,
    parse_status  TEXT NOT NULL DEFAULT 'pending',
    parse_message TEXT NOT NULL DEFAULT '',
    narrator_type TEXT NOT NULL DEFAULT 'unknown',
    narrator_character_id INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Series <-> Novel junction
CREATE TABLE IF NOT EXISTS series_novels (
    id              SERIAL PRIMARY KEY,
    series_id       INTEGER NOT NULL REFERENCES series(id) ON DELETE CASCADE,
    novel_meta_id   INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    book_order      INTEGER NOT NULL,
    UNIQUE(series_id, novel_meta_id),
    UNIQUE(series_id, book_order)
);

-- Voice actors with metadata for character matching
CREATE TABLE IF NOT EXISTS voice_actors (
    id                 SERIAL PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    gender             TEXT NOT NULL,
    age_range          TEXT NOT NULL,
    tone_tags          TEXT[] NOT NULL DEFAULT '{}',
    default_emotion    TEXT NOT NULL DEFAULT 'neutral',
    notes              TEXT NOT NULL DEFAULT '',
    protagonist_suited BOOLEAN NOT NULL DEFAULT FALSE,
    sample_dir         TEXT NOT NULL,
    emotions           JSONB NOT NULL DEFAULT '{}',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Characters extracted in Pass 1
CREATE TABLE IF NOT EXISTS characters (
    id                       SERIAL PRIMARY KEY,
    novel_meta_id            INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    series_id                INTEGER REFERENCES series(id),
    name                     TEXT NOT NULL,
    aliases                  TEXT[] NOT NULL DEFAULT '{}',
    gender                   TEXT NOT NULL DEFAULT 'unknown',
    age_range                TEXT NOT NULL DEFAULT 'unknown',
    role                     TEXT NOT NULL DEFAULT 'minor',
    description              TEXT NOT NULL DEFAULT '',
    voice_actor_id           INTEGER REFERENCES voice_actors(id),
    is_narrator              BOOLEAN NOT NULL DEFAULT FALSE,
    first_appearance_section INTEGER,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(novel_meta_id, name)
);

-- Dynamic character profiles (base state + temporal updates)
CREATE TABLE IF NOT EXISTS character_profiles (
    id              SERIAL PRIMARY KEY,
    character_id    INTEGER NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    novel_meta_id   INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    section_index   INTEGER NOT NULL,
    profile_type    TEXT NOT NULL DEFAULT 'update',
    emotional_state TEXT NOT NULL DEFAULT '',
    relationships   JSONB NOT NULL DEFAULT '{}',
    knowledge       TEXT[] NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT '',
    summary         TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(character_id, novel_meta_id, section_index, profile_type)
);

-- Dialogue entries from Pass 2 (core audiobook data)
CREATE TABLE IF NOT EXISTS dialogue_entries (
    id                    SERIAL PRIMARY KEY,
    novel_meta_id         INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    section_index         INTEGER NOT NULL,
    sequence_number       INTEGER NOT NULL,
    entry_type            TEXT NOT NULL DEFAULT 'dialogue',
    raw_text              TEXT NOT NULL,
    original_text         TEXT NOT NULL DEFAULT '',
    speaker_id            INTEGER REFERENCES characters(id),
    speaker_name          TEXT NOT NULL DEFAULT '',
    emotion               TEXT NOT NULL DEFAULT 'neutral',
    emotion_intensity     TEXT NOT NULL DEFAULT 'low',
    associated_characters INTEGER[] NOT NULL DEFAULT '{}',
    context_before        TEXT NOT NULL DEFAULT '',
    context_after         TEXT NOT NULL DEFAULT '',
    llm_confidence        REAL NOT NULL DEFAULT 0.0,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(novel_meta_id, section_index, sequence_number)
);

-- Novel events for future RAG
CREATE TABLE IF NOT EXISTS novel_events (
    id                  SERIAL PRIMARY KEY,
    novel_meta_id       INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    section_index       INTEGER NOT NULL,
    sequence_number     INTEGER NOT NULL,
    event_type          TEXT NOT NULL DEFAULT 'plot',
    summary             TEXT NOT NULL,
    characters_involved INTEGER[] NOT NULL DEFAULT '{}',
    speakers            INTEGER[] NOT NULL DEFAULT '{}',
    importance          TEXT NOT NULL DEFAULT 'minor',
    spoiler_level       INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(novel_meta_id, section_index, sequence_number)
);

-- Parse progress tracking (for resumability)
CREATE TABLE IF NOT EXISTS parse_progress (
    id              SERIAL PRIMARY KEY,
    novel_meta_id   INTEGER NOT NULL REFERENCES novels_meta(id) ON DELETE CASCADE,
    pass_number     INTEGER NOT NULL,
    current_section INTEGER NOT NULL DEFAULT 0,
    total_sections  INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    error_message   TEXT NOT NULL DEFAULT '',
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(novel_meta_id, pass_number)
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_characters_novel ON characters(novel_meta_id);
CREATE INDEX IF NOT EXISTS idx_characters_series ON characters(series_id);
CREATE INDEX IF NOT EXISTS idx_dialogue_novel_section ON dialogue_entries(novel_meta_id, section_index);
CREATE INDEX IF NOT EXISTS idx_events_novel_section ON novel_events(novel_meta_id, section_index);
CREATE INDEX IF NOT EXISTS idx_profiles_character ON character_profiles(character_id);
CREATE INDEX IF NOT EXISTS idx_series_novels_series ON series_novels(series_id);
"""


class DatabaseManager:
    """PostgreSQL connection pool and schema management.

    Usage::

        db = DatabaseManager()          # reads config from env
        db.init_schema()                # create tables if needed
        with db.connection() as conn:
            rows = conn.execute("SELECT ...").fetchall()
        db.close()
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        conninfo = self._settings.database_url
        kwargs: dict[str, Any] = {
            "conninfo": conninfo,
            "min_size": 1,
            "max_size": 5,
            "kwargs": {"row_factory": dict_row},
        }
        # SSL for Supabase
        ssl_mode = self._settings.postgres_ssl_mode
        if ssl_mode:
            kwargs["kwargs"]["sslmode"] = ssl_mode  # type: ignore[index]
        self._pool = ConnectionPool(**kwargs)
        logger.info("PostgreSQL pool created (host derived from DATABASE_URL)")

    # ── connections ────────────────────────────────────────────────────────

    @contextmanager
    def connection(self) -> Iterator[psycopg.Connection]:
        """Yield a connection from the pool, auto-commit on success."""
        with self._pool.connection() as conn:
            yield conn

    def close(self) -> None:
        """Shut down the pool."""
        self._pool.close()
        logger.info("PostgreSQL pool closed")

    # ── schema ─────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """Create all tables and indexes if they don't exist."""
        with self.connection() as conn:
            conn.execute(SCHEMA_SQL)
        logger.info("PostgreSQL schema initialised")

    # ── novels_meta CRUD ───────────────────────────────────────────────────

    def upsert_novel_meta(
        self, novel_uuid: str, novel_title: str
    ) -> int:
        """Create or fetch the novels_meta row, return its id."""
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO novels_meta (novel_uuid, novel_title)
                VALUES (%s, %s)
                ON CONFLICT (novel_uuid) DO UPDATE
                    SET novel_title = EXCLUDED.novel_title,
                        updated_at  = NOW()
                RETURNING id
                """,
                (novel_uuid, novel_title),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def set_parse_status(
        self, novel_meta_id: int, status: str, message: str = ""
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE novels_meta
                SET parse_status = %s, parse_message = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (status, message, novel_meta_id),
            )

    def set_narrator_info(
        self,
        novel_meta_id: int,
        narrator_type: str,
        narrator_character_id: Optional[int] = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE novels_meta
                SET narrator_type = %s, narrator_character_id = %s, updated_at = NOW()
                WHERE id = %s
                """,
                (narrator_type, narrator_character_id, novel_meta_id),
            )

    def get_novel_meta(self, novel_uuid: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM novels_meta WHERE novel_uuid = %s",
                (novel_uuid,),
            ).fetchone()

    # ── wipe for re-run ───────────────────────────────────────────────────

    def wipe_novel_parse_data(self, novel_meta_id: int) -> None:
        """Delete all parse data for a novel (cascade handles children).

        Preserves the novels_meta row itself and any series linkage.
        """
        with self.connection() as conn:
            for table in (
                "dialogue_entries",
                "novel_events",
                "character_profiles",
                "characters",
                "parse_progress",
            ):
                conn.execute(
                    f"DELETE FROM {table} WHERE novel_meta_id = %s",
                    (novel_meta_id,),
                )
            conn.execute(
                """
                UPDATE novels_meta
                SET parse_status = 'pending', parse_message = '',
                    narrator_type = 'unknown', narrator_character_id = NULL,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (novel_meta_id,),
            )
        logger.info("Wiped parse data for novel_meta_id=%s", novel_meta_id)

    # ── characters ─────────────────────────────────────────────────────────

    def insert_character(
        self,
        novel_meta_id: int,
        *,
        name: str,
        aliases: list[str] | None = None,
        gender: str = "unknown",
        age_range: str = "unknown",
        role: str = "minor",
        description: str = "",
        voice_actor_id: int | None = None,
        is_narrator: bool = False,
        first_appearance_section: int | None = None,
        series_id: int | None = None,
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO characters
                    (novel_meta_id, series_id, name, aliases, gender, age_range,
                     role, description, voice_actor_id, is_narrator,
                     first_appearance_section)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (novel_meta_id, name) DO UPDATE SET
                    aliases = EXCLUDED.aliases,
                    gender = EXCLUDED.gender,
                    age_range = EXCLUDED.age_range,
                    role = EXCLUDED.role,
                    description = EXCLUDED.description,
                    voice_actor_id = EXCLUDED.voice_actor_id,
                    is_narrator = EXCLUDED.is_narrator,
                    first_appearance_section = COALESCE(
                        characters.first_appearance_section,
                        EXCLUDED.first_appearance_section
                    ),
                    updated_at = NOW()
                RETURNING id
                """,
                (
                    novel_meta_id,
                    series_id,
                    name,
                    aliases or [],
                    gender,
                    age_range,
                    role,
                    description,
                    voice_actor_id,
                    is_narrator,
                    first_appearance_section,
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def get_characters(self, novel_meta_id: int) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM characters WHERE novel_meta_id = %s ORDER BY id",
                (novel_meta_id,),
            ).fetchall()

    def get_series_characters(self, series_id: int) -> list[dict[str, Any]]:
        """Get all characters across all books in a series."""
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM characters WHERE series_id = %s ORDER BY id",
                (series_id,),
            ).fetchall()

    def get_characters_up_to_book(
        self, series_id: int, max_book_order: int
    ) -> list[dict[str, Any]]:
        """Characters from books 1..max_book_order in a series (spoiler-safe)."""
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT c.*
                FROM characters c
                JOIN series_novels sn ON sn.novel_meta_id = c.novel_meta_id
                WHERE sn.series_id = %s AND sn.book_order <= %s
                ORDER BY c.id
                """,
                (series_id, max_book_order),
            ).fetchall()

    # ── character profiles ─────────────────────────────────────────────────

    def insert_character_profile(
        self,
        character_id: int,
        novel_meta_id: int,
        section_index: int,
        *,
        profile_type: str = "update",
        emotional_state: str = "",
        relationships: dict | None = None,
        knowledge: list[str] | None = None,
        status: str = "",
        summary: str = "",
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO character_profiles
                    (character_id, novel_meta_id, section_index, profile_type,
                     emotional_state, relationships, knowledge, status, summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (character_id, novel_meta_id, section_index, profile_type)
                DO UPDATE SET
                    emotional_state = EXCLUDED.emotional_state,
                    relationships   = EXCLUDED.relationships,
                    knowledge       = EXCLUDED.knowledge,
                    status          = EXCLUDED.status,
                    summary         = EXCLUDED.summary,
                    created_at      = NOW()
                RETURNING id
                """,
                (
                    character_id,
                    novel_meta_id,
                    section_index,
                    profile_type,
                    emotional_state,
                    psycopg.types.json.Jsonb(relationships or {}),
                    knowledge or [],
                    status,
                    summary,
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def get_latest_profiles(
        self,
        novel_meta_id: int,
        up_to_section: int | None = None,
    ) -> list[dict[str, Any]]:
        """Latest profile snapshot for each character, up to a section index."""
        section_filter = ""
        params: list[Any] = [novel_meta_id]
        if up_to_section is not None:
            section_filter = "AND cp.section_index <= %s"
            params.append(up_to_section)
        with self.connection() as conn:
            return conn.execute(
                f"""
                SELECT DISTINCT ON (cp.character_id)
                    cp.*, c.name AS character_name
                FROM character_profiles cp
                JOIN characters c ON c.id = cp.character_id
                WHERE cp.novel_meta_id = %s {section_filter}
                ORDER BY cp.character_id, cp.section_index DESC, cp.id DESC
                """,
                params,
            ).fetchall()

    def get_latest_profiles_for_series(
        self,
        series_id: int,
        max_book_order: int,
    ) -> list[dict[str, Any]]:
        """Latest profile for each character across books up to max_book_order."""
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT DISTINCT ON (cp.character_id)
                    cp.*, c.name AS character_name
                FROM character_profiles cp
                JOIN characters c ON c.id = cp.character_id
                JOIN series_novels sn ON sn.novel_meta_id = cp.novel_meta_id
                WHERE sn.series_id = %s AND sn.book_order <= %s
                ORDER BY cp.character_id, sn.book_order DESC,
                         cp.section_index DESC, cp.id DESC
                """,
                (series_id, max_book_order),
            ).fetchall()

    # ── dialogue entries ───────────────────────────────────────────────────

    def insert_dialogue_entry(
        self,
        novel_meta_id: int,
        section_index: int,
        sequence_number: int,
        *,
        entry_type: str = "dialogue",
        raw_text: str,
        original_text: str = "",
        speaker_id: int | None = None,
        speaker_name: str = "",
        emotion: str = "neutral",
        emotion_intensity: str = "low",
        associated_characters: list[int] | None = None,
        context_before: str = "",
        context_after: str = "",
        llm_confidence: float = 0.0,
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO dialogue_entries
                    (novel_meta_id, section_index, sequence_number, entry_type,
                     raw_text, original_text, speaker_id, speaker_name,
                     emotion, emotion_intensity, associated_characters,
                     context_before, context_after, llm_confidence)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (novel_meta_id, section_index, sequence_number)
                DO UPDATE SET
                    entry_type            = EXCLUDED.entry_type,
                    raw_text              = EXCLUDED.raw_text,
                    original_text         = EXCLUDED.original_text,
                    speaker_id            = EXCLUDED.speaker_id,
                    speaker_name          = EXCLUDED.speaker_name,
                    emotion               = EXCLUDED.emotion,
                    emotion_intensity     = EXCLUDED.emotion_intensity,
                    associated_characters = EXCLUDED.associated_characters,
                    context_before        = EXCLUDED.context_before,
                    context_after         = EXCLUDED.context_after,
                    llm_confidence        = EXCLUDED.llm_confidence
                RETURNING id
                """,
                (
                    novel_meta_id, section_index, sequence_number, entry_type,
                    raw_text, original_text, speaker_id, speaker_name,
                    emotion, emotion_intensity, associated_characters or [],
                    context_before, context_after, llm_confidence,
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def get_dialogue_entries(
        self, novel_meta_id: int, section_index: int
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM dialogue_entries
                WHERE novel_meta_id = %s AND section_index = %s
                ORDER BY sequence_number
                """,
                (novel_meta_id, section_index),
            ).fetchall()

    def get_recent_dialogue_entries(
        self, novel_meta_id: int, section_index: int, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Last N entries up to (but not including) the given section, for history buffer."""
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM dialogue_entries
                WHERE novel_meta_id = %s AND section_index < %s
                ORDER BY section_index DESC, sequence_number DESC
                LIMIT %s
                """,
                (novel_meta_id, section_index, limit),
            ).fetchall()[::-1]  # reverse to chronological order

    # ── novel events ───────────────────────────────────────────────────────

    def insert_event(
        self,
        novel_meta_id: int,
        section_index: int,
        sequence_number: int,
        *,
        event_type: str = "plot",
        summary: str,
        characters_involved: list[int] | None = None,
        speakers: list[int] | None = None,
        importance: str = "minor",
        spoiler_level: int = 0,
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO novel_events
                    (novel_meta_id, section_index, sequence_number, event_type,
                     summary, characters_involved, speakers, importance, spoiler_level)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (novel_meta_id, section_index, sequence_number)
                DO UPDATE SET
                    event_type          = EXCLUDED.event_type,
                    summary             = EXCLUDED.summary,
                    characters_involved = EXCLUDED.characters_involved,
                    speakers            = EXCLUDED.speakers,
                    importance          = EXCLUDED.importance,
                    spoiler_level       = EXCLUDED.spoiler_level
                RETURNING id
                """,
                (
                    novel_meta_id, section_index, sequence_number, event_type,
                    summary, characters_involved or [], speakers or [],
                    importance, spoiler_level,
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    # ── voice actors ───────────────────────────────────────────────────────

    def upsert_voice_actor(
        self,
        name: str,
        *,
        gender: str,
        age_range: str,
        tone_tags: list[str] | None = None,
        default_emotion: str = "neutral",
        notes: str = "",
        protagonist_suited: bool = False,
        sample_dir: str = "",
        emotions: dict | None = None,
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO voice_actors
                    (name, gender, age_range, tone_tags, default_emotion,
                     notes, protagonist_suited, sample_dir, emotions)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (name) DO UPDATE SET
                    gender             = EXCLUDED.gender,
                    age_range          = EXCLUDED.age_range,
                    tone_tags          = EXCLUDED.tone_tags,
                    default_emotion    = EXCLUDED.default_emotion,
                    notes              = EXCLUDED.notes,
                    protagonist_suited = EXCLUDED.protagonist_suited,
                    sample_dir         = EXCLUDED.sample_dir,
                    emotions           = EXCLUDED.emotions
                RETURNING id
                """,
                (
                    name, gender, age_range, tone_tags or [],
                    default_emotion, notes, protagonist_suited, sample_dir,
                    psycopg.types.json.Jsonb(emotions or {}),
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def get_all_voice_actors(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM voice_actors ORDER BY id"
            ).fetchall()

    # ── parse progress ─────────────────────────────────────────────────────

    def upsert_parse_progress(
        self,
        novel_meta_id: int,
        pass_number: int,
        *,
        current_section: int = 0,
        total_sections: int = 0,
        status: str = "pending",
        error_message: str = "",
    ) -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO parse_progress
                    (novel_meta_id, pass_number, current_section, total_sections,
                     status, error_message,
                     started_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                ON CONFLICT (novel_meta_id, pass_number)
                DO UPDATE SET
                    current_section = EXCLUDED.current_section,
                    total_sections  = EXCLUDED.total_sections,
                    status          = EXCLUDED.status,
                    error_message   = EXCLUDED.error_message,
                    updated_at      = NOW()
                RETURNING id
                """,
                (
                    novel_meta_id, pass_number, current_section,
                    total_sections, status, error_message,
                ),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def mark_progress_complete(
        self, novel_meta_id: int, pass_number: int
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE parse_progress
                SET status = 'complete', completed_at = NOW(), updated_at = NOW()
                WHERE novel_meta_id = %s AND pass_number = %s
                """,
                (novel_meta_id, pass_number),
            )

    def get_parse_progress(
        self, novel_meta_id: int, pass_number: int
    ) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM parse_progress
                WHERE novel_meta_id = %s AND pass_number = %s
                """,
                (novel_meta_id, pass_number),
            ).fetchone()

    # ── series CRUD ────────────────────────────────────────────────────────

    def create_series(self, name: str, description: str = "") -> int:
        with self.connection() as conn:
            row = conn.execute(
                """
                INSERT INTO series (name, description)
                VALUES (%s, %s) RETURNING id
                """,
                (name, description),
            ).fetchone()
            return int(row["id"])  # type: ignore[index]

    def add_novel_to_series(
        self, series_id: int, novel_meta_id: int, book_order: int
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO series_novels (series_id, novel_meta_id, book_order)
                VALUES (%s, %s, %s)
                ON CONFLICT (series_id, novel_meta_id)
                DO UPDATE SET book_order = EXCLUDED.book_order
                """,
                (series_id, novel_meta_id, book_order),
            )

    def get_series_for_novel(self, novel_meta_id: int) -> Optional[dict[str, Any]]:
        """Return the series info + book_order for a novel, or None."""
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT s.*, sn.book_order
                FROM series s
                JOIN series_novels sn ON sn.series_id = s.id
                WHERE sn.novel_meta_id = %s
                """,
                (novel_meta_id,),
            ).fetchone()

    def get_series_novels(self, series_id: int) -> list[dict[str, Any]]:
        """All novels in a series, ordered by book_order."""
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT nm.*, sn.book_order
                FROM novels_meta nm
                JOIN series_novels sn ON sn.novel_meta_id = nm.id
                WHERE sn.series_id = %s
                ORDER BY sn.book_order
                """,
                (series_id,),
            ).fetchall()
