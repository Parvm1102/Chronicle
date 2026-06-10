"""Top-level orchestrator — runs the full two-pass parsing pipeline.

Ties together Pass 1 (character extraction), Pass 2 (dialogue analysis),
voice actor assignment, and series-aware context loading.

Can be called from the ingestion hook or standalone.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from .config import Settings, get_settings
from .database import DatabaseManager
from .llm_client import LLMClient
from .pass1_characters import CharacterExtractor
from .pass2_dialogue import DialogueAnalyzer
from .series_manager import SeriesManager
from .voice_actors import VoiceActorAssigner, VoiceActorManager

logger = logging.getLogger(__name__)


class NovelParsingPipeline:
    """Full two-pass novel parsing pipeline."""

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMClient,
        settings: Optional[Settings] = None,
    ) -> None:
        self._db = db
        self._llm = llm
        self._settings = settings or get_settings()
        self._series_mgr = SeriesManager(db)
        self._voice_mgr = VoiceActorManager(db, self._settings)
        self._voice_assigner = VoiceActorAssigner(db)
        self._char_extractor = CharacterExtractor(db, llm)
        self._dialogue_analyzer = DialogueAnalyzer(
            db, llm,
            history_size=self._settings.parse_dialogue_history_size,
            concurrency=self._settings.parse_concurrency,
        )

    def run(
        self,
        novel_uuid: str,
        sections: list[dict[str, Any]],
        novel_title: str,
        *,
        wipe_existing: bool = True,
    ) -> None:
        """Run the full pipeline (Pass 1 + Pass 2).

        Args:
            novel_uuid: The novel's UUID from the SQLite store.
            sections: Section dicts from SQLite (title, text, section_index).
            novel_title: Human-readable title.
            wipe_existing: If True, delete any existing parse data first (re-run).
        """
        logger.info("Pipeline: starting for novel '%s' (uuid=%s)", novel_title, novel_uuid)

        # Ensure voice actors are seeded
        self._voice_mgr.seed()

        # Create or get novels_meta row
        novel_meta_id = self._db.upsert_novel_meta(novel_uuid, novel_title)

        # Wipe on re-run
        if wipe_existing:
            self._db.wipe_novel_parse_data(novel_meta_id)

        # Check for series context
        series_info = self._series_mgr.get_series_for_novel(novel_meta_id)
        series_id: int | None = series_info["id"] if series_info else None
        book_order: int = series_info["book_order"] if series_info else 1

        prior_characters: list[dict[str, Any]] = []
        if series_id and book_order > 1:
            prior_characters = self._series_mgr.get_prior_characters(
                series_id, book_order
            )
            logger.info(
                "Pipeline: loaded %d characters from prior books in series",
                len(prior_characters),
            )

        try:
            self._db.set_parse_status(novel_meta_id, "pass1_running", "Extracting characters...")

            # ── Pass 1 ────────────────────────────────────────────────────
            characters = self._char_extractor.run(
                novel_meta_id, sections, novel_title,
                prior_characters=prior_characters,
                series_id=series_id,
            )
            logger.info("Pipeline: Pass 1 complete — %d characters", len(characters))

            # ── Voice actor assignment ────────────────────────────────────
            assignments = self._voice_assigner.assign_all(novel_meta_id, characters)
            logger.info("Pipeline: assigned %d voice actors", len(assignments))

            # Refresh characters with voice assignments
            characters = self._db.get_characters(novel_meta_id)

            self._db.set_parse_status(novel_meta_id, "pass2_running", "Analysing dialogue...")

            # ── Pass 2 ────────────────────────────────────────────────────
            self._dialogue_analyzer.run(novel_meta_id, sections, characters)
            logger.info("Pipeline: Pass 2 complete")

            # ── Pass 3 — vector indexing (optional) ───────────────────────
            self._run_rag_indexing(novel_meta_id, novel_uuid, sections)

            self._db.set_parse_status(novel_meta_id, "complete", "Parsing complete")
            logger.info("Pipeline: finished for novel '%s'", novel_title)

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            self._db.set_parse_status(
                novel_meta_id, "error", f"Pipeline failed: {exc}"
            )
            raise

    def _run_rag_indexing(
        self,
        novel_meta_id: int,
        novel_uuid: str,
        sections: list[dict[str, Any]],
    ) -> None:
        """Pass 3 — index the novel into Qdrant (best-effort, never fatal).

        Gated by ENABLE_RAG_INDEXING. Resolves series context so chunks carry a
        stable ``series_key`` (the first book's uuid) and ``novel_number``.
        A failure here must not fail the parse — it is logged and swallowed.
        """
        try:
            from novel_rag.config import get_rag_settings
            from novel_rag.indexer import RagIndexer
        except ImportError:
            logger.debug("novel_rag not installed — skipping Pass 3")
            return

        rag_settings = get_rag_settings()
        if not rag_settings.enable_rag_indexing:
            return

        # Series context: standalone defaults to the novel acting as its own series.
        series_key = novel_uuid
        series_name = ""
        novel_number = 1
        series_info = self._series_mgr.get_series_for_novel(novel_meta_id)
        if series_info:
            series_name = series_info.get("name", "")
            novel_number = int(series_info.get("book_order", 1))
            books = self._db.get_series_novels(series_info["id"])
            if books:
                series_key = books[0].get("novel_uuid", novel_uuid)

        try:
            self._db.set_parse_status(novel_meta_id, "pass2_running", "Building vector index...")
            indexer = RagIndexer(rag_settings)
            count = indexer.run(
                novel_uuid,
                sections,
                series_key=series_key,
                series_name=series_name,
                novel_number=novel_number,
                db=self._db,
                novel_meta_id=novel_meta_id,
            )
            logger.info("Pipeline: Pass 3 indexed %d chunks for %s", count, novel_uuid)
        except Exception as exc:
            logger.warning("Pipeline: Pass 3 (RAG indexing) failed for %s: %s", novel_uuid, exc)

    def resume(
        self,
        novel_uuid: str,
        sections: list[dict[str, Any]],
        novel_title: str,
    ) -> None:
        """Resume a previously interrupted parse.

        Checks parse_progress and pass1_extractions to determine where to restart.
        Pass 1: Uses incremental extractions saved per-chapter to avoid re-running LLM calls.
        Pass 2: Uses section-level skipping (DialogueAnalyzer handles this internally).
        """
        # Ensure voice actors are seeded
        self._voice_mgr.seed()

        meta = self._db.get_novel_meta(novel_uuid)
        if not meta:
            logger.info("Resume: no existing parse — running from scratch")
            self.run(novel_uuid, sections, novel_title, wipe_existing=True)
            return

        novel_meta_id = meta["id"]
        status = meta["parse_status"]

        if status == "complete":
            logger.info("Resume: already complete — nothing to do")
            return

        if status in ("pending", "error"):
            logger.info("Resume: status=%s — running from scratch", status)
            self.run(novel_uuid, sections, novel_title, wipe_existing=True)
            return

        # Check which pass needs resuming
        p1_progress = self._db.get_parse_progress(novel_meta_id, 1)

        if status == "pass1_running" or (p1_progress and p1_progress["status"] != "complete"):
            # Check if we have saved intermediate results
            saved = self._db.get_pass1_extractions(novel_meta_id)
            if saved:
                logger.info(
                    "Resume: Pass 1 interrupted with %d saved extractions — resuming",
                    len(saved),
                )
            else:
                logger.info("Resume: Pass 1 interrupted with no saved data — restarting")
                self._db.wipe_novel_parse_data(novel_meta_id)

            # Check for series context
            series_info = self._series_mgr.get_series_for_novel(novel_meta_id)
            series_id: int | None = series_info["id"] if series_info else None
            book_order: int = series_info["book_order"] if series_info else 1

            prior_characters: list[dict[str, Any]] = []
            if series_id and book_order > 1:
                prior_characters = self._series_mgr.get_prior_characters(
                    series_id, book_order
                )

            try:
                self._db.set_parse_status(novel_meta_id, "pass1_running", "Resuming character extraction...")

                # CharacterExtractor.run() handles resume internally via pass1_extractions
                characters = self._char_extractor.run(
                    novel_meta_id, sections, novel_title,
                    prior_characters=prior_characters,
                    series_id=series_id,
                )
                logger.info("Resume: Pass 1 complete — %d characters", len(characters))

                # Voice actor assignment
                assignments = self._voice_assigner.assign_all(novel_meta_id, characters)
                logger.info("Resume: assigned %d voice actors", len(assignments))

                characters = self._db.get_characters(novel_meta_id)

                self._db.set_parse_status(novel_meta_id, "pass2_running", "Analysing dialogue...")

                # Pass 2 — DialogueAnalyzer.run() handles section-level resume internally
                self._dialogue_analyzer.run(novel_meta_id, sections, characters)
                logger.info("Resume: Pass 2 complete")

                self._run_rag_indexing(novel_meta_id, novel_uuid, sections)

                self._db.set_parse_status(novel_meta_id, "complete", "Parsing complete")

            except Exception as exc:
                logger.error("Resume pipeline error: %s", exc, exc_info=True)
                self._db.set_parse_status(novel_meta_id, "error", f"Resume failed: {exc}")
                raise
            return

        # Pass 1 complete, Pass 2 needs resuming
        logger.info("Resume: Pass 1 complete, resuming Pass 2")
        characters = self._db.get_characters(novel_meta_id)
        self._db.set_parse_status(novel_meta_id, "pass2_running", "Resuming dialogue analysis...")
        try:
            # DialogueAnalyzer.run() handles section-level resume internally
            self._dialogue_analyzer.run(novel_meta_id, sections, characters)
            self._run_rag_indexing(novel_meta_id, novel_uuid, sections)
            self._db.set_parse_status(novel_meta_id, "complete", "Parsing complete")
        except Exception as exc:
            self._db.set_parse_status(novel_meta_id, "error", f"Resume failed: {exc}")
            raise


# ───────────────────────────────────────────────────────────────────────────
# Async trigger (for use from ingestion pipeline)
# ───────────────────────────────────────────────────────────────────────────

def trigger_parsing_async(
    novel_uuid: str,
    sections: list[dict[str, Any]],
    novel_title: str,
) -> None:
    """Fire-and-forget trigger for the parsing pipeline.

    Uses resume() for crash-safety — if a prior parse was interrupted,
    it picks up where it left off instead of restarting from scratch.
    """
    def _run() -> None:
        try:
            settings = get_settings()
            db = DatabaseManager(settings)
            db.init_schema()
            llm = LLMClient(settings)
            pipeline = NovelParsingPipeline(db, llm, settings)
            pipeline.resume(novel_uuid, sections, novel_title)
        except Exception as exc:
            logger.error("Async parsing failed: %s", exc, exc_info=True)
        finally:
            try:
                db.close()
            except Exception:
                pass

    thread = threading.Thread(target=_run, daemon=True, name=f"parse-{novel_uuid[:8]}")
    thread.start()
    logger.info("Triggered async parsing for novel %s", novel_uuid)
