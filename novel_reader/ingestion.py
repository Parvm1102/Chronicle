from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from .config import SUPPORTED_FORMATS
from .parsers import extract_cover_info, parse_book
from .storage import LibraryStore

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Base ingestion pipeline; AI/RAG stages will slot in after parsing."""

    def __init__(self, store: LibraryStore) -> None:
        self.store = store

    def ingest(self, upload_path: str) -> int:
        path = Path(upload_path)
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format. Use: {', '.join(sorted(SUPPORTED_FORMATS))}")
        novel_id = self.store.create_novel_record(path)
        thread = threading.Thread(target=self._run_job, args=(novel_id,), daemon=True)
        thread.start()
        return novel_id

    def _run_job(self, novel_id: int) -> None:
        novel = self.store.get_novel(novel_id)
        if not novel:
            return
        try:
            self.store.set_status(novel_id, "running", "Parsing book text")
            source_path = Path(novel["stored_path"])
            parsed = parse_book(source_path)
            self.store.save_parsed_book(novel_id, parsed)

            # Extract cover image and extra metadata
            file_size = source_path.stat().st_size if source_path.exists() else 0
            try:
                cover_info = extract_cover_info(source_path)
                if cover_info.cover_b64:
                    self.store.save_cover_image(novel_id, cover_info.cover_b64)
                self.store.save_extra_metadata(
                    novel_id,
                    series=cover_info.series,
                    genres=cover_info.genres,
                    file_size=file_size,
                )
            except Exception:
                # Non-fatal: cover extraction failures should not break ingestion
                self.store.save_extra_metadata(novel_id, file_size=file_size)

            # ── Trigger novel parsing pipeline (if enabled) ───────────
            self._maybe_trigger_parsing(novel_id)

        except Exception as exc:
            self.store.set_status(novel_id, "error", str(exc))

    def _maybe_trigger_parsing(self, novel_id: int) -> None:
        """Optionally trigger the two-pass LLM parsing pipeline.

        Gated by ENABLE_NOVEL_PARSING env var.  Runs in a separate thread
        so it does not block the reader UI.
        """
        if os.environ.get("ENABLE_NOVEL_PARSING", "false").lower() != "true":
            return

        novel = self.store.get_novel(novel_id)
        if not novel:
            return

        # Gather sections for the parser
        sections = self.store.list_sections_full(novel_id)

        try:
            from novel_parser.pipeline import trigger_parsing_async

            trigger_parsing_async(
                novel_uuid=novel["uuid"],
                sections=sections,
                novel_title=novel["title"],
            )
            logger.info("Triggered novel parsing for '%s'", novel["title"])
        except ImportError:
            logger.debug("novel_parser package not available — skipping parsing")
        except Exception as exc:
            logger.warning("Failed to trigger novel parsing: %s", exc)
