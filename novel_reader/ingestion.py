from __future__ import annotations

import threading
from pathlib import Path

from .config import SUPPORTED_FORMATS
from .parsers import extract_cover_info, parse_book
from .storage import LibraryStore


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
        except Exception as exc:
            self.store.set_status(novel_id, "error", str(exc))
