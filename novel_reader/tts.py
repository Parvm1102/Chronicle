"""TTS orchestration for the reader (runs inside the main, lightweight app).

This module bridges three things:

* the reader's SQLite library (sections, reading progress, novel ``uuid``),
* the parser's PostgreSQL data (``dialogue_entries`` with speaker / emotion /
  intensity per unit, ``characters`` → ``voice_actors``),
* the standalone Chatterbox-Turbo HTTP service (``tts_service/``).

It turns a chapter into an ordered *playlist* of speakable units (each with the
resolved voice sample + the tag-annotated text), pre-warms the TTS service cache
for upcoming units, and proxies audio bytes to the browser.

No heavy ML deps are imported here — only an HTTP client and the existing parser
DB layer.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from .config import TTS_DEFAULT_NARRATOR_ACTOR, TTS_SERVICE_URL
from .storage import LibraryStore

logger = logging.getLogger(__name__)

# Default Turbo sampling params (kept here so the audio + prefetch paths agree,
# which keeps the service-side cache keys identical).
_SYNTH_PARAMS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 1000,
    "repetition_penalty": 1.2,
}


class TTSClient:
    """Minimal HTTP client for the TTS microservice (stdlib only)."""

    def __init__(self, base_url: str = TTS_SERVICE_URL, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/health", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def synthesize(self, text: str, voice_ref: str) -> bytes:
        payload = json.dumps({"text": text, "voice_ref": voice_ref, **_SYNTH_PARAMS}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            f"{self.base_url}/synthesize",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read()


class TTSOrchestrator:
    """Builds chapter playlists, prefetches audio, and serves wav bytes."""

    def __init__(self, store: Optional[LibraryStore] = None) -> None:
        self._store = store or LibraryStore()
        self._client = TTSClient()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-prefetch")
        self._lock = threading.Lock()

        # Lazily-initialised parser-side helpers (PostgreSQL + voice samples).
        self._db: Any = None
        self._voice: Any = None
        self._voice_dir: Optional[Path] = None
        self._parser_ready: Optional[bool] = None

        # In-flight / submitted prefetch keys, to avoid duplicate work.
        self._prefetched: set[tuple[int, int, int]] = set()

        # Short-lived playlist cache so the audio route doesn't rebuild from the
        # DB for every unit. Keyed by (novel_id, section) → (timestamp, result).
        self._playlist_cache: dict[tuple[int, int], tuple[float, dict]] = {}
        self._playlist_ttl = 120.0

    # ── parser-side lazy wiring ─────────────────────────────────────────────

    def _ensure_parser(self) -> bool:
        """Connect to PostgreSQL + voice manager once. Returns availability."""
        if self._parser_ready is not None:
            return self._parser_ready
        with self._lock:
            if self._parser_ready is not None:
                return self._parser_ready
            try:
                from novel_parser.config import get_settings
                from novel_parser.database import DatabaseManager
                from novel_parser.voice_actors import VoiceActorManager

                settings = get_settings()
                self._db = DatabaseManager(settings)
                self._voice = VoiceActorManager(self._db, settings)
                self._voice_dir = Path(settings.voice_samples_dir).resolve()
                self._parser_ready = True
            except Exception as exc:  # postgres down, package missing, etc.
                logger.warning("TTS parser backend unavailable: %s", exc)
                self._parser_ready = False
            return self._parser_ready

    # ── voice resolution ────────────────────────────────────────────────────

    def _voice_ref(self, actor_name: str, emotion: str, intensity: str) -> Optional[str]:
        """Return a voice sample path relative to the voice samples dir."""
        path = self._voice.get_sample_path(actor_name, emotion, intensity)
        if path is None and actor_name != TTS_DEFAULT_NARRATOR_ACTOR:
            path = self._voice.get_sample_path(TTS_DEFAULT_NARRATOR_ACTOR, "neutral", "low")
        if path is None:
            return None
        try:
            return Path(path).resolve().relative_to(self._voice_dir).as_posix()
        except ValueError:
            # Sample lives outside the samples dir — fall back to "<actor>/<file>".
            return f"{Path(path).parent.name}/{Path(path).name}"

    # ── playlist ────────────────────────────────────────────────────────────

    def build_playlist(self, novel_id: int, section_index: int) -> dict:
        """Build an ordered, speakable playlist for one chapter.

        Returns a dict with ``status`` in {"ready", "in_progress", "unavailable"}.
        Only "ready" carries ``units``.
        """
        import time

        cache_key = (novel_id, section_index)
        cached = self._playlist_cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < self._playlist_ttl:
            return cached[1]
        result = self._build_playlist(novel_id, section_index)
        # Only cache a successful build; transient states should be re-checked.
        if result.get("status") == "ready":
            self._playlist_cache[cache_key] = (time.monotonic(), result)
        return result

    def _build_playlist(self, novel_id: int, section_index: int) -> dict:
        if not novel:
            return {"status": "unavailable", "message": "Book not found."}

        if not self._ensure_parser():
            return {
                "status": "unavailable",
                "message": "TTS backend is offline.",
            }

        uuid = novel.get("uuid")
        try:
            meta = self._db.get_novel_meta(uuid) if uuid else None
        except Exception as exc:
            logger.warning("get_novel_meta failed: %s", exc)
            return {"status": "unavailable", "message": "TTS backend is offline."}

        if not meta:
            return {"status": "in_progress", "message": "Parsing in progress…"}

        meta_id = int(meta["id"])
        try:
            entries = self._db.get_dialogue_entries(meta_id, section_index)
        except Exception as exc:
            logger.warning("get_dialogue_entries failed: %s", exc)
            return {"status": "unavailable", "message": "TTS backend is offline."}

        if not entries:
            progress = self._db.get_parse_progress(meta_id, 2)
            if progress and progress.get("status") == "complete":
                return {"status": "ready", "section": section_index, "units": []}
            return {"status": "in_progress", "message": "Parsing in progress…"}

        # Lookup tables for voice resolution.
        char_map = {c["id"]: c for c in self._db.get_characters(meta_id)}
        actor_map = {a["id"]: a["name"] for a in self._db.get_all_voice_actors()}
        narrator_char_id = meta.get("narrator_character_id")

        def actor_for(entry: dict) -> str:
            speaker_id = entry.get("speaker_id")
            char = char_map.get(speaker_id) if speaker_id else None
            if char and char.get("voice_actor_id") in actor_map:
                return actor_map[char["voice_actor_id"]]
            # Narration / unattributed → narrator character's actor, else default.
            if narrator_char_id and narrator_char_id in char_map:
                ncid = char_map[narrator_char_id].get("voice_actor_id")
                if ncid in actor_map:
                    return actor_map[ncid]
            return TTS_DEFAULT_NARRATOR_ACTOR

        units: list[dict] = []
        for entry in entries:
            text = (entry.get("raw_text") or entry.get("original_text") or "").strip()
            if not text:
                continue
            emotion = entry.get("emotion") or "neutral"
            intensity = entry.get("emotion_intensity") or "low"
            voice_ref = self._voice_ref(actor_for(entry), emotion, intensity)
            if not voice_ref:
                logger.debug("No voice sample for entry seq=%s — skipping", entry.get("sequence_number"))
                continue
            seq = int(entry["sequence_number"])
            units.append(
                {
                    "seq": seq,
                    "speaker": entry.get("speaker_name") or "Narrator",
                    "text": text,
                    "voice_ref": voice_ref,
                    "audio_url": f"/tts/audio/{novel_id}/{section_index}/{seq}",
                }
            )

        return {"status": "ready", "section": section_index, "units": units}

    # ── prefetch + audio ────────────────────────────────────────────────────

    def prefetch(self, novel_id: int, section_index: int) -> None:
        """Warm the TTS cache for this section (and the next) in the background."""
        for sec in (section_index, section_index + 1):
            self._pool.submit(self._prefetch_section, novel_id, sec)

    def _prefetch_section(self, novel_id: int, section_index: int) -> None:
        try:
            playlist = self.build_playlist(novel_id, section_index)
            if playlist.get("status") != "ready":
                return
            for unit in playlist["units"]:
                key = (novel_id, section_index, unit["seq"])
                with self._lock:
                    if key in self._prefetched:
                        continue
                    self._prefetched.add(key)
                try:
                    self._client.synthesize(unit["text"], unit["voice_ref"])
                except Exception as exc:
                    logger.debug("Prefetch failed (seq=%s): %s", unit["seq"], exc)
                    with self._lock:
                        self._prefetched.discard(key)
        except Exception as exc:
            logger.debug("Prefetch section %s failed: %s", section_index, exc)

    def synthesize_unit(self, novel_id: int, section_index: int, seq: int) -> Optional[bytes]:
        """Return wav bytes for a single unit, generating on demand if needed."""
        playlist = self.build_playlist(novel_id, section_index)
        if playlist.get("status") != "ready":
            return None
        unit = next((u for u in playlist["units"] if u["seq"] == seq), None)
        if not unit:
            return None
        return self._client.synthesize(unit["text"], unit["voice_ref"])

    def save_progress(self, novel_id: int, section_index: int) -> None:
        try:
            self._store.update_progress(novel_id, section_index)
        except Exception as exc:
            logger.debug("save_progress failed: %s", exc)
