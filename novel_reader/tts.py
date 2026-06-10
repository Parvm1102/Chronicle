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
import re
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from .config import (
    TTS_DEFAULT_NARRATOR_ACTOR,
    TTS_LOOKAHEAD,
    TTS_MAX_CHARS,
    TTS_SERVICE_URL,
)
from .storage import LibraryStore

logger = logging.getLogger(__name__)

# Each speakable unit's ``seq`` packs the source entry's ``sequence_number`` and
# the chunk index within that entry: ``seq = sequence_number * _SEQ_STRIDE + i``.
# This keeps unit seqs globally unique *and* monotonically ordered (so prefetch
# windows and the audio route still match by seq), while letting the client map
# a unit back to its source entry via ``seq // _SEQ_STRIDE``. The stride caps the
# number of chunks one entry may produce; entries never approach this many.
_SEQ_STRIDE = 1000

# Sentence boundary: end punctuation (incl. closing quotes/brackets) + whitespace.
_SENTENCE_RE = re.compile(r"(?<=[.!?…])[\"'”’)\]]*\s+")
# Clause/secondary break points, used only when a single sentence is too long.
_CLAUSE_RE = re.compile(r"[,;:—–-]")


def _split_oversized(text: str, max_len: int) -> list[str]:
    """Split one over-long sentence into ``<= max_len`` pieces.

    Breaks as late as possible while staying under the limit: prefers the latest
    clause punctuation (``, ; : — – -``) at/before the boundary, then the latest
    word boundary. Only hard-cuts when a single word itself exceeds the limit, and
    even then never mid-word unless unavoidable.
    """
    pieces: list[str] = []
    remaining = text.strip()
    while len(remaining) > max_len:
        window = remaining[:max_len]
        cut = None
        # Latest clause break within the limit.
        for m in _CLAUSE_RE.finditer(window):
            cut = m.end()
        if not cut:
            # Latest word boundary within the limit.
            ws = window.rfind(" ")
            cut = ws if ws > 0 else max_len  # single word > limit → hard cut
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return [p for p in pieces if p]


def _split_text_for_tts(text: str, max_len: int = TTS_MAX_CHARS) -> list[str]:
    """Split ``text`` into speakable chunks each ``<= max_len`` characters.

    Greedy packing: whole sentences are accumulated into a chunk until the next
    one would exceed ``max_len``, minimizing the number of breaks. A single
    sentence longer than ``max_len`` is split via :func:`_split_oversized`.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_len:
        return [text]

    sentences = [s for s in (p.strip() for p in _SENTENCE_RE.split(text)) if s] or [text]

    chunks: list[str] = []
    buf = ""
    for sentence in sentences:
        if len(sentence) > max_len:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_oversized(sentence, max_len))
            continue
        candidate = f"{buf} {sentence}" if buf else sentence
        if len(candidate) <= max_len:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = sentence
    if buf:
        chunks.append(buf)
    return chunks

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
        # A couple of background workers so prefetch runs parallel to the UI and
        # to playback. (The TTS engine itself serialises on the GPU, so a wide
        # pool buys nothing — the win is overlapping prefetch with playback.)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-prefetch")
        self._lock = threading.Lock()
        self._lookahead = max(1, TTS_LOOKAHEAD)

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
        novel = self._store.get_novel(novel_id)
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
            entry_seq = int(entry["sequence_number"])
            # The untagged source text drives client-side highlight matching; the
            # tagged ``raw_text`` is what we actually synthesize. Chunking applies
            # to the synthesized text so no single TTS request exceeds the limit.
            original_text = (entry.get("original_text") or text).strip()
            chunks = _split_text_for_tts(text) or [text]
            for chunk_idx, chunk in enumerate(chunks):
                seq = entry_seq * _SEQ_STRIDE + chunk_idx
                units.append(
                    {
                        "seq": seq,
                        "entry_seq": entry_seq,
                        "chunk_index": chunk_idx,
                        "chunk_count": len(chunks),
                        "entry_type": entry.get("entry_type") or "narration",
                        "speaker": entry.get("speaker_name") or "Narrator",
                        "text": chunk,
                        "original_text": original_text,
                        "voice_ref": voice_ref,
                        "audio_url": f"/tts/audio/{novel_id}/{section_index}/{seq}",
                    }
                )

        return {"status": "ready", "section": section_index, "units": units}

    # ── prefetch + audio ────────────────────────────────────────────────────

    def prefetch_window(
        self,
        novel_id: int,
        section_index: int,
        start_seq: Optional[int] = None,
        lookahead: Optional[int] = None,
    ) -> None:
        """Warm a *bounded* window of upcoming units in the background.

        Only ``lookahead`` units at/after ``start_seq`` are synthesized — a
        sliding window that tracks the playback head rather than the whole
        chapter. When the window runs past the end of the section it spills into
        the next section's first units so chapter transitions stay seamless.

        Called once when the user presses play (``start_seq=None`` → from the
        top) and again each time the player advances a line (``start_seq`` = the
        line now playing), which slides the window forward.
        """
        n = max(1, lookahead or self._lookahead)
        self._pool.submit(self._prefetch_window, novel_id, section_index, start_seq, n)

    def _prefetch_window(
        self, novel_id: int, section_index: int, start_seq: Optional[int], lookahead: int
    ) -> None:
        try:
            playlist = self.build_playlist(novel_id, section_index)
            if playlist.get("status") != "ready":
                return
            units = playlist["units"]
            if start_seq is None:
                start_idx = 0
            else:
                start_idx = next(
                    (i for i, u in enumerate(units) if u["seq"] >= start_seq), len(units)
                )
            window = units[start_idx : start_idx + lookahead]
            self._synthesize_units(novel_id, section_index, window)

            # Spill remaining budget into the next section near a chapter boundary.
            remaining = lookahead - len(window)
            if remaining > 0:
                nxt = self.build_playlist(novel_id, section_index + 1)
                if nxt.get("status") == "ready":
                    self._synthesize_units(novel_id, section_index + 1, nxt["units"][:remaining])
        except Exception as exc:
            logger.debug("Prefetch window (sec=%s, seq=%s) failed: %s", section_index, start_seq, exc)

    def _synthesize_units(self, novel_id: int, section_index: int, units: list[dict]) -> None:
        """Synthesize a list of units, skipping ones already cached/in-flight."""
        for unit in units:
            key = (novel_id, section_index, unit["seq"])
            with self._lock:
                if key in self._prefetched:
                    continue
                self._prefetched.add(key)
            try:
                self._client.synthesize(unit["text"], unit["voice_ref"])
            except Exception as exc:
                logger.debug("Prefetch synth failed (seq=%s): %s", unit["seq"], exc)
                with self._lock:
                    self._prefetched.discard(key)  # allow a later retry

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
