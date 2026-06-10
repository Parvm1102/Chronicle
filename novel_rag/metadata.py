"""Per-chunk metadata enrichment from the parser's PostgreSQL store.

Attributes each chunk with the speakers and characters present in it and the
plot events it covers, by exploiting the fact that ``dialogue_entries`` and
``novel_events`` share the same ``(section_index, sequence_number)`` coordinate
space produced by Pass 2.

Pipeline per chapter:
1. Locate each dialogue entry's source text inside the chapter to get a char
   position, then map that position to the chunk that contains it.
2. From the entries inside a chunk, collect ``speakers`` and
   ``associated_characters``.
3. Bucket each event into the chunk holding the dialogue entry with the largest
   ``sequence_number <= event.sequence_number`` (events are appended right after
   their chunk's dialogue), recording ``event_ids`` and ``max_spoiler_level``.

This runs as part of Pass 3 (after Pass 2), so enrichment is always available.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

from .chunker import Chunk

logger = logging.getLogger(__name__)


def enrich_chunks(
    db: Any,
    novel_meta_id: int,
    sections: list[dict[str, Any]],
    chunks: list[Chunk],
) -> None:
    """Enrich ``chunks`` in place with speaker/character/event metadata.

    Args:
        db: novel_parser ``DatabaseManager`` (PostgreSQL).
        novel_meta_id: Parser-side novel id.
        sections: Section dicts with ``section_index`` and ``text``.
        chunks: Chunks produced by :func:`novel_rag.chunker.chunk_sections`.
    """
    text_by_chapter = {
        int(s.get("section_index", 0)): (s.get("text", "") or "") for s in sections
    }
    id_to_name = {c["id"]: c["name"] for c in db.get_characters(novel_meta_id)}

    chunks_by_chapter: dict[int, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_chapter.setdefault(chunk.chapter_index, []).append(chunk)

    for chapter_index, chapter_chunks in chunks_by_chapter.items():
        chapter_chunks.sort(key=lambda c: c.chunk_index_in_chapter)
        _enrich_chapter(
            db,
            novel_meta_id,
            chapter_index,
            text_by_chapter.get(chapter_index, ""),
            chapter_chunks,
            id_to_name,
        )


def _enrich_chapter(
    db: Any,
    novel_meta_id: int,
    chapter_index: int,
    chapter_text: str,
    chapter_chunks: list[Chunk],
    id_to_name: dict[int, str],
) -> None:
    entries = db.get_dialogue_entries(novel_meta_id, chapter_index)
    events = db.get_events_for_section(novel_meta_id, chapter_index)
    if not chapter_chunks:
        return

    # Map a char position to the chunk that contains it (fall back to the
    # nearest chunk so nothing is silently dropped).
    def chunk_for_pos(pos: int) -> Chunk:
        for chunk in chapter_chunks:
            if chunk.char_start <= pos < chunk.char_end:
                return chunk
        # Past the last span (overlap rounding) -> assign to the closest chunk.
        return min(chapter_chunks, key=lambda c: abs(c.char_start - pos))

    # seq -> chunk_serial, used to bucket events by sequence number.
    seq_to_serial: list[tuple[int, int]] = []

    cursor = 0
    for entry in entries:
        needle = entry.get("original_text") or entry.get("raw_text") or ""
        pos = _locate(chapter_text, needle, cursor)
        if pos >= 0:
            cursor = pos + max(1, len(needle))
        else:
            pos = cursor  # keep ordering even when the exact text isn't found

        chunk = chunk_for_pos(pos)
        seq_to_serial.append((int(entry["sequence_number"]), chunk.chunk_serial))

        speaker = entry.get("speaker_name") or ""
        if speaker and speaker not in chunk.speakers:
            chunk.speakers.append(speaker)
        for cid in entry.get("associated_characters") or []:
            name = id_to_name.get(cid)
            if name and name not in chunk.associated_characters:
                chunk.associated_characters.append(name)

    # Bucket events into the chunk owning the largest sequence_number <= event seq.
    seq_to_serial.sort(key=lambda x: x[0])
    seqs = [s for s, _ in seq_to_serial]
    serial_to_chunk = {c.chunk_serial: c for c in chapter_chunks}
    first_chunk = chapter_chunks[0]

    for event in events:
        event_seq = int(event["sequence_number"])
        idx = bisect.bisect_right(seqs, event_seq) - 1
        if idx >= 0:
            target = serial_to_chunk.get(seq_to_serial[idx][1], first_chunk)
        else:
            target = first_chunk
        target.event_ids.append(int(event["id"]))
        target.max_spoiler_level = max(
            target.max_spoiler_level, int(event.get("spoiler_level") or 0)
        )


def _locate(haystack: str, needle: str, start: int) -> int:
    """Find ``needle`` at/after ``start``; retry from 0; else use a prefix."""
    if not needle:
        return -1
    pos = haystack.find(needle, start)
    if pos >= 0:
        return pos
    pos = haystack.find(needle)
    if pos >= 0:
        return pos
    prefix = needle[:48]
    return haystack.find(prefix, start) if prefix else -1
