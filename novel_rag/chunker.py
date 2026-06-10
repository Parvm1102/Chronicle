"""Chapter- and paragraph-aware greedy chunking with serial numbers + char spans.

Two hard boundaries:
- **Chapter**: each chapter (section) is chunked independently — a chunk never
  spans two chapters.
- **Paragraph**: the chunk *body* is packed with whole paragraphs (greedy — fit
  as many complete paragraphs as the token budget allows; a paragraph that won't
  fully fit starts the next chunk; never split a paragraph unless it alone
  exceeds the budget).

Overlap is sentence-granular: each chunk's ``char_start`` is extended backward to
include the last ~``chunk_overlap`` tokens of the previous chunk, snapped to a
sentence boundary. This keeps a consistent overlap regardless of paragraph size
while leaving ``text == chapter_text[char_start:char_end]`` real and contiguous,
so dialogue/event attribution during enrichment stays clean.

Hierarchical (parent/child) chunking is intentionally deferred — a ``parent_id``
payload slot is reserved in the indexer so it can be added later without a
re-model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Chunk:
    """A single retrievable unit of novel text."""

    chapter_index: int
    chunk_index_in_chapter: int
    chunk_serial: int
    chapter_title: str
    text: str
    char_start: int
    char_end: int
    # Filled in later by metadata enrichment (kept here so the type is complete).
    speakers: list[str] = field(default_factory=list)
    associated_characters: list[str] = field(default_factory=list)
    event_ids: list[int] = field(default_factory=list)
    max_spoiler_level: int = 0


def chunk_sections(
    sections: list[dict[str, Any]],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split all chapters of a novel into ordered, serially-numbered chunks.

    Greedy whole-paragraph packing per chapter, with sentence-granular overlap
    between adjacent chunks (see module docstring).

    Args:
        sections: Section dicts with ``section_index``, ``title``, ``text``.
        chunk_size: Target tokens per chunk.
        chunk_overlap: Token overlap (tail of the previous chunk) between chunks.

    Returns:
        Flat list of :class:`Chunk` ordered by (chapter_index, position).
    """
    # Imported lazily so importing novel_rag.config etc. doesn't require the
    # (heavier) llama-index stack until indexing actually runs.
    from llama_index.core.utils import get_tokenizer

    tokenizer = get_tokenizer()
    token_len: Callable[[str], int] = lambda s: len(tokenizer(s))

    chunks: list[Chunk] = []
    serial = 0

    for section in sorted(sections, key=lambda s: s.get("section_index", 0)):
        text = section.get("text", "") or ""
        if not text.strip():
            continue
        serial = _chunk_chapter(
            text=text,
            chapter_index=int(section.get("section_index", 0)),
            title=section.get("title", "") or "",
            serial=serial,
            chunks=chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            token_len=token_len,
        )

    return chunks


def _chunk_chapter(
    *,
    text: str,
    chapter_index: int,
    title: str,
    serial: int,
    chunks: list[Chunk],
    chunk_size: int,
    chunk_overlap: int,
    token_len: Callable[[str], int],
) -> int:
    """Greedily pack whole paragraphs of one chapter into chunks.

    Returns the next free serial number.
    """
    position = 0
    prev_last: tuple[int, int] | None = None  # span of previous chunk's tail
    current: list[tuple[int, int, int]] = []   # (start, end, tokens) paragraphs
    current_tokens = 0

    def emit(body_start: int, body_end: int) -> None:
        nonlocal serial, position, prev_last
        start = body_start
        if prev_last is not None and chunk_overlap > 0:
            start = _overlap_start(text, *prev_last, chunk_overlap, token_len)
        chunks.append(
            Chunk(
                chapter_index=chapter_index,
                chunk_index_in_chapter=position,
                chunk_serial=serial,
                chapter_title=title,
                text=text[start:body_end],
                char_start=start,
                char_end=body_end,
            )
        )
        serial += 1
        position += 1
        prev_last = (body_start, body_end)

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        emit(current[0][0], current[-1][1])
        current = []
        current_tokens = 0

    for p_start, p_end, p_tokens in _paragraphs_with_spans(text, token_len):
        # A single paragraph larger than the budget: flush, then sentence-pack it.
        if p_tokens > chunk_size:
            flush()
            for g_start, g_end in _pack_sentences(text, p_start, p_end, chunk_size, token_len):
                emit(g_start, g_end)
            continue

        # Next paragraph would overflow the current chunk -> close it first.
        if current and current_tokens + p_tokens > chunk_size:
            flush()

        current.append((p_start, p_end, p_tokens))
        current_tokens += p_tokens

    flush()
    return serial


def _overlap_start(
    text: str,
    start: int,
    end: int,
    overlap_tokens: int,
    token_len: Callable[[str], int],
) -> int:
    """Walk back from the previous chunk's end to ~overlap_tokens, snapped to a sentence."""
    chosen = end
    total = 0
    for s_text, s_off in reversed(_split_sentences(text[start:end])):
        chosen = start + s_off
        total += token_len(s_text)
        if total >= overlap_tokens:
            break
    return chosen


def _pack_sentences(
    text: str,
    start: int,
    end: int,
    chunk_size: int,
    token_len: Callable[[str], int],
) -> list[tuple[int, int]]:
    """Greedily group an oversized paragraph's sentences into (start, end) spans."""
    groups: list[tuple[int, int]] = []
    g_start: int | None = None
    g_end = start
    g_tokens = 0
    for s_text, s_off in _split_sentences(text[start:end]):
        s_abs = start + s_off
        s_tokens = token_len(s_text)
        if g_start is not None and g_tokens + s_tokens > chunk_size:
            groups.append((g_start, g_end))
            g_start, g_tokens = None, 0
        if g_start is None:
            g_start = s_abs
        g_end = s_abs + len(s_text)
        g_tokens += s_tokens
    if g_start is not None:
        groups.append((g_start, g_end))
    return groups or [(start, end)]


_SENTENCE_RE = re.compile(r'.*?[.!?]["\u201d\u2019\')\]]*(?:\s+|$)|.+$', re.DOTALL)


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Split into (sentence, offset) pairs, preserving char offsets."""
    return [(m.group(0), m.start()) for m in _SENTENCE_RE.finditer(text) if m.group(0).strip()]


def _paragraphs_with_spans(
    text: str, token_len: Callable[[str], int]
) -> list[tuple[int, int, int]]:
    """Split text into (char_start, char_end, tokens) paragraphs, preserving offsets.

    Splits on blank lines; falls back to single newlines when the text has many
    line breaks but no blank-line paragraph structure (common in plain TXT).
    """
    def collect(pattern: str) -> list[tuple[int, int, int]]:
        spans: list[tuple[int, int, int]] = []
        pos = 0
        for match in re.finditer(pattern, text):
            segment = text[pos:match.start()]
            if segment.strip():
                spans.append((pos, match.start(), token_len(segment)))
            pos = match.end()
        tail = text[pos:]
        if tail.strip():
            spans.append((pos, len(text), token_len(tail)))
        return spans

    spans = collect(r"\n\s*\n")
    if len(spans) <= 1 and text.count("\n") > 5:
        spans = collect(r"\n+")
    return spans

