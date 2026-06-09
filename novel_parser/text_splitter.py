"""Regex-based text splitter: breaks section text into dialogue, narration,
thought, and action blocks for Pass 2 processing.

Each block preserves its position in the source text so that context windows
and sequence numbering are straightforward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextBlock:
    """A single unit of text extracted from a section."""
    index: int                       # position in the section's block list
    text: str                        # the raw text content
    block_type: str = "narration"    # dialogue / narration / thought / action
    paragraph_index: int = 0         # which paragraph this came from


# ───────────────────────────────────────────────────────────────────────────
# Patterns
# ───────────────────────────────────────────────────────────────────────────

# Matches text within double quotes, smart double quotes, or guillemets.
# Single quotes are intentionally NOT treated as dialogue: in English prose they
# collide with contractions and possessives (don't, it's, the dog's bone).
_DIALOGUE_RE = re.compile(
    r'("(?:[^"\\]|\\.)*"'           # "double quoted"
    r"|\u201c[^\u201d]*\u201d"      # “smart double quoted”
    r"|«[^»]*»"                     # «guillemets»
    r'|"[^"\u201d]*[\u201d"]'       # "mixed smart double close”
    r")",
    re.DOTALL,
)

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_BLOCK_CHARS = 1400
_MAX_CHUNK_CHARS = 3200

# Italicised text often indicates inner thoughts
_THOUGHT_RE = re.compile(
    r"(?:<em>|<i>|\*)(.*?)(?:</em>|</i>|\*)",
    re.DOTALL,
)

# Action lines — typically short, often surrounded by asterisks or in brackets
_ACTION_RE = re.compile(
    r"(?:^\s*\*[^*]+\*\s*$"         # *action text*
    r"|^\s*\[[^\]]+\]\s*$"          # [action text]
    r")",
    re.MULTILINE,
)

# Section titles that indicate non-story front/back matter.
_FRONT_MATTER_TITLE_RE = re.compile(
    r"\b(preface|foreword|author'?s?\s+note|translator'?s?\s+note|"
    r"dedication|acknowledge?ments?|copyright|about\s+the\s+author|"
    r"title\s+page|table\s+of\s+contents|colophon|epigraph|"
    r"publisher'?s?\s+note|introduction)\b",
    re.IGNORECASE,
)


def is_front_matter(title: str) -> bool:
    """Return True if a section title indicates non-story front/back matter.

    Front matter (preface, author's note, dedication, etc.) has no characters
    and must not influence narrator detection. It is still stored and voiced by
    MISC_VOICE so the reader can play it when on that section.
    """
    if not title:
        return False
    return bool(_FRONT_MATTER_TITLE_RE.search(title))


class TextSplitter:
    """Split section text into ordered TextBlocks.

    Strategy:
    1. Split by paragraphs (double newline or single newline for short lines)
    2. Within each paragraph, identify dialogue spans
    3. Everything outside dialogue = narration (unless it matches thought/action)
    4. Return a flat, ordered list of blocks
    """

    def split(self, text: str) -> list[TextBlock]:
        """Split *text* into an ordered list of TextBlocks."""
        if not text or not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        blocks: list[TextBlock] = []
        block_idx = 0

        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Check for pure action lines first
            if _ACTION_RE.fullmatch(para):
                blocks.append(TextBlock(
                    index=block_idx,
                    text=para.strip("*[] \t"),
                    block_type="action",
                    paragraph_index=para_idx,
                ))
                block_idx += 1
                continue

            # Split paragraph into dialogue / non-dialogue spans
            para_blocks = self._split_dialogue(para, para_idx, block_idx)
            for b in para_blocks:
                for part in self._split_long_block(b, _MAX_BLOCK_CHARS):
                    part.index = block_idx
                    blocks.append(part)
                    block_idx += 1

        return blocks

    def split_into_chunks(
        self,
        blocks: list[TextBlock],
        max_blocks_per_chunk: int = 6,
        max_chars_per_chunk: int = _MAX_CHUNK_CHARS,
    ) -> list[list[TextBlock]]:
        """Group blocks into chunks for LLM processing.

        Tries to keep paragraph boundaries together; falls back to
        *max_blocks_per_chunk* and *max_chars_per_chunk* as hard limits.
        """
        if not blocks:
            return []

        chunks: list[list[TextBlock]] = []
        current_chunk: list[TextBlock] = []
        current_para = blocks[0].paragraph_index if blocks else 0
        para_count = 0
        current_chars = 0

        for block in blocks:
            # New paragraph?
            if block.paragraph_index != current_para:
                para_count += 1
                current_para = block.paragraph_index

            # Start new chunk after ~2 paragraphs or max blocks
            next_size = current_chars + len(block.text) + (2 if current_chunk else 0)
            if (
                para_count >= 2
                or len(current_chunk) >= max_blocks_per_chunk
                or next_size > max_chars_per_chunk
            ) and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                para_count = 0
                current_chars = 0

            current_chunk.append(block)
            current_chars += len(block.text) + (2 if len(current_chunk) > 1 else 0)

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double newlines; keep single-newline lines as separate paragraphs
        only if they are short (likely dialogue tags)."""
        # First split on double newlines
        raw_paras = re.split(r"\n\s*\n", text)
        raw_paras = [p.strip() for p in raw_paras if p.strip()]

        # If we got only 1 paragraph but there are multiple single newlines, split on single newlines
        if len(raw_paras) <= 1 and text.count("\n") > 5:
            raw_paras = re.split(r"\n+", text)
            raw_paras = [p.strip() for p in raw_paras if p.strip()]

        return raw_paras

    @staticmethod
    def _split_dialogue(
        paragraph: str,
        para_idx: int,
        start_block_idx: int,
    ) -> list[TextBlock]:
        """Split a single paragraph into dialogue and narration blocks."""
        blocks: list[TextBlock] = []
        idx = start_block_idx
        last_end = 0

        for m in _DIALOGUE_RE.finditer(paragraph):
            # Narration before this dialogue
            before = paragraph[last_end:m.start()].strip()
            if before:
                # Check if it looks like a thought
                block_type = "narration"
                thought_match = _THOUGHT_RE.search(before)
                if thought_match and len(thought_match.group(0)) > len(before) * 0.5:
                    block_type = "thought"

                blocks.append(TextBlock(
                    index=idx, text=before,
                    block_type=block_type, paragraph_index=para_idx,
                ))
                idx += 1

            # The dialogue itself — strip the outer quotes
            dialogue_text = m.group(0)
            # Remove surrounding quote marks
            if len(dialogue_text) >= 2:
                first, last = dialogue_text[0], dialogue_text[-1]
                if first in ('"', "'", "«", "\u201c", "\u2018"):
                    dialogue_text = dialogue_text[1:]
                if last in ('"', "'", "»", "\u201d", "\u2019"):
                    dialogue_text = dialogue_text[:-1]

            blocks.append(TextBlock(
                index=idx, text=dialogue_text.strip(),
                block_type="dialogue", paragraph_index=para_idx,
            ))
            idx += 1
            last_end = m.end()

        # Remaining narration after last dialogue
        after = paragraph[last_end:].strip()
        if after:
            block_type = "narration"
            thought_match = _THOUGHT_RE.search(after)
            if thought_match and len(thought_match.group(0)) > len(after) * 0.5:
                block_type = "thought"

            blocks.append(TextBlock(
                index=idx, text=after,
                block_type=block_type, paragraph_index=para_idx,
            ))

        # If no dialogue was found at all, the whole paragraph is narration
        if not blocks:
            blocks.append(TextBlock(
                index=start_block_idx, text=paragraph,
                block_type="narration", paragraph_index=para_idx,
            ))

        return blocks

    @staticmethod
    def _split_long_block(block: TextBlock, max_chars: int) -> list[TextBlock]:
        """Split oversized narration/dialogue blocks so one paragraph cannot dominate a prompt."""
        if len(block.text) <= max_chars:
            return [block]

        parts: list[str] = []
        current = ""

        for sentence in _SENTENCE_BOUNDARY_RE.split(block.text):
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    parts.append(current)
                    current = ""
                parts.extend(TextSplitter._split_long_piece(sentence, max_chars))
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > max_chars:
                if current:
                    parts.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            parts.append(current)

        return [
            TextBlock(
                index=block.index,
                text=part,
                block_type=block.block_type,
                paragraph_index=block.paragraph_index,
            )
            for part in parts
        ] or [block]

    @staticmethod
    def _split_long_piece(text: str, max_chars: int) -> list[str]:
        """Split text with no sentence boundaries at whitespace near max_chars."""
        words = text.split()
        if not words:
            return [text[:max_chars]]

        parts: list[str] = []
        current: list[str] = []
        current_len = 0

        for word in words:
            if len(word) > max_chars:
                if current:
                    parts.append(" ".join(current))
                    current = []
                    current_len = 0
                parts.extend(word[i:i + max_chars] for i in range(0, len(word), max_chars))
                continue

            extra = len(word) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                parts.append(" ".join(current))
                current = [word]
                current_len = len(word)
            else:
                current.append(word)
                current_len += extra

        if current:
            parts.append(" ".join(current))
        return parts

    @staticmethod
    def blocks_to_text(blocks: list[TextBlock]) -> str:
        """Reconstruct readable text from blocks (for context windows)."""
        return "\n".join(b.text for b in blocks)
