"""Regex-based text splitter: breaks section text into dialogue, narration,
thought, and action blocks for Pass 2 processing.

Each block preserves its position in the source text so that context windows
and sequence numbering are straightforward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


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

# Matches text within double quotes, single quotes, or guillemets
_DIALOGUE_RE = re.compile(
    r'("(?:[^"\\]|\\.)*"'           # "double quoted"
    r"|'(?:[^'\\]|\\.)*'"           # 'single quoted'  (smart quotes)
    r"|«[^»]*»"                     # «guillemets»
    r'|"[^"\u201d]*[\u201d"]'       # "smart double quotes"
    r"|\u2018[^\u2019]*\u2019"      # 'smart single quotes'
    r")",
    re.DOTALL,
)

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
                blocks.append(b)
                block_idx += 1

        return blocks

    def split_into_chunks(
        self,
        blocks: list[TextBlock],
        max_blocks_per_chunk: int = 6,
    ) -> list[list[TextBlock]]:
        """Group blocks into chunks for LLM processing.

        Tries to keep paragraph boundaries together; falls back to
        *max_blocks_per_chunk* as the hard limit.
        """
        if not blocks:
            return []

        chunks: list[list[TextBlock]] = []
        current_chunk: list[TextBlock] = []
        current_para = blocks[0].paragraph_index if blocks else 0
        para_count = 0

        for block in blocks:
            # New paragraph?
            if block.paragraph_index != current_para:
                para_count += 1
                current_para = block.paragraph_index

            # Start new chunk after ~2 paragraphs or max blocks
            if (para_count >= 2 or len(current_chunk) >= max_blocks_per_chunk) and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                para_count = 0

            current_chunk.append(block)

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
        return [p.strip() for p in raw_paras if p.strip()]

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
    def blocks_to_text(blocks: list[TextBlock]) -> str:
        """Reconstruct readable text from blocks (for context windows)."""
        return "\n".join(b.text for b in blocks)
