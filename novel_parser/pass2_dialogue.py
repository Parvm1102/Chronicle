"""Pass 2 — Paragraph-chunk dialogue analysis with sliding history.

Processes each section by:
1. Splitting text into blocks via TextSplitter
2. Grouping blocks into small chunks (~2 paragraphs)
3. Sending each chunk to the LLM with:
   - Character list + aliases
   - Sliding window of recent dialogue history
   - Lookahead context
4. Storing dialogue entries, profile updates, and events
5. Injecting Chatterbox TTS paralinguistic tags
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .database import DatabaseManager
from .llm_client import LLMClient
from .models import (
    EMOTION_INTENSITY_PROMPT_MAP,
    ChunkAnalysisResult,
    resolve_emotion_intensity,
)
from .text_splitter import TextBlock, TextSplitter

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Chatterbox supported tags
# ───────────────────────────────────────────────────────────────────────────

CHATTERBOX_TAGS = [
    "[sigh]", "[gasp]", "[cough]", "[laugh]", "[whisper]", "[breath]",
    "[chuckle]", "[clear throat]", "[sniff]", "[groan]", "[shush]",
]

# ───────────────────────────────────────────────────────────────────────────
# Prompt templates
# ───────────────────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You are an expert audiobook producer analysing novel text for TTS production.
You must identify speakers, emotions, and inject Chatterbox TTS tags.
Respond ONLY with valid JSON matching the schema below.
"""

_ANALYSIS_USER = """\
KNOWN CHARACTERS (use these exact names for speaker identification):
{characters_json}

RECENT DIALOGUE HISTORY (for speaker continuity context):
{history_lines}

--- ANALYSE THIS TEXT ---
{chunk_text}
--- END ---

UPCOMING TEXT (lookahead — do NOT analyse, only use for context):
{lookahead}

For EACH text block above (separated by blank lines or marked as dialogue/narration),\
 provide an entry with:
- entry_type: "dialogue" | "narration" | "thought" | "action"
- speaker: character name from the KNOWN CHARACTERS list, or "NARRATOR" for narration
- emotion + emotion_intensity: MUST be a valid combination from the list below
- raw_text: the spoken/narrated text with Chatterbox TTS tags injected where appropriate
- original_text: the exact source text (unchanged)
- associated_characters: list of character names present in or referenced by this block
- confidence: 0.0 to 1.0 (how confident you are about speaker attribution)

{emotion_map}

CHATTERBOX TTS TAGS — inject these INLINE where the source text implies them:
{tags_list}

Tag rules:
- ONLY use tags from the list above — no other tags
- Place at natural positions (start of sentence, between clauses, end)
- Only add when the original text explicitly or strongly implies the action
- Do NOT add tags for sounds not in the list (door slams, footsteps, etc.)
- The tag replaces the attribution verb (if text says "he laughed", output: [laugh] dialogue text)
- Do not over-tag — only where it clearly adds value

PROFILE UPDATES — if a character undergoes a significant emotional, relationship,\
 or status change in this chunk, include in "profile_updates":
- character_name, emotional_state, relationships (dict), knowledge (list), status, summary

EVENTS — if a significant plot event occurs, include in "events":
- event_type: plot/reveal/conflict/resolution/transition
- summary, characters_involved (names), speakers (names)
- importance: minor/moderate/major/critical
- spoiler_level: 0-10

Respond ONLY with valid JSON:
{{
  "entries": [
    {{"entry_type": "...", "speaker": "...", "emotion": "...", "emotion_intensity": "...",\
 "raw_text": "...", "original_text": "...", "associated_characters": [...], "confidence": 0.0}}
  ],
  "profile_updates": [],
  "events": []
}}
"""


class DialogueAnalyzer:
    """Orchestrates Pass 2: dialogue-by-dialogue analysis."""

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMClient,
        history_size: int = 8,
    ) -> None:
        self._db = db
        self._llm = llm
        self._splitter = TextSplitter()
        self._history_size = history_size

    def run(
        self,
        novel_meta_id: int,
        sections: list[dict[str, Any]],
        characters: list[dict[str, Any]],
    ) -> None:
        """Run Pass 2 on all sections.

        Args:
            novel_meta_id: The novel's PG id.
            sections: Section dicts from SQLite (title, text, section_index).
            characters: Character dicts from PostgreSQL (after Pass 1).
        """
        total = len(sections)
        logger.info("Pass 2: starting dialogue analysis for %d sections", total)

        # Build character reference for prompts
        char_prompt_data = self._build_char_prompt_data(characters)
        char_name_to_id = self._build_name_map(characters)

        # Sliding dialogue history
        history: list[str] = []

        # Global sequence counter across sections
        global_seq = 0

        # Update progress
        self._db.upsert_parse_progress(
            novel_meta_id, pass_number=2,
            total_sections=total, status="running",
        )

        for sec_idx, section in enumerate(sections):
            section_index = section.get("section_index", sec_idx)
            section_title = section.get("title", f"Section {sec_idx + 1}")
            section_text = section.get("text", "")

            if not section_text.strip():
                self._db.upsert_parse_progress(
                    novel_meta_id, pass_number=2,
                    current_section=sec_idx + 1, total_sections=total,
                    status="running",
                )
                continue

            logger.info(
                "Pass 2: section %d/%d — '%s'", sec_idx + 1, total, section_title
            )

            # Split section into blocks, then chunks
            blocks = self._splitter.split(section_text)
            chunks = self._splitter.split_into_chunks(blocks)

            for chunk_idx, chunk_blocks in enumerate(chunks):
                # Build context
                chunk_text = self._format_chunk(chunk_blocks)

                # Lookahead: next chunk's text
                lookahead = ""
                if chunk_idx + 1 < len(chunks):
                    lookahead = self._format_chunk(chunks[chunk_idx + 1])
                elif sec_idx + 1 < total:
                    # Peek at start of next section
                    next_text = sections[sec_idx + 1].get("text", "")
                    if next_text:
                        next_blocks = self._splitter.split(next_text)
                        if next_blocks:
                            lookahead = self._format_chunk(next_blocks[:3])

                # Call LLM
                try:
                    result = self._analyse_chunk(
                        char_prompt_data, history, chunk_text, lookahead
                    )
                except Exception as exc:
                    logger.error(
                        "Pass 2 LLM error on section %d chunk %d: %s",
                        sec_idx + 1, chunk_idx + 1, exc,
                    )
                    # Store blocks as-is with unknown speaker
                    for block in chunk_blocks:
                        self._db.insert_dialogue_entry(
                            novel_meta_id, section_index, global_seq,
                            entry_type=block.block_type,
                            raw_text=block.text,
                            original_text=block.text,
                            speaker_name="UNKNOWN",
                            emotion="neutral",
                            emotion_intensity="low",
                            llm_confidence=0.0,
                        )
                        global_seq += 1
                    continue

                # Align original blocks with LLM entries to prevent skipping
                unmatched_entries = list(result.entries)

                for block in chunk_blocks:
                    best_entry = None
                    best_score = 0.0

                    # Find the LLM entry that matches this original block best
                    for entry in unmatched_entries:
                        orig = (entry.original_text or entry.raw_text or "").strip()
                        if not orig:
                            continue

                        # Score based on word overlap ratio
                        block_words = set(block.text.lower().split())
                        entry_words = set(orig.lower().split())
                        if not block_words or not entry_words:
                            score = 0.0
                        else:
                            intersection = block_words & entry_words
                            score = len(intersection) / max(len(block_words), len(entry_words))

                        # Boost score if one string fully contains the other
                        if orig.lower() in block.text.lower() or block.text.lower() in orig.lower():
                            score += 0.5

                        if score > best_score:
                            best_score = score
                            best_entry = entry

                    # Use matched LLM entry if it meets a similarity threshold
                    if best_entry and best_score > 0.3:
                        unmatched_entries.remove(best_entry)

                        speaker_name = best_entry.speaker or ("NARRATOR" if block.block_type == "narration" else "UNKNOWN")
                        speaker_id = char_name_to_id.get(speaker_name)

                        # Validate and snap emotion/intensity
                        emotion, intensity = resolve_emotion_intensity(
                            best_entry.emotion, best_entry.emotion_intensity
                        )

                        # Resolve associated characters
                        assoc_ids = [
                            char_name_to_id[n]
                            for n in (best_entry.associated_characters or [])
                            if n in char_name_to_id
                        ]

                        # Detect Chatterbox tags from best_entry.raw_text and inject into original text
                        raw_text = block.text
                        entry_raw = best_entry.raw_text or ""
                        for tag in CHATTERBOX_TAGS:
                            if tag in entry_raw:
                                raw_text = f"{tag} {raw_text}"
                                break

                        self._db.insert_dialogue_entry(
                            novel_meta_id, section_index, global_seq,
                            entry_type=block.block_type,
                            raw_text=raw_text,
                            original_text=block.text,
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                            emotion=emotion,
                            emotion_intensity=intensity,
                            associated_characters=assoc_ids,
                            context_before="\n".join(history[-2:]) if history else "",
                            context_after=lookahead[:200] if lookahead else "",
                            llm_confidence=best_entry.confidence or 0.5,
                        )

                        # Update history buffer
                        history_line = (
                            f"{speaker_name}: {emotion}_{intensity} — "
                            f"{raw_text[:60]}..."
                            if len(raw_text) > 60
                            else f"{speaker_name}: {emotion}_{intensity} — {raw_text}"
                        )
                        history.append(history_line)
                        if len(history) > self._history_size:
                            history = history[-self._history_size:]

                    else:
                        # Fallback for this specific block: keep original text, default metadata
                        speaker_name = "NARRATOR" if block.block_type == "narration" else "UNKNOWN"
                        speaker_id = char_name_to_id.get(speaker_name)

                        self._db.insert_dialogue_entry(
                            novel_meta_id, section_index, global_seq,
                            entry_type=block.block_type,
                            raw_text=block.text,
                            original_text=block.text,
                            speaker_id=speaker_id,
                            speaker_name=speaker_name,
                            emotion="neutral",
                            emotion_intensity="low",
                            associated_characters=[],
                            context_before="\n".join(history[-2:]) if history else "",
                            context_after=lookahead[:200] if lookahead else "",
                            llm_confidence=0.0,
                        )

                    global_seq += 1

                # Store profile updates
                for update in result.profile_updates:
                    char_id = char_name_to_id.get(update.character_name)
                    if char_id:
                        self._db.insert_character_profile(
                            char_id, novel_meta_id, section_index,
                            profile_type="update",
                            emotional_state=update.emotional_state,
                            relationships=update.relationships,
                            knowledge=update.knowledge,
                            status=update.status,
                            summary=update.summary,
                        )

                # Store events
                for evt_idx, event in enumerate(result.events):
                    involved_ids = [
                        char_name_to_id[n]
                        for n in event.characters_involved
                        if n in char_name_to_id
                    ]
                    speaker_ids = [
                        char_name_to_id[n]
                        for n in event.speakers
                        if n in char_name_to_id
                    ]
                    self._db.insert_event(
                        novel_meta_id, section_index,
                        sequence_number=global_seq + evt_idx,
                        event_type=event.event_type,
                        summary=event.summary,
                        characters_involved=involved_ids,
                        speakers=speaker_ids,
                        importance=event.importance,
                        spoiler_level=event.spoiler_level,
                    )

            # Update progress
            self._db.upsert_parse_progress(
                novel_meta_id, pass_number=2,
                current_section=sec_idx + 1, total_sections=total,
                status="running",
            )

        # Mark complete
        self._db.mark_progress_complete(novel_meta_id, pass_number=2)
        logger.info("Pass 2 complete: %d entries stored", global_seq)

    # ── LLM call ───────────────────────────────────────────────────────────

    def _analyse_chunk(
        self,
        char_prompt_data: str,
        history: list[str],
        chunk_text: str,
        lookahead: str,
    ) -> ChunkAnalysisResult:
        """Send a single chunk to the LLM for analysis."""
        history_str = "\n".join(history[-self._history_size:]) if history else "(start of novel)"
        tags_str = ", ".join(CHATTERBOX_TAGS)

        user_msg = _ANALYSIS_USER.format(
            characters_json=char_prompt_data,
            history_lines=history_str,
            chunk_text=chunk_text,
            lookahead=lookahead[:500] if lookahead else "(end of section)",
            tags_list=tags_str,
            emotion_map=EMOTION_INTENSITY_PROMPT_MAP,
        )

        return self._llm.chat_structured(
            messages=[
                {"role": "system", "content": _ANALYSIS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_model=ChunkAnalysisResult,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_char_prompt_data(characters: list[dict[str, Any]]) -> str:
        """Build a compact character reference for prompts."""
        chars = []
        for c in characters:
            entry = {
                "name": c["name"],
                "aliases": c.get("aliases", []),
                "role": c.get("role", "minor"),
            }
            chars.append(entry)
        return json.dumps(chars, indent=2)

    @staticmethod
    def _build_name_map(characters: list[dict[str, Any]]) -> dict[str, int]:
        """Build a name/alias → character_id lookup."""
        name_map: dict[str, int] = {}
        for c in characters:
            char_id = c["id"]
            name_map[c["name"]] = char_id
            name_map[c["name"].lower()] = char_id
            for alias in c.get("aliases", []):
                name_map[alias] = char_id
                name_map[alias.lower()] = char_id
        # Always map NARRATOR
        narrator_chars = [c for c in characters if c.get("is_narrator")]
        if narrator_chars:
            name_map["NARRATOR"] = narrator_chars[0]["id"]
        return name_map

    @staticmethod
    def _format_chunk(blocks: list[TextBlock]) -> str:
        """Format blocks into readable text for the LLM prompt."""
        lines = []
        for block in blocks:
            if block.block_type == "dialogue":
                lines.append(f'"{ block.text}"')
            elif block.block_type == "thought":
                lines.append(f"*{block.text}*")
            elif block.block_type == "action":
                lines.append(f"[{block.text}]")
            else:
                lines.append(block.text)
        return "\n\n".join(lines)
