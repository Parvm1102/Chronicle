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
import threading
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
Respond ONLY with valid JSON matching the schema below. Do NOT output any preamble, markdown code blocks, XML tags, conversational intro/outro, <think>, <thought>, chain-of-thought, analysis, or reasoning notes. Start your response directly with the opening brace '{'.
"""

_ANALYSIS_USER = """\
/no_think

KNOWN CHARACTERS (use these exact names for speaker identification):
{characters_json}

If one known character has "is_narrator": true, "NARRATOR" entries are voiced by\
 that character. Still use "NARRATOR" for narration entries; the parser resolves\
 it to the narrator character.

PRECEDING TEXT (context only — do NOT analyse):
{prev_chunk_text}

RECENT SPEAKER HISTORY (for continuity):
{history_lines}

--- ANALYSE THIS TEXT ---
{chunk_text}
--- END ---

UPCOMING TEXT (context only — do NOT analyse):
{lookahead}

CRITICAL RULE — SPLITTING MIXED PARAGRAPHS:
When a paragraph contains BOTH dialogue and narration interleaved, you MUST\
 split them into separate entries in reading order. Example:

Input: "Look what I shot," Gale holds up a loaf of bread with an arrow stuck\
 in it, and I laugh.

Expected output (2 entries):
  1. entry_type: "dialogue", speaker: "Gale", original_text: "Look what I shot,"
  2. entry_type: "narration", speaker: "NARRATOR", original_text: "Gale holds up a loaf of bread with an arrow stuck in it, and I laugh."

Every distinct dialogue quote MUST be a separate entry from the surrounding\
 narration. Never merge dialogue and narration into one entry.

For EACH text segment (in reading order), provide an entry with:
- entry_type: "dialogue" | "narration" | "thought" | "action"
- speaker: character name from the KNOWN CHARACTERS list, or "NARRATOR" for narration,\
 or "MISC_VOICE" for non-character speakers (PA systems, machine voices, TV/radio broadcasts,\
 crowd chants, written signs/letters read aloud)
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

Respond ONLY with valid JSON. Do NOT think out loud, write markdown lists, or output reasoning before outputting the JSON. Start your response immediately with the opening curly brace '{{' of the JSON:
Do not include <think>, <thought>, markdown, notes, bullets, or analysis.
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
        concurrency: int = 4,
    ) -> None:
        self._db = db
        self._llm = llm
        self._splitter = TextSplitter()
        self._history_size = history_size
        self._concurrency = concurrency

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
        logger.info("Pass 2: starting dialogue analysis for %d sections with concurrency %d", total, self._concurrency)

        narrator_character_id = self._db.get_narrator_character_id(novel_meta_id)
        characters = self._apply_narrator_flag(characters, narrator_character_id)

        # Build character reference for prompts
        char_prompt_data = self._build_char_prompt_data(characters)
        speaker_lookup = self._build_speaker_lookup(characters)

        # Get the set of already completed section indices in the database
        completed_sections = self._db.get_completed_dialogue_sections(novel_meta_id)
        if completed_sections:
            logger.info(
                "Pass 2: resuming — %d sections already completed in DB",
                len(completed_sections),
            )

        # Pre-calculate block counts for all sections to assign global_seq deterministically
        section_block_counts = []
        for idx, section in enumerate(sections):
            text = section.get("text", "")
            blocks = self._splitter.split(text)
            section_block_counts.append(len(blocks))

        # Initialise completed sections counter (pre-populated with skipped sections)
        self._completed_sections = len(completed_sections)

        # Update initial progress
        self._db.upsert_parse_progress(
            novel_meta_id, pass_number=2,
            current_section=self._completed_sections, total_sections=total,
            status="running",
        )

        completed_sections_lock = threading.Lock()

        # Identify sections that need to be processed
        sections_to_process = []
        for sec_idx, section in enumerate(sections):
            section_index = section.get("section_index", sec_idx)
            if section_index in completed_sections:
                logger.info("Pass 2: skipping section %d (already completed)", section_index)
                continue

            # If section text is blank, we can just treat it as completed
            section_text = section.get("text", "")
            if not section_text.strip():
                logger.info("Pass 2: skipping section %d (empty text)", section_index)
                with completed_sections_lock:
                    self._completed_sections += 1
                    self._db.upsert_parse_progress(
                        novel_meta_id, pass_number=2,
                        current_section=self._completed_sections, total_sections=total,
                        status="running",
                    )
                continue

            # Calculate the starting sequence number for this section
            start_seq = sum(section_block_counts[:sec_idx])
            sections_to_process.append((sec_idx, section, start_seq))

        if sections_to_process:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._concurrency) as executor:
                # Submit tasks to executor
                futures = {
                    executor.submit(
                        self._process_section,
                        novel_meta_id,
                        sec_idx,
                        section,
                        start_seq,
                        char_prompt_data,
                        speaker_lookup,
                        total,
                        sections,
                        completed_sections_lock,
                    ): sec_idx
                    for sec_idx, section, start_seq in sections_to_process
                }

                # Check results and raise exceptions if any occurred in the threads
                for future in concurrent.futures.as_completed(futures):
                    sec_idx = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            "Pass 2: Section %d failed with exception: %s",
                            sec_idx + 1, exc, exc_info=True
                        )
                        raise exc

        # Mark complete
        self._db.mark_progress_complete(novel_meta_id, pass_number=2)
        total_blocks = sum(section_block_counts)
        logger.info("Pass 2 complete: %d blocks processed across all sections", total_blocks)

    def _process_section(
        self,
        novel_meta_id: int,
        sec_idx: int,
        section: dict[str, Any],
        start_seq: int,
        char_prompt_data: str,
        speaker_lookup: dict[str, tuple[int | None, str]],
        total: int,
        sections: list[dict[str, Any]],
        completed_sections_lock: threading.Lock,
    ) -> None:
        section_index = section.get("section_index", sec_idx)
        section_title = section.get("title", f"Section {sec_idx + 1}")
        section_text = section.get("text", "")

        logger.info(
            "Pass 2: processing section %d/%d — '%s'", sec_idx + 1, total, section_title
        )

        blocks = self._splitter.split(section_text)
        chunks = self._splitter.split_into_chunks(blocks)

        history: list[str] = []
        prev_chunk_text = ""

        # Seed prev_chunk_text with the last chunk of the previous section
        if sec_idx > 0:
            prev_sec_text = sections[sec_idx - 1].get("text", "")
            if prev_sec_text:
                prev_blocks = self._splitter.split(prev_sec_text)
                if prev_blocks:
                    prev_chunks = self._splitter.split_into_chunks(prev_blocks)
                    if prev_chunks:
                        prev_chunk_text = self._format_chunk(prev_chunks[-1])

        # Prepare lists to buffer inserts
        dialogue_inserts = []
        profile_inserts = []
        event_inserts = []

        local_seq = start_seq

        for chunk_idx, chunk_blocks in enumerate(chunks):
            chunk_text = self._format_chunk(chunk_blocks)
            logger.info(
                "Pass 2: section %d/%d chunk %d/%d (%d blocks, %d chars)",
                sec_idx + 1, total, chunk_idx + 1, len(chunks),
                len(chunk_blocks), len(chunk_text),
            )
            self._db.set_parse_status(
                novel_meta_id,
                "pass2_running",
                (
                    f"Analysing dialogue: section {sec_idx + 1}/{total}, "
                    f"chunk {chunk_idx + 1}/{len(chunks)}"
                ),
            )

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

            if self._is_deterministic_chunk(chunk_blocks):
                for block in chunk_blocks:
                    speaker_id, speaker_name = self._resolve_speaker(
                        None, block.block_type, speaker_lookup
                    )
                    dialogue_inserts.append({
                        "entry_type": block.block_type,
                        "raw_text": block.text,
                        "original_text": block.text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "emotion": "neutral",
                        "emotion_intensity": "low",
                        "associated_characters": [],
                        "context_before": "\n".join(history[-2:]) if history else "",
                        "context_after": lookahead[:200] if lookahead else "",
                        "llm_confidence": 1.0,
                        "sequence_number": local_seq,
                    })
                    local_seq += 1
                prev_chunk_text = chunk_text
                continue

            # Call LLM
            try:
                result = self._analyse_chunk(
                    char_prompt_data, history, chunk_text, lookahead,
                    prev_chunk_text=prev_chunk_text,
                )
            except Exception as exc:
                logger.error(
                    "Pass 2 LLM error on section %d chunk %d: %s",
                    sec_idx + 1, chunk_idx + 1, exc,
                )
                # Store blocks as-is with deterministic fallback speakers.
                for block in chunk_blocks:
                    speaker_id, speaker_name = self._resolve_speaker(
                        None, block.block_type, speaker_lookup
                    )
                    dialogue_inserts.append({
                        "entry_type": block.block_type,
                        "raw_text": block.text,
                        "original_text": block.text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "emotion": "neutral",
                        "emotion_intensity": "low",
                        "associated_characters": [],
                        "context_before": "\n".join(history[-2:]) if history else "",
                        "context_after": lookahead[:200] if lookahead else "",
                        "llm_confidence": 0.0,
                        "sequence_number": local_seq,
                    })
                    local_seq += 1
                continue

            # Align original blocks with LLM entries to prevent skipping
            unmatched_entries = list(result.entries)

            for block in chunk_blocks:
                best_entry = None
                best_score = 0.0

                for entry in unmatched_entries:
                    orig = (entry.original_text or entry.raw_text or "").strip()
                    if not orig:
                        continue

                    block_words = set(block.text.lower().split())
                    entry_words = set(orig.lower().split())
                    if not block_words or not entry_words:
                        score = 0.0
                    else:
                        intersection = block_words & entry_words
                        score = len(intersection) / max(len(block_words), len(entry_words))

                    if orig.lower() in block.text.lower() or block.text.lower() in orig.lower():
                        score += 0.5

                    if score > best_score:
                        best_score = score
                        best_entry = entry

                if best_entry and best_score > 0.3:
                    unmatched_entries.remove(best_entry)

                    speaker_id, speaker_name = self._resolve_speaker(
                        best_entry.speaker, block.block_type, speaker_lookup
                    )

                    emotion, intensity = resolve_emotion_intensity(
                        best_entry.emotion, best_entry.emotion_intensity
                    )

                    assoc_ids = self._resolve_character_ids(
                        best_entry.associated_characters or [], speaker_lookup
                    )

                    raw_text = block.text
                    entry_raw = best_entry.raw_text or ""
                    for tag in CHATTERBOX_TAGS:
                        if tag in entry_raw:
                            raw_text = f"{tag} {raw_text}"
                            break

                    dialogue_inserts.append({
                        "entry_type": block.block_type,
                        "raw_text": raw_text,
                        "original_text": block.text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "emotion": emotion,
                        "emotion_intensity": intensity,
                        "associated_characters": assoc_ids,
                        "context_before": "\n".join(history[-2:]) if history else "",
                        "context_after": lookahead[:200] if lookahead else "",
                        "llm_confidence": best_entry.confidence or 0.5,
                        "sequence_number": local_seq,
                    })

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
                    speaker_id, speaker_name = self._resolve_speaker(
                        None, block.block_type, speaker_lookup
                    )

                    dialogue_inserts.append({
                        "entry_type": block.block_type,
                        "raw_text": block.text,
                        "original_text": block.text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker_name,
                        "emotion": "neutral",
                        "emotion_intensity": "low",
                        "associated_characters": [],
                        "context_before": "\n".join(history[-2:]) if history else "",
                        "context_after": lookahead[:200] if lookahead else "",
                        "llm_confidence": 0.0,
                        "sequence_number": local_seq,
                    })

                local_seq += 1

            # Store profile updates
            for update in result.profile_updates:
                resolved_update_char = self._resolve_character(update.character_name, speaker_lookup)
                char_id = resolved_update_char[0] if resolved_update_char else None
                if char_id:
                    profile_inserts.append({
                        "character_id": char_id,
                        "profile_type": "update",
                        "emotional_state": update.emotional_state,
                        "relationships": update.relationships,
                        "knowledge": update.knowledge,
                        "status": update.status,
                        "summary": update.summary,
                    })

            # Store events
            for evt_idx, event in enumerate(result.events):
                involved_ids = self._resolve_character_ids(
                    event.characters_involved, speaker_lookup
                )
                speaker_ids = self._resolve_character_ids(
                    event.speakers, speaker_lookup
                )
                event_inserts.append({
                    "sequence_number": local_seq + evt_idx,
                    "event_type": event.event_type,
                    "summary": event.summary,
                    "characters_involved": involved_ids,
                    "speakers": speaker_ids,
                    "importance": event.importance,
                    "spoiler_level": event.spoiler_level,
                })

            prev_chunk_text = chunk_text

        # ── Write all data for the section to DB in a single transaction ──
        with self._db.connection() as conn:
            with conn.transaction():
                for d in dialogue_inserts:
                    self._db.insert_dialogue_entry(
                        novel_meta_id,
                        section_index,
                        d["sequence_number"],
                        entry_type=d["entry_type"],
                        raw_text=d["raw_text"],
                        original_text=d["original_text"],
                        speaker_id=d["speaker_id"],
                        speaker_name=d["speaker_name"],
                        emotion=d["emotion"],
                        emotion_intensity=d["emotion_intensity"],
                        associated_characters=d["associated_characters"],
                        context_before=d["context_before"],
                        context_after=d["context_after"],
                        llm_confidence=d["llm_confidence"],
                        conn=conn,
                    )
                for p in profile_inserts:
                    self._db.insert_character_profile(
                        p["character_id"],
                        novel_meta_id,
                        section_index,
                        profile_type=p["profile_type"],
                        emotional_state=p["emotional_state"],
                        relationships=p["relationships"],
                        knowledge=p["knowledge"],
                        status=p["status"],
                        summary=p["summary"],
                        conn=conn,
                    )
                for e in event_inserts:
                    self._db.insert_event(
                        novel_meta_id,
                        section_index,
                        e["sequence_number"],
                        event_type=e["event_type"],
                        summary=e["summary"],
                        characters_involved=e["characters_involved"],
                        speakers=e["speakers"],
                        importance=e["importance"],
                        spoiler_level=e["spoiler_level"],
                        conn=conn,
                    )

        # ── Update progress safely ──
        with completed_sections_lock:
            self._completed_sections += 1
            self._db.upsert_parse_progress(
                novel_meta_id, pass_number=2,
                current_section=self._completed_sections, total_sections=total,
                status="running",
            )

    # ── LLM call ───────────────────────────────────────────────────────────

    def _analyse_chunk(
        self,
        char_prompt_data: str,
        history: list[str],
        chunk_text: str,
        lookahead: str,
        *,
        prev_chunk_text: str = "",
    ) -> ChunkAnalysisResult:
        """Send a single chunk to the LLM for analysis."""
        history_str = "\n".join(history[-self._history_size:]) if history else "(start of novel)"
        tags_str = ", ".join(CHATTERBOX_TAGS)

        user_msg = _ANALYSIS_USER.format(
            characters_json=char_prompt_data,
            prev_chunk_text=prev_chunk_text[:800] if prev_chunk_text else "(start of section)",
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
            max_tokens=4096,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _build_char_prompt_data(characters: list[dict[str, Any]]) -> str:
        """Build a rich character reference for prompts."""
        chars = []
        for c in characters:
            desc = c.get("description", "")
            if len(desc) > 200:
                desc = desc[:200] + "..."
            entry = {
                "name": c["name"],
                "aliases": c.get("aliases", []),
                "role": c.get("role", "minor"),
                "gender": c.get("gender", "unknown"),
                "age_range": c.get("age_range", "unknown"),
                "description": desc,
                "is_narrator": c.get("is_narrator", False),
            }
            chars.append(entry)
        return json.dumps(chars, indent=2)

    @staticmethod
    def _apply_narrator_flag(
        characters: list[dict[str, Any]],
        narrator_character_id: int | None,
    ) -> list[dict[str, Any]]:
        """Merge novels_meta narrator info into character rows used by Pass 2."""
        if narrator_character_id is None:
            return characters

        flagged: list[dict[str, Any]] = []
        for c in characters:
            row = dict(c)
            if row.get("id") == narrator_character_id:
                row["is_narrator"] = True
            flagged.append(row)
        return flagged

    @staticmethod
    def _build_speaker_lookup(characters: list[dict[str, Any]]) -> dict[str, tuple[int | None, str]]:
        """Build a name/alias to (character_id, canonical speaker name) lookup."""
        name_map: dict[str, tuple[int | None, str]] = {}

        def add(key: str, char_id: int | None, canonical: str) -> None:
            key = key.strip()
            if not key:
                return
            name_map[key] = (char_id, canonical)
            name_map[key.lower()] = (char_id, canonical)

        for c in characters:
            char_id = c["id"]
            canonical = c["name"]
            add(canonical, char_id, canonical)
            for alias in c.get("aliases", []):
                add(alias, char_id, canonical)

        # Always map NARRATOR to the narrator character when one exists.
        narrator_chars = [c for c in characters if c.get("is_narrator")]
        if narrator_chars:
            narrator = narrator_chars[0]
            add("NARRATOR", narrator["id"], narrator["name"])

        # Always map MISC_VOICE
        misc_chars = [c for c in characters if c.get("role") == "misc_voice"]
        if misc_chars:
            misc = misc_chars[0]
            add("MISC_VOICE", misc["id"], misc["name"])

        add("UNKNOWN", None, "UNKNOWN")
        return name_map

    @staticmethod
    def _resolve_speaker(
        speaker: str | None,
        block_type: str,
        speaker_lookup: dict[str, tuple[int | None, str]],
    ) -> tuple[int | None, str]:
        """Resolve an LLM speaker label to a canonical DB speaker."""
        fallback = "NARRATOR" if block_type in {"narration", "thought", "action"} else "UNKNOWN"
        label = (speaker or fallback).strip() or fallback
        return DialogueAnalyzer._resolve_character(label, speaker_lookup) or speaker_lookup["UNKNOWN"]

    @staticmethod
    def _resolve_character(
        label: str | None,
        speaker_lookup: dict[str, tuple[int | None, str]],
    ) -> tuple[int | None, str] | None:
        if not label:
            return None
        label = label.strip()
        return speaker_lookup.get(label) or speaker_lookup.get(label.lower())

    @staticmethod
    def _resolve_character_ids(
        labels: list[str],
        speaker_lookup: dict[str, tuple[int | None, str]],
    ) -> list[int]:
        ids: list[int] = []
        for label in labels:
            resolved = DialogueAnalyzer._resolve_character(label, speaker_lookup)
            if resolved and resolved[0] is not None and resolved[0] not in ids:
                ids.append(resolved[0])
        return ids

    @staticmethod
    def _is_deterministic_chunk(blocks: list[TextBlock]) -> bool:
        """Chunks with no dialogue do not need LLM speaker attribution."""
        return bool(blocks) and all(block.block_type != "dialogue" for block in blocks)

    @staticmethod
    def _format_chunk(blocks: list[TextBlock]) -> str:
        """Format blocks into readable text for the LLM prompt."""
        lines = []
        for block in blocks:
            if block.block_type == "dialogue":
                lines.append(f'"{block.text}"')
            elif block.block_type == "thought":
                lines.append(f"*{block.text}*")
            elif block.block_type == "action":
                lines.append(f"[{block.text}]")
            else:
                lines.append(block.text)
        return "\n\n".join(lines)
