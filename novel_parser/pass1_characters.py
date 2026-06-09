"""Pass 1 — Chapter-wise character extraction with rolling summarisation.

Processes the novel one chapter (section) at a time, maintaining:
- A rolling summary of previous chapters (for token budget)
- An accumulating character list
- Alias-aware deduplication

After all chapters, runs a consolidation pass to merge duplicates,
finalise roles, and detect narrator type.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .database import DatabaseManager
from .llm_client import LLMClient
from .models import (
    ChapterExtractionResult,
    ConsolidatedCharacter,
    ConsolidationResult,
    ExtractedCharacter,
    NarratorDetection,
    Pass1SectionExtraction,
)
from .text_splitter import is_front_matter
from .voice_actors import VoiceActorAssigner

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Prompt templates
# ───────────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are a literary analyst extracting characters from a novel, chapter by chapter.
You must respond ONLY with valid JSON matching the schema below. Do NOT output any preamble, markdown code blocks, XML tags, conversational intro/outro, <think>, <thought>, chain-of-thought, analysis, or reasoning notes. Start your response directly with the opening brace '{'.
"""

_EXTRACTION_USER = """\
STORY SO FAR (summary of previous chapters):
{rolling_summary}

CHARACTERS FOUND SO FAR:
{characters_json}

CURRENT CHAPTER ({chapter_index}/{total_chapters}) — "{chapter_title}":
---
{chapter_text}
---

Tasks:
0. Decide if this section is actual STORY content. Set "is_story_content": false if it is
 NOT part of the narrative — e.g. author's/translator's notes, foreword/afterword, preface,
 dedication, acknowledgements, copyright, table of contents, glossary, appendix, character
 profile lists, standalone "story so far"/recap pages, advertisements, merchandise/bonus
 pages, or previews of other books. Set "is_story_content": true for normal narrative
 chapters (including prologue/epilogue). If false, leave the character lists empty.
1. Extract ALL named characters appearing in this chapter (new or previously seen). Only extract actual characters participating in or present in the story. Do NOT extract historical or famous figures mentioned in passing, fictional characters from other stories referenced in dialogue, or names appearing solely in dedications, prefaces, introductions, or author notes which are not a part of the story.
2. For new characters: provide name, aliases (other names/titles/nicknames used),\
 gender (male/female/unknown), age_range (child/teen/young/young_adult/adult/middle_aged/elderly/unknown),\
 role_hint (protagonist/deuteragonist/antagonist/major/minor — note: protagonist, deuteragonist, and antagonist should only be assigned to the main 1-2 characters of the overall story; almost all other characters should be minor or major), description.
3. For existing characters from the "CHARACTERS FOUND SO FAR" list: note any new aliases,\
 updated description, or role changes. Only include characters that have new information. Be conservative with role changes.
4. Write a 2-3 sentence summary of this chapter's key events.

Respond ONLY with valid JSON. Do NOT think out loud, write markdown lists, or output reasoning before outputting the JSON. Do not include <think>, <thought>, markdown, notes, bullets, or analysis. Start your response immediately with the opening curly brace '{{' of the JSON:
{{
  "is_story_content": true,
  "new_characters": [
    {{"name": "...", "aliases": [...], "gender": "...", "age_range": "...",\
 "role_hint": "...", "description": "...", "first_seen_here": true}}
  ],
  "updated_characters": [
    {{"name": "...", "aliases": [...], "gender": "...", "age_range": "...",\
 "role_hint": "...", "description": "...", "first_seen_here": false}}
  ],
  "chapter_summary": "..."
}}
"""

_CONSOLIDATION_SYSTEM = """\
You are a literary analyst performing a final review of all characters extracted from a novel.
Your task is to merge duplicates, finalise roles, and detect the narrator type.
Respond ONLY with valid JSON. Do NOT output any preamble, markdown code blocks, XML tags, conversational intro/outro, <think>, <thought>, chain-of-thought, analysis, or reasoning notes. Start your response directly with the opening brace '{'.
"""

_CONSOLIDATION_USER = """\
Novel title: "{novel_title}"

All characters extracted (may contain duplicates or characters referred to by different names):
{all_characters_json}

Full chapter summaries:
{chapter_summaries}

Tasks:
1. Merge any duplicate characters (same person referred to by different names/titles/nicknames).
   Keep the most commonly used name as the primary name, others as aliases.
2. For each unique character, provide the final consolidated profile:
   name, aliases, gender, age_range, role (protagonist/deuteragonist/antagonist/major/minor).
   Guidelines for role:
   - protagonist: The single main character of the novel (usually exactly 1, e.g., Katniss Everdeen).
   - deuteragonist: The second most important character (usually only 1 or 2, e.g., Peeta Mellark).
   - antagonist: The main opponent of the protagonist (usually only 1 or 2, e.g., President Snow / Cato).
   - major: Important characters with significant dialogue or plot influence (e.g., Haymitch, Gale, Primrose, Rue, Effie).
   - minor: All other characters (background figures, guards, townspeople, etc. e.g., Madge, Octavia, Flavius). Almost all characters should be minor or major. Do NOT over-assign protagonist/deuteragonist/antagonist roles.
   description (comprehensive summary across all chapters).
   Do NOT create characters for non-person voices (PA systems, machines, broadcasts, signs, announcements).
3. Determine the narrator type based ONLY on the actual story chapters\
 (ignore dedications, prefaces, author notes, epigraphs, introductions):
   - "character": if the story is narrated in first person ("I did this", "I saw that").\
 The narrator IS one of the characters in your consolidated list — identify exactly which\
 one by name. Do NOT treat the narrator as a separate person from the character.
   - "external": if the story uses third-person or omniscient narration.
   Clue: If the chapter summaries consistently describe events from one character's\
 first-person perspective ("I", "me", "my"), that character is the first-person narrator.\
 The narrator_character_name must EXACTLY match one of the character names in your list.
   IMPORTANT: Occasional non-character voices (PA systems, machines, TV/radio broadcasts,\
 signs, descriptions) or non-story front and back matter do NOT make the narrator "external". Judge the narrator\
 purely from how the main story is told. These stray voices are handled separately.

Respond ONLY with valid JSON. Do NOT think out loud, write markdown lists, or output reasoning before outputting the JSON. Do not include <think>, <thought>, markdown, notes, bullets, or analysis. Start your response immediately with the opening curly brace '{{' of the JSON:
{{
  "characters": [
    {{"name": "...", "aliases": [...], "gender": "...", "age_range": "...",\
 "role": "...", "description": "..."}}
  ],
  "narrator": {{
    "narrator_type": "character" or "external",
    "narrator_character_name": "..." or null,
    "reasoning": "..."
  }}
}}
"""


class CharacterExtractor:
    """Orchestrates Pass 1: chapter-by-chapter character extraction."""

    def __init__(
        self,
        db: DatabaseManager,
        llm: LLMClient,
    ) -> None:
        self._db = db
        self._llm = llm

    def run(
        self,
        novel_meta_id: int,
        sections: list[dict[str, Any]],
        novel_title: str,
        *,
        prior_characters: list[dict[str, Any]] | None = None,
        series_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run Pass 1 on all sections.

        Args:
            novel_meta_id: The novel's PG id.
            sections: List of section dicts from SQLite (must have 'title', 'text').
            novel_title: For prompts.
            prior_characters: Characters from earlier books in the series.
            series_id: If part of a series.

        Returns:
            List of character dicts as stored in PostgreSQL.
        """
        total = len(sections)
        logger.info("Pass 1: starting character extraction for %d sections", total)

        # Initialise from prior series characters if available
        accumulated_chars = self._init_from_prior(prior_characters or [])
        rolling_summary = ""
        chapter_summaries: list[str] = []

        # Sections that are not part of the story (front/back matter, recaps,
        # author notes, glossaries, ads, etc.). Title keywords catch the obvious
        # cases cheaply; the LLM flags the long tail via is_story_content.
        non_story_sections: set[int] = {
            section.get("section_index", idx)
            for idx, section in enumerate(sections)
            if is_front_matter(section.get("title", ""))
        }

        # ── Check for saved intermediate results (crash-safe resume) ──────
        saved_extractions = self._db.get_pass1_extractions(novel_meta_id)
        saved_section_indices: set[int] = set()
        if saved_extractions:
            logger.info("Pass 1: found %d saved extractions — replaying", len(saved_extractions))
            for row in saved_extractions:
                sec_idx = row["section_index"]
                data = row["extraction_json"]
                saved_section_indices.add(sec_idx)

                # Honour a previously detected non-story section.
                if not data.get("is_story_content", True):
                    non_story_sections.add(sec_idx)
                    continue

                # Replay new characters
                for char_data in data.get("new_characters", []):
                    char = ExtractedCharacter.model_validate(char_data)
                    if not self._find_existing(accumulated_chars, char.name, char.aliases):
                        accumulated_chars.append(self._extracted_to_dict(char, section_idx=sec_idx))

                # Replay updated characters
                for char_data in data.get("updated_characters", []):
                    char = ExtractedCharacter.model_validate(char_data)
                    existing = self._find_existing(accumulated_chars, char.name, char.aliases)
                    if existing:
                        self._merge_update(existing, char)

                # Replay summary
                summary = data.get("chapter_summary", "")
                if summary:
                    chapter_summaries.append(f"Chapter {sec_idx + 1}: {summary}")

            rolling_summary = "\n".join(chapter_summaries[-5:])

        # Update progress
        self._db.upsert_parse_progress(
            novel_meta_id, pass_number=1,
            total_sections=total, status="running",
        )

        # Track extraction outcomes to distinguish a genuine failure (LLM/JSON
        # errors) from legitimately character-free content (self-help, PDFs).
        attempted = 0
        failed = 0

        for idx, section in enumerate(sections):
            section_index = section.get("section_index", idx)
            chapter_title = section.get("title", f"Section {idx + 1}")
            chapter_text = section.get("text", "")

            # Skip already-processed sections
            if section_index in saved_section_indices:
                logger.info("Pass 1: skipping section %d (already saved)", section_index)
                self._db.upsert_parse_progress(
                    novel_meta_id, pass_number=1,
                    current_section=idx + 1, total_sections=total,
                    status="running",
                )
                continue

            # Front matter (preface, author's note, etc.) has no characters and
            # must not skew narrator detection. It is still stored and voiced in
            # Pass 2 — here we only skip character extraction.
            if is_front_matter(chapter_title):
                logger.info("Pass 1: skipping front matter section %d — '%s'", idx + 1, chapter_title)
                self._db.upsert_parse_progress(
                    novel_meta_id, pass_number=1,
                    current_section=idx + 1, total_sections=total,
                    status="running",
                )
                continue

            if not chapter_text.strip():
                chapter_summaries.append(f"Chapter {idx + 1}: [empty]")
                self._db.upsert_parse_progress(
                    novel_meta_id, pass_number=1,
                    current_section=idx + 1, total_sections=total,
                    status="running",
                )
                continue

            logger.info("Pass 1: processing section %d/%d — '%s'", idx + 1, total, chapter_title)

            # Build prompt
            chars_json = json.dumps(
                [self._char_to_prompt_dict(c) for c in accumulated_chars],
                indent=2,
            )
            user_msg = _EXTRACTION_USER.format(
                rolling_summary=rolling_summary or "(This is the first chapter)",
                characters_json=chars_json,
                chapter_index=idx + 1,
                total_chapters=total,
                chapter_title=chapter_title,
                chapter_text=chapter_text,
            )

            # Call LLM
            attempted += 1
            try:
                result = self._llm.chat_structured(
                    messages=[
                        {"role": "system", "content": _EXTRACTION_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    response_model=ChapterExtractionResult,
                )
            except Exception as exc:
                failed += 1
                logger.error("Pass 1 LLM error on section %d: %s", idx + 1, exc)
                chapter_summaries.append(f"Chapter {idx + 1}: [extraction failed]")
                continue

            # ── Persist extraction immediately (crash-safe) ───────────────
            self._db.save_pass1_extraction(
                novel_meta_id, section_index, result.model_dump()
            )

            # Non-story section (author note, recap, glossary, ad, etc.):
            # record it and skip character/summary accumulation so it cannot
            # pollute the character list or narrator detection.
            if not result.is_story_content:
                logger.info("Pass 1: section %d flagged as non-story content", idx + 1)
                non_story_sections.add(section_index)
                self._db.upsert_parse_progress(
                    novel_meta_id, pass_number=1,
                    current_section=idx + 1, total_sections=total,
                    status="running",
                )
                continue

            # Merge new characters
            for char in result.new_characters:
                if not self._find_existing(accumulated_chars, char.name, char.aliases):
                    accumulated_chars.append(self._extracted_to_dict(char, section_idx=section_index))

            # Update existing characters
            for char in result.updated_characters:
                existing = self._find_existing(accumulated_chars, char.name, char.aliases)
                if existing:
                    self._merge_update(existing, char)

            # Update rolling summary
            if result.chapter_summary:
                chapter_summaries.append(
                    f"Chapter {idx + 1} ({chapter_title}): {result.chapter_summary}"
                )
                # Keep rolling summary manageable — last 5 chapter summaries
                rolling_summary = "\n".join(chapter_summaries[-5:])

            # Update progress
            self._db.upsert_parse_progress(
                novel_meta_id, pass_number=1,
                current_section=idx + 1, total_sections=total,
                status="running",
            )

        # ── Fail fast on genuine extraction failure ────────────────────────
        # 0 characters is valid (narrator-only content). The definite error
        # signal is a high LLM/JSON failure rate, not the character count.
        if attempted and failed / attempted > 0.3:
            raise RuntimeError(
                f"Pass 1 extraction failed for {failed}/{attempted} sections — "
                "aborting so the parse can be retried"
            )

        # ── Consolidation pass ─────────────────────────────────────────────
        logger.info("Pass 1: running consolidation pass on %d characters", len(accumulated_chars))

        consolidated = self._consolidate(
            novel_title, accumulated_chars, chapter_summaries
        )

        # ── Store in PostgreSQL ────────────────────────────────────────────
        stored_chars = self._store_characters(
            novel_meta_id, consolidated, series_id
        )

        # ── Persist non-story sections for Pass 2 routing ──────────────────
        self._db.set_non_story_sections(novel_meta_id, sorted(non_story_sections))

        # ── Clean up temporary extraction data ─────────────────────────────
        self._db.delete_pass1_extractions(novel_meta_id)

        # Mark complete
        self._db.mark_progress_complete(novel_meta_id, pass_number=1)
        logger.info("Pass 1 complete: %d characters stored", len(stored_chars))

        return stored_chars

    # ── consolidation ──────────────────────────────────────────────────────

    def _consolidate(
        self,
        novel_title: str,
        accumulated_chars: list[dict[str, Any]],
        chapter_summaries: list[str],
    ) -> ConsolidationResult:
        """Run the final consolidation LLM call."""
        # Slim down descriptions in accumulated_chars to prevent context/output truncation
        slim_chars = []
        for c in accumulated_chars:
            desc = c.get("description", "")
            if len(desc) > 300:
                desc = desc[:300] + "..."
            slim_chars.append({
                "name": c.get("name", ""),
                "aliases": c.get("aliases", []),
                "gender": c.get("gender", "unknown"),
                "age_range": c.get("age_range", "unknown"),
                "role_hint": c.get("role_hint", "minor"),
                "description": desc,
            })

        # Truncate each chapter summary to keep the overall payload context compact
        short_summaries = []
        for s in chapter_summaries:
            if len(s) > 150:
                short_summaries.append(s[:150] + "...")
            else:
                short_summaries.append(s)

        user_msg = _CONSOLIDATION_USER.format(
            novel_title=novel_title,
            all_characters_json=json.dumps(slim_chars, indent=2),
            chapter_summaries="\n".join(short_summaries),
        )

        def _fallback() -> ConsolidationResult:
            # Wrap accumulated chars as-is (LLM unavailable or dropped them).
            chars = [
                ConsolidatedCharacter(
                    name=c.get("name", "Unknown"),
                    aliases=c.get("aliases", []),
                    gender=c.get("gender", "unknown"),
                    age_range=c.get("age_range", "unknown"),
                    role=c.get("role_hint", "minor"),
                    description=c.get("description", ""),
                )
                for c in accumulated_chars
            ]
            return ConsolidationResult(characters=chars, narrator=NarratorDetection())

        try:
            # Consolidation handles a large character list + summaries, so we allow a generous timeout
            result = self._llm.chat_structured(
                messages=[
                    {"role": "system", "content": _CONSOLIDATION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_model=ConsolidationResult,
                timeout=300.0,
            )
        except Exception as exc:
            logger.error("Consolidation LLM error: %s — using raw characters", exc)
            return _fallback()

        # If consolidation dropped every character but we had some, keep ours.
        # (0 accumulated characters is left as-is: valid narrator-only content.)
        if not result.characters and accumulated_chars:
            logger.warning(
                "Consolidation returned no characters despite %d accumulated — using raw characters",
                len(accumulated_chars),
            )
            return _fallback()

        return result

    # ── storage ────────────────────────────────────────────────────────────

    def _store_characters(
        self,
        novel_meta_id: int,
        result: ConsolidationResult,
        series_id: int | None,
    ) -> list[dict[str, Any]]:
        """Store consolidated characters and narrator info in PostgreSQL."""
        # Store narrator info
        narrator = result.narrator
        narrator_char_id: int | None = None

        with self._db.connection() as conn:
            with conn.transaction():
                # Store each character
                for char in result.characters:
                    is_narrator = False
                    if narrator.narrator_type == "character" and narrator.narrator_character_name:
                        if self._narrator_matches(
                            narrator.narrator_character_name, char.name, char.aliases
                        ):
                            is_narrator = True

                    char_id = self._db.insert_character(
                        novel_meta_id,
                        name=char.name,
                        aliases=char.aliases,
                        gender=char.gender,
                        age_range=char.age_range,
                        role=char.role,
                        description=char.description,
                        series_id=series_id,
                        is_narrator=is_narrator,
                        conn=conn,
                    )

                    # Track narrator character id
                    if is_narrator:
                        narrator_char_id = char_id

                    # Create base profile
                    self._db.insert_character_profile(
                        char_id,
                        novel_meta_id,
                        section_index=0,
                        profile_type="base",
                        summary=char.description,
                        status="active",
                        conn=conn,
                    )

                # Decide the effective narrator. A first-person ("character")
                # narrator that we could not map to any character falls back to
                # an external NARRATOR so narration is always voiced.
                narrator_type = narrator.narrator_type
                if narrator_type == "character" and narrator_char_id is None:
                    logger.warning(
                        "Narrator '%s' not matched to any character — using external NARRATOR",
                        narrator.narrator_character_name,
                    )
                    narrator_type = "external"

                if narrator_type == "external":
                    narrator_char_id = self._db.insert_character(
                        novel_meta_id,
                        name="NARRATOR",
                        gender="unknown",
                        role="narrator",
                        description="External omniscient narrator",
                        series_id=series_id,
                        is_narrator=True,
                        conn=conn,
                    )

                # Update novels_meta with narrator info
                self._db.set_narrator_info(
                    novel_meta_id,
                    narrator_type=narrator_type,
                    narrator_character_id=narrator_char_id,
                    conn=conn,
                )

                # Always create a MISC_VOICE character for non-person speakers
                # (PA systems, machines, broadcasts, signs, crowd chants, etc.)
                self._db.insert_character(
                    novel_meta_id,
                    name="MISC_VOICE",
                    gender="unknown",
                    role="misc_voice",
                    description="Miscellaneous non-character voices (PA systems, machines, broadcasts, signs, crowd chants)",
                    series_id=series_id,
                    is_narrator=False,
                    conn=conn,
                )

        return self._db.get_characters(novel_meta_id)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _char_to_prompt_dict(char: dict[str, Any]) -> dict[str, Any]:
        """Slim character dict for inclusion in prompts."""
        desc = char.get("description", "")
        if len(desc) > 200:
            desc = desc[:200] + "..."
        return {
            "name": char.get("name", ""),
            "aliases": char.get("aliases", []),
            "gender": char.get("gender", "unknown"),
            "age_range": char.get("age_range", "unknown"),
            "role_hint": char.get("role_hint", char.get("role", "minor")),
            "description": desc,
        }

    @staticmethod
    def _extracted_to_dict(
        char: ExtractedCharacter, section_idx: int
    ) -> dict[str, Any]:
        return {
            "name": char.name,
            "aliases": char.aliases,
            "gender": char.gender,
            "age_range": char.age_range,
            "role_hint": char.role_hint,
            "description": char.description,
            "first_appearance_section": section_idx,
        }

    @staticmethod
    def _narrator_matches(
        narrator_name: str, char_name: str, char_aliases: list[str]
    ) -> bool:
        """Match the LLM's narrator name to a character.

        Uses token-aware matching so a first-person narrator named "Katniss"
        maps to the character "Katniss Everdeen", while avoiding false positives
        between merely similar names (e.g. "Kate" vs "Katherine").
        """
        norm_narrator = narrator_name.lower().strip()
        candidates = {char_name.lower().strip()} | {a.lower().strip() for a in char_aliases}
        if norm_narrator in candidates:
            return True
        narrator_tokens = set(norm_narrator.split())
        if not narrator_tokens:
            return False
        for cand in candidates:
            cand_tokens = set(cand.split())
            if cand_tokens and (narrator_tokens <= cand_tokens or cand_tokens <= narrator_tokens):
                return True
        return False

    @staticmethod
    def _find_existing(
        chars: list[dict[str, Any]], name: str, aliases: list[str]
    ) -> Optional[dict[str, Any]]:
        """Find an existing character by name or alias match."""
        name_lower = name.lower()
        all_names = {name_lower} | {a.lower() for a in aliases}

        for char in chars:
            char_names = {char["name"].lower()} | {
                a.lower() for a in char.get("aliases", [])
            }
            if char_names & all_names:
                return char
        return None

    @staticmethod
    def _merge_update(
        existing: dict[str, Any], update: ExtractedCharacter
    ) -> None:
        """Merge updated character info into an existing character dict."""
        # Merge aliases
        existing_aliases = set(existing.get("aliases", []))
        existing_aliases.update(update.aliases)
        # Don't add the primary name as alias
        existing_aliases.discard(existing["name"])
        existing["aliases"] = list(existing_aliases)

        # Update fields if the update has more specific info
        if update.gender != "unknown" and existing.get("gender") == "unknown":
            existing["gender"] = update.gender
        if update.age_range != "unknown" and existing.get("age_range") == "unknown":
            existing["age_range"] = update.age_range
        if update.description:
            desc_lower = existing.get("description", "").lower()
            if update.description.lower() not in desc_lower:
                existing["description"] = (
                    existing.get("description", "") + " " + update.description
                ).strip()
        # Upgrade role if more important
        role_priority = {
            "protagonist": 0, "deuteragonist": 1, "antagonist": 2,
            "major": 3, "minor": 4,
        }
        if role_priority.get(update.role_hint, 4) < role_priority.get(
            existing.get("role_hint", "minor"), 4
        ):
            existing["role_hint"] = update.role_hint

    @staticmethod
    def _init_from_prior(
        prior_characters: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert prior series characters into the accumulator format."""
        return [
            {
                "name": c.get("name", ""),
                "aliases": c.get("aliases", []),
                "gender": c.get("gender", "unknown"),
                "age_range": c.get("age_range", "unknown"),
                "role_hint": c.get("role", "minor"),
                "description": c.get("description", ""),
                "first_appearance_section": c.get("first_appearance_section"),
                "_from_prior_book": True,
            }
            for c in prior_characters
        ]
