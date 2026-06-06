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
)
from .voice_actors import VoiceActorAssigner

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Prompt templates
# ───────────────────────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are a literary analyst extracting characters from a novel, chapter by chapter.
You must respond ONLY with valid JSON matching the schema below.
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
1. Extract ALL named characters appearing in this chapter (new or previously seen).
2. For new characters: provide name, aliases (other names/titles/nicknames used),\
 gender (male/female/unknown), age_range (child/teen/young/young_adult/adult/middle_aged/elderly/unknown),\
 role_hint (protagonist/deuteragonist/antagonist/major/minor), description.
3. For existing characters from the "CHARACTERS FOUND SO FAR" list: note any new aliases,\
 updated description, or role changes. Only include characters that have new information.
4. Write a 2-3 sentence summary of this chapter's key events.

Respond ONLY with valid JSON:
{{
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
Respond ONLY with valid JSON.
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
   description (comprehensive summary across all chapters).
3. Determine the narrator type:
   - "character": if the novel is narrated in first person by a character in the story.\
 Provide that character's name.
   - "external": if the novel uses third-person or omniscient narration.
   Provide brief reasoning.

Respond ONLY with valid JSON:
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

        # Update progress
        self._db.upsert_parse_progress(
            novel_meta_id, pass_number=1,
            total_sections=total, status="running",
        )

        for idx, section in enumerate(sections):
            chapter_title = section.get("title", f"Section {idx + 1}")
            chapter_text = section.get("text", "")

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
            try:
                result = self._llm.chat_structured(
                    messages=[
                        {"role": "system", "content": _EXTRACTION_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                    response_model=ChapterExtractionResult,
                )
            except Exception as exc:
                logger.error("Pass 1 LLM error on section %d: %s", idx + 1, exc)
                chapter_summaries.append(f"Chapter {idx + 1}: [extraction failed]")
                continue

            # Merge new characters
            for char in result.new_characters:
                if not self._find_existing(accumulated_chars, char.name, char.aliases):
                    accumulated_chars.append(self._extracted_to_dict(char, section_idx=idx))

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

        # ── Consolidation pass ─────────────────────────────────────────────
        logger.info("Pass 1: running consolidation pass on %d characters", len(accumulated_chars))

        consolidated = self._consolidate(
            novel_title, accumulated_chars, chapter_summaries
        )

        # ── Store in PostgreSQL ────────────────────────────────────────────
        stored_chars = self._store_characters(
            novel_meta_id, consolidated, series_id
        )

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
        user_msg = _CONSOLIDATION_USER.format(
            novel_title=novel_title,
            all_characters_json=json.dumps(accumulated_chars, indent=2),
            chapter_summaries="\n".join(chapter_summaries),
        )

        try:
            return self._llm.chat_structured(
                messages=[
                    {"role": "system", "content": _CONSOLIDATION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                response_model=ConsolidationResult,
            )
        except Exception as exc:
            logger.error("Consolidation LLM error: %s — using raw characters", exc)
            # Fallback: wrap accumulated chars as-is
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
            return ConsolidationResult(
                characters=chars,
                narrator=NarratorDetection(),
            )

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

        # Store each character
        for char in result.characters:
            char_id = self._db.insert_character(
                novel_meta_id,
                name=char.name,
                aliases=char.aliases,
                gender=char.gender,
                age_range=char.age_range,
                role=char.role,
                description=char.description,
                series_id=series_id,
                is_narrator=(
                    narrator.narrator_type == "character"
                    and narrator.narrator_character_name
                    and char.name.lower() == narrator.narrator_character_name.lower()
                ),
            )

            # Track narrator character id
            if (
                narrator.narrator_type == "character"
                and narrator.narrator_character_name
                and char.name.lower() == narrator.narrator_character_name.lower()
            ):
                narrator_char_id = char_id

            # Create base profile
            self._db.insert_character_profile(
                char_id,
                novel_meta_id,
                section_index=0,
                profile_type="base",
                summary=char.description,
                status="active",
            )

        # If external narrator, create a NARRATOR character
        if narrator.narrator_type == "external":
            narrator_char_id = self._db.insert_character(
                novel_meta_id,
                name="NARRATOR",
                gender="unknown",
                role="narrator",
                description="External omniscient narrator",
                series_id=series_id,
                is_narrator=True,
            )

        # Update novels_meta with narrator info
        self._db.set_narrator_info(
            novel_meta_id,
            narrator_type=narrator.narrator_type,
            narrator_character_id=narrator_char_id,
        )

        return self._db.get_characters(novel_meta_id)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _char_to_prompt_dict(char: dict[str, Any]) -> dict[str, Any]:
        """Slim character dict for inclusion in prompts."""
        return {
            "name": char.get("name", ""),
            "aliases": char.get("aliases", []),
            "gender": char.get("gender", "unknown"),
            "age_range": char.get("age_range", "unknown"),
            "role_hint": char.get("role_hint", char.get("role", "minor")),
            "description": char.get("description", ""),
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
