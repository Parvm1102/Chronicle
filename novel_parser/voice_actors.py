"""Voice actor registry — metadata, sample discovery, and character-to-actor matching.

On first run the hardcoded actor metadata is seeded into PostgreSQL.
The emotion/intensity map is auto-discovered by scanning the voice_samples directory.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from .config import Settings, get_settings
from .database import DatabaseManager

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────
# Hardcoded voice actor metadata (from user-provided descriptions)
# ───────────────────────────────────────────────────────────────────────────

_ACTOR_METADATA: list[dict[str, Any]] = [
    {
        "name": "Vivian",
        "gender": "female",
        "age_range": "young_adult",
        "tone_tags": ["bright", "edgy"],
        "default_emotion": "neutral",
        "notes": "Good protagonist. Wide emotional range.",
        "protagonist_suited": True,
    },
    {
        "name": "Serena",
        "gender": "female",
        "age_range": "young",
        "tone_tags": ["warm", "gentle"],
        "default_emotion": "serious_low",
        "notes": "Not preferred protagonist. Prefer serious over neutral.",
        "protagonist_suited": False,
    },
    {
        "name": "Sohee",
        "gender": "female",
        "age_range": "adult",
        "tone_tags": ["warm", "clear", "rich"],
        "default_emotion": "neutral",
        "notes": "Good adult protagonist. Clear voice.",
        "protagonist_suited": True,
    },
    {
        "name": "Ono_Anna",
        "gender": "female",
        "age_range": "teen",
        "tone_tags": ["playful", "light", "nimble"],
        "default_emotion": "neutral",
        "notes": "Japanese accent. Good teen protagonist. Quieter.",
        "protagonist_suited": True,
    },
    {
        "name": "Dylan",
        "gender": "male",
        "age_range": "young",
        "tone_tags": ["youthful", "clear", "gloomy"],
        "default_emotion": "neutral",
        "notes": "Gloomy voice. Use happy_high+ and higher intensities except sad. Protagonist/deuteragonist.",
        "protagonist_suited": True,
    },
    {
        "name": "Ryan",
        "gender": "male",
        "age_range": "adult",
        "tone_tags": ["dynamic", "rhythmic"],
        "default_emotion": "serious_low",
        "notes": "Older protagonist. Strong sad range.",
        "protagonist_suited": True,
    },
    {
        "name": "Aiden",
        "gender": "male",
        "age_range": "young_adult",
        "tone_tags": ["sunny", "clear"],
        "default_emotion": "neutral",
        "notes": "Protagonist/deuteragonist. Less emotional range than Ryan.",
        "protagonist_suited": True,
    },
    {
        "name": "Uncle_Fu",
        "gender": "male",
        "age_range": "middle_aged",
        "tone_tags": ["seasoned", "low", "mellow"],
        "default_emotion": "neutral",
        "notes": "Very high emotional range.",
        "protagonist_suited": False,
    },
    {
        "name": "Eric",
        "gender": "male",
        "age_range": "young",
        "tone_tags": ["lively", "husky", "bright"],
        "default_emotion": "neutral",
        "notes": "Chinese/Asian dialect. Suited for Asian characters.",
        "protagonist_suited": False,
    },
]

# File naming convention: {ActorName}_{emotion}_{intensity}.wav
# e.g. Aiden_happy_high.wav, Vivian_neutral.wav (neutral has no intensity suffix)
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_]+?)_(?P<emotion>[a-z]+)(?:_(?P<intensity>low|med|high))?\.wav$",
    re.IGNORECASE,
)


class VoiceActorManager:
    """Seed, discover, and query voice actors."""

    def __init__(
        self,
        db: DatabaseManager,
        settings: Optional[Settings] = None,
    ) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._samples_dir = self._settings.voice_samples_dir

    def seed(self) -> None:
        """Populate voice_actors table with metadata + auto-discovered emotions."""
        for meta in _ACTOR_METADATA:
            emotions = self._discover_emotions(meta["name"])
            self._db.upsert_voice_actor(
                meta["name"],
                gender=meta["gender"],
                age_range=meta["age_range"],
                tone_tags=meta["tone_tags"],
                default_emotion=meta["default_emotion"],
                notes=meta["notes"],
                protagonist_suited=meta["protagonist_suited"],
                sample_dir=str(self._samples_dir / meta["name"]),
                emotions=emotions,
            )
            logger.info(
                "Seeded voice actor %s — %d emotion categories",
                meta["name"], len(emotions),
            )

    def _discover_emotions(self, actor_name: str) -> dict[str, list[str]]:
        """Scan the actor's directory and build an emotion→intensities map.

        Returns e.g.::

            {
                "neutral": [],           # single file, no intensity suffix
                "happy": ["low", "med", "high"],
                "mysterious": ["low"],    # only low exists
                "warm": ["low", "high"],  # no med
                ...
            }
        """
        actor_dir = self._samples_dir / actor_name
        emotions: dict[str, list[str]] = {}

        if not actor_dir.is_dir():
            logger.warning("Voice samples directory not found: %s", actor_dir)
            return emotions

        for wav in sorted(actor_dir.glob("*.wav")):
            m = _SAMPLE_RE.match(wav.name)
            if not m:
                continue
            emotion = m.group("emotion").lower()
            intensity = m.group("intensity")  # None if no suffix
            if intensity:
                emotions.setdefault(emotion, []).append(intensity.lower())
            else:
                # e.g. "Aiden_neutral.wav" — no intensity suffix
                emotions.setdefault(emotion, [])

        return emotions

    # Ordered by proximity so fallback picks the closest available intensity
    _INTENSITY_FALLBACK: dict[str, list[str]] = {
        "low":  ["low", "med", "high"],
        "med":  ["med", "low", "high"],  # prefer lower when med missing
        "high": ["high", "med", "low"],
    }

    def get_sample_path(
        self,
        actor_name: str,
        emotion: str = "neutral",
        intensity: str = "low",
    ) -> Optional[Path]:
        """Resolve the .wav path for a given actor/emotion/intensity.

        Fallback chain:
        1. Exact match (actor_emotion_intensity.wav)
        2. Nearest available intensity for the same emotion
        3. Actor's neutral sample
        """
        actor_dir = self._samples_dir / actor_name

        # Emotions without intensity suffix (e.g. neutral)
        if emotion == "neutral":
            exact = actor_dir / f"{actor_name}_neutral.wav"
            return exact if exact.exists() else None

        # Try exact match first
        exact = actor_dir / f"{actor_name}_{emotion}_{intensity}.wav"
        if exact.exists():
            return exact

        # Fallback: try closest intensity for the same emotion
        for fb_intensity in self._INTENSITY_FALLBACK.get(intensity, ["low", "med", "high"]):
            if fb_intensity == intensity:
                continue  # already tried
            fb = actor_dir / f"{actor_name}_{emotion}_{fb_intensity}.wav"
            if fb.exists():
                logger.debug(
                    "Fallback: %s_%s_%s → %s_%s_%s",
                    actor_name, emotion, intensity,
                    actor_name, emotion, fb_intensity,
                )
                return fb

        # Last resort: neutral
        neutral = actor_dir / f"{actor_name}_neutral.wav"
        if neutral.exists():
            logger.debug(
                "Fallback to neutral: %s_%s_%s → neutral",
                actor_name, emotion, intensity,
            )
            return neutral
        return None


class VoiceActorAssigner:
    """Match characters to voice actors based on traits."""

    # Age proximity scores (lower = closer match)
    _AGE_DISTANCE: dict[tuple[str, str], int] = {}

    _AGE_ORDER = ["child", "teen", "young", "young_adult", "adult", "middle_aged", "elderly"]

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        # Pre-compute age distance table
        for i, a in enumerate(self._AGE_ORDER):
            for j, b in enumerate(self._AGE_ORDER):
                self._AGE_DISTANCE[(a, b)] = abs(i - j)

    def assign_all(
        self,
        novel_meta_id: int,
        characters: list[dict[str, Any]],
    ) -> dict[int, int]:
        """Assign voice actors to characters.

        Returns mapping {character_id: voice_actor_id}.
        Major characters get unique actors; minor characters can share.
        """
        actors = self._db.get_all_voice_actors()
        if not actors:
            logger.warning("No voice actors in database — skipping assignment")
            return {}

        # Sort characters: protagonist/deuteragonist/major first
        priority_order = {
            "protagonist": 0, "deuteragonist": 1, "antagonist": 2,
            "major": 3, "narrator": 4, "minor": 5,
        }
        sorted_chars = sorted(
            characters,
            key=lambda c: priority_order.get(c.get("role", "minor"), 5),
        )

        used_actor_ids: set[int] = set()
        assignments: dict[int, int] = {}

        for char in sorted_chars:
            best_actor = self._find_best_actor(
                char, actors, used_actor_ids,
                reserve_for_major=char.get("role", "minor") in (
                    "protagonist", "deuteragonist", "antagonist", "major", "narrator"
                ),
            )
            if best_actor:
                assignments[char["id"]] = best_actor["id"]
                # Reserve actor for major characters only
                if char.get("role", "minor") in (
                    "protagonist", "deuteragonist", "antagonist", "major", "narrator"
                ):
                    used_actor_ids.add(best_actor["id"])

        # Write assignments to DB
        with self._db.connection() as conn:
            for char_id, actor_id in assignments.items():
                conn.execute(
                    "UPDATE characters SET voice_actor_id = %s, updated_at = NOW() WHERE id = %s",
                    (actor_id, char_id),
                )

        return assignments

    def _find_best_actor(
        self,
        character: dict[str, Any],
        actors: list[dict[str, Any]],
        used_ids: set[int],
        reserve_for_major: bool,
    ) -> Optional[dict[str, Any]]:
        """Score each actor and return the best match."""
        char_gender = character.get("gender", "unknown")
        char_age = character.get("age_range", "unknown")
        char_role = character.get("role", "minor")

        scored: list[tuple[float, dict[str, Any]]] = []

        for actor in actors:
            # Hard filter: gender must match (unless unknown)
            if char_gender != "unknown" and actor["gender"] != char_gender:
                continue

            score = 0.0

            # Age proximity (max 3 points)
            age_dist = self._AGE_DISTANCE.get(
                (char_age, actor["age_range"]), 3
            )
            score += max(0, 3 - age_dist)

            # Protagonist suitability (2 points)
            if char_role in ("protagonist", "deuteragonist") and actor.get("protagonist_suited"):
                score += 2.0

            # Prefer unused actors for major characters (1 point)
            if reserve_for_major and actor["id"] not in used_ids:
                score += 1.0

            scored.append((score, actor))

        if not scored:
            # No gender match — return any unused actor
            for actor in actors:
                if actor["id"] not in used_ids:
                    return actor
            return actors[0] if actors else None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
