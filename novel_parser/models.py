"""Pydantic v2 models — enums, LLM I/O schemas, and DB row helpers.

These are the canonical type definitions shared across the entire
novel_parser package.  Every LLM prompt and every DB write goes through
one of these models so that we never deal with untyped dicts.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ───────────────────────────────────────────────────────────────────────────
# Constrained enums — match voice-sample file naming exactly
# ───────────────────────────────────────────────────────────────────────────

class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    WARM = "warm"
    HAPPY = "happy"
    ANGRY = "angry"
    SAD = "sad"
    FEARFUL = "fearful"
    MYSTERIOUS = "mysterious"
    SERIOUS = "serious"
    WHISPER = "whisper"


class EmotionIntensity(str, Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


# ── Valid combinations (matches actual voice sample files) ────────────────
# neutral has no intensity suffix at all (single file per actor).
# All other emotions have a subset of low/med/high.

VALID_EMOTION_INTENSITIES: dict[EmotionType, list[EmotionIntensity]] = {
    EmotionType.NEUTRAL:    [],                                                  # single file, no intensity
    EmotionType.HAPPY:      [EmotionIntensity.LOW, EmotionIntensity.MED, EmotionIntensity.HIGH],
    EmotionType.ANGRY:      [EmotionIntensity.LOW, EmotionIntensity.MED, EmotionIntensity.HIGH],
    EmotionType.SAD:        [EmotionIntensity.LOW, EmotionIntensity.MED, EmotionIntensity.HIGH],
    EmotionType.FEARFUL:    [EmotionIntensity.LOW, EmotionIntensity.MED, EmotionIntensity.HIGH],
    EmotionType.WARM:       [EmotionIntensity.LOW, EmotionIntensity.HIGH],        # no med
    EmotionType.SERIOUS:    [EmotionIntensity.LOW, EmotionIntensity.HIGH],        # no med
    EmotionType.WHISPER:    [EmotionIntensity.LOW, EmotionIntensity.HIGH],        # no med
    EmotionType.MYSTERIOUS: [EmotionIntensity.LOW],                               # only low
}

# Human-readable summary for LLM prompts
EMOTION_INTENSITY_PROMPT_MAP: str = """Available emotion + intensity combinations:
- neutral (no intensity — use alone)
- happy: low, med, high
- angry: low, med, high
- sad: low, med, high
- fearful: low, med, high
- warm: low, high (NO med)
- serious: low, high (NO med)
- whisper: low, high (NO med)
- mysterious: low ONLY (NO med or high)"""


def resolve_emotion_intensity(
    emotion: Optional[str], intensity: Optional[str]
) -> tuple[str, str]:
    """Snap an emotion/intensity pair to the nearest valid combination.

    Returns (emotion, intensity) — guaranteed to match an actual voice sample.
    """
    if not emotion:
        emotion = "neutral"
    if not intensity:
        intensity = "low"

    # Validate emotion
    try:
        emo = EmotionType(emotion.lower())
    except (ValueError, AttributeError):
        return EmotionType.NEUTRAL.value, EmotionIntensity.LOW.value

    valid = VALID_EMOTION_INTENSITIES[emo]

    # neutral has no intensity
    if not valid:
        return emo.value, EmotionIntensity.LOW.value

    # Check if requested intensity is valid
    try:
        inten = EmotionIntensity(intensity.lower())
    except (ValueError, AttributeError):
        return emo.value, valid[0].value

    if inten in valid:
        return emo.value, inten.value

    # Snap to nearest: prefer lower intensity as fallback
    _PROXIMITY = {
        EmotionIntensity.LOW:  [EmotionIntensity.LOW, EmotionIntensity.MED, EmotionIntensity.HIGH],
        EmotionIntensity.MED:  [EmotionIntensity.MED, EmotionIntensity.LOW, EmotionIntensity.HIGH],
        EmotionIntensity.HIGH: [EmotionIntensity.HIGH, EmotionIntensity.MED, EmotionIntensity.LOW],
    }
    for candidate in _PROXIMITY.get(inten, list(EmotionIntensity)):
        if candidate in valid:
            return emo.value, candidate.value

    return emo.value, valid[0].value


class EntryType(str, Enum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    THOUGHT = "thought"
    ACTION = "action"


class CharacterRole(str, Enum):
    PROTAGONIST = "protagonist"
    ANTAGONIST = "antagonist"
    DEUTERAGONIST = "deuteragonist"
    MAJOR = "major"
    MINOR = "minor"
    NARRATOR = "narrator"
    MISC_VOICE = "misc_voice"


class EventType(str, Enum):
    PLOT = "plot"
    REVEAL = "reveal"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    TRANSITION = "transition"


class EventImportance(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


# ───────────────────────────────────────────────────────────────────────────
# Pass 1 — Character extraction LLM I/O
# ───────────────────────────────────────────────────────────────────────────

class ExtractedCharacter(BaseModel):
    """A single character found by the LLM in one chapter."""
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: str = "unknown"
    age_range: str = "unknown"
    role_hint: str = "minor"
    description: str = ""
    first_seen_here: bool = False


class ChapterExtractionResult(BaseModel):
    """LLM output for a single chapter in Pass 1."""
    is_story_content: bool = True  # False for author notes, afterwords, recaps, glossaries, ads, etc.
    new_characters: list[ExtractedCharacter] = Field(default_factory=list)
    updated_characters: list[ExtractedCharacter] = Field(default_factory=list)
    chapter_summary: str = ""


class Pass1SectionExtraction(BaseModel):
    """One chapter's raw extraction result, persisted for crash-safe resumability."""
    section_index: int
    is_story_content: bool = True
    new_characters: list[ExtractedCharacter] = Field(default_factory=list)
    updated_characters: list[ExtractedCharacter] = Field(default_factory=list)
    chapter_summary: str = ""


class ConsolidatedCharacter(BaseModel):
    """Final merged character after the consolidation pass."""
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: str = "unknown"
    age_range: str = "unknown"
    role: str = "minor"
    description: str = ""


class NarratorDetection(BaseModel):
    """LLM output for narrator type detection."""
    narrator_type: str = "external"  # "character" or "external"
    narrator_character_name: Optional[str] = None  # if narrator is a character
    reasoning: str = ""


class ConsolidationResult(BaseModel):
    """LLM output for the character consolidation pass."""
    characters: list[ConsolidatedCharacter] = Field(default_factory=list)
    narrator: NarratorDetection = Field(default_factory=NarratorDetection)


# ───────────────────────────────────────────────────────────────────────────
# Pass 2 — Dialogue analysis LLM I/O
# ───────────────────────────────────────────────────────────────────────────

class DialogueEntryResult(BaseModel):
    """A single analysed text block from Pass 2."""
    entry_type: Optional[str] = "dialogue"
    speaker: Optional[str] = "NARRATOR"
    emotion: Optional[str] = "neutral"
    emotion_intensity: Optional[str] = "low"
    raw_text: Optional[str] = ""           # with chatterbox tags injected
    original_text: Optional[str] = ""      # source text before tags
    associated_characters: list[str] = Field(default_factory=list)
    confidence: Optional[float] = 0.0


class ProfileUpdateResult(BaseModel):
    """Profile delta for one character from a chunk."""
    character_name: str
    emotional_state: Optional[str] = ""
    relationships: dict[str, str] = Field(default_factory=dict)
    knowledge: list[str] = Field(default_factory=list)
    status: Optional[str] = ""
    summary: Optional[str] = ""


class EventResult(BaseModel):
    """A plot event detected in a chunk."""
    event_type: Optional[str] = "plot"
    summary: Optional[str] = ""
    characters_involved: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    importance: Optional[str] = "minor"
    spoiler_level: Optional[int] = 0


class ChunkAnalysisResult(BaseModel):
    """Full LLM output for a single paragraph chunk in Pass 2."""
    entries: list[DialogueEntryResult] = Field(default_factory=list)
    profile_updates: list[ProfileUpdateResult] = Field(default_factory=list)
    events: list[EventResult] = Field(default_factory=list)


# ───────────────────────────────────────────────────────────────────────────
# Voice actor metadata
# ───────────────────────────────────────────────────────────────────────────

class VoiceActorInfo(BaseModel):
    """Metadata about a single voice actor."""
    name: str
    gender: str
    age_range: str
    tone_tags: list[str] = Field(default_factory=list)
    default_emotion: str = "neutral"
    notes: str = ""
    protagonist_suited: bool = False
    sample_dir: str = ""
    emotions: dict[str, list[str]] = Field(default_factory=dict)
    # e.g. {"neutral": ["low"], "happy": ["low", "med", "high"], ...}
