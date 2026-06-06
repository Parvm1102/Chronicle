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
    entry_type: str = "dialogue"
    speaker: str = "NARRATOR"
    emotion: str = "neutral"
    emotion_intensity: str = "low"
    raw_text: str = ""           # with chatterbox tags injected
    original_text: str = ""      # source text before tags
    associated_characters: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class ProfileUpdateResult(BaseModel):
    """Profile delta for one character from a chunk."""
    character_name: str
    emotional_state: str = ""
    relationships: dict[str, str] = Field(default_factory=dict)
    knowledge: list[str] = Field(default_factory=list)
    status: str = ""
    summary: str = ""


class EventResult(BaseModel):
    """A plot event detected in a chunk."""
    event_type: str = "plot"
    summary: str = ""
    characters_involved: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    importance: str = "minor"
    spoiler_level: int = 0


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
