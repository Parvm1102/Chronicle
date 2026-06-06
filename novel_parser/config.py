"""Centralised configuration loaded from environment variables / .env file.

Works identically on local dev, Docker, Hugging Face Spaces, and any other
host — everything is driven by env vars.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """Supported LLM back-ends."""
    OLLAMA = "ollama"
    CLOUD = "cloud"  # Groq, OpenRouter, or any OpenAI-compatible API


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Single source of truth for all novel_parser configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL (local Docker or Supabase) ---
    database_url: str = "postgresql://novel_reader:changeme@localhost:5432/novel_reader"
    postgres_ssl: str = ""  # set to "require" for Supabase

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.OLLAMA
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3.5:4b"
    llm_api_key: str = ""
    llm_max_context: int = 8192
    llm_temperature: float = 0.3
    llm_max_retries: int = 3

    # --- Voice samples ---
    voice_samples_dir: Path = Path("./voice_samples")

    # --- Parser behaviour ---
    enable_novel_parsing: bool = False
    parse_max_retries: int = 3
    parse_context_window: int = 3  # paragraphs of lookahead/lookbehind
    parse_dialogue_history_size: int = 8  # sliding window of recent entries

    # --- Derived / helpers ---
    @property
    def postgres_ssl_mode(self) -> Optional[str]:
        return self.postgres_ssl if self.postgres_ssl else None

    @model_validator(mode="after")
    def _validate_cloud_key(self) -> "Settings":
        if self.llm_provider == LLMProvider.CLOUD and not self.llm_api_key:
            raise ValueError(
                "LLM_API_KEY is required when LLM_PROVIDER=cloud "
                "(set it in .env or as an environment variable / HF Space secret)"
            )
        return self


# Singleton — import and use everywhere
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton, creating it on first call."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
