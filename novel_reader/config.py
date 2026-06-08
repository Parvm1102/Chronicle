from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOVELS_DIR = DATA_DIR / "novels"
DATABASE_PATH = DATA_DIR / "reader.sqlite3"

SUPPORTED_FORMATS = {".txt", ".epub", ".pdf"}

# --- TTS (Chatterbox-Turbo microservice) ---
# Base URL of the standalone TTS service. Swap this to a Modal URL with no code
# changes. See tts_service/ for the service itself.
TTS_SERVICE_URL = os.environ.get("TTS_SERVICE_URL", "http://localhost:8070")
# Where the TTS service writes generated wavs (kept in sync via mounts/Modal vols).
TTS_CACHE_DIR = Path(os.environ.get("TTS_CACHE_DIR", str(DATA_DIR / "tts_cache")))
# Fallback voice actor for narration when no narrator character/actor is assigned.
TTS_DEFAULT_NARRATOR_ACTOR = os.environ.get("TTS_DEFAULT_NARRATOR_ACTOR", "Sohee")


def ensure_app_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    NOVELS_DIR.mkdir(exist_ok=True)
