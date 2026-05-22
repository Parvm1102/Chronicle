from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NOVELS_DIR = DATA_DIR / "novels"
DATABASE_PATH = DATA_DIR / "reader.sqlite3"

SUPPORTED_FORMATS = {".txt", ".epub", ".pdf"}


def ensure_app_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    NOVELS_DIR.mkdir(exist_ok=True)
