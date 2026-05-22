from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedSection:
    index: int
    title: str
    text: str
    source_locator: str
    html: str = ""


@dataclass(frozen=True)
class ParsedBook:
    title: str
    author: str
    file_format: str
    sections: list[ParsedSection]


@dataclass
class BookCoverInfo:
    """Holds the extracted cover image and extra book metadata."""
    cover_b64: str = ""   # data-URI or empty string
    series: str = ""
    genres: str = ""
