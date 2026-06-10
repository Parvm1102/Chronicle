"""novel_rag — temporal-aware, spoiler-safe RAG layer over Qdrant.

Indexes the full text of every novel as overlapping chunks with rich metadata
(series / novel / chapter / chunk serials, speakers, associated characters,
spoiler level, linked event ids) and exposes a retriever that never returns
content beyond the reader's current chapter.

The chat layer is intentionally NOT implemented here — this package only builds
and queries the vector store foundation.
"""

from __future__ import annotations

from .config import RagSettings, get_rag_settings

__all__ = ["RagSettings", "get_rag_settings"]
