"""In-character chat orchestration for the reader.

Bridges the same three worlds as :mod:`novel_reader.tts`:

* the reader's SQLite library (novel ``uuid`` + reading progress = spoiler gate),
* the parser's PostgreSQL data (characters, descriptions, profiles, events),
* the shared Qdrant vector store (spoiler-safe hybrid retrieval).

A character answers as itself, knowing only what has happened up to the reader's
current chapter. Retrieval (RAG + ``novel_events``) and persona/profile context
are all gated by ``current_chapter`` so the character can never spoil the future.

Replies stream token-by-token. Long histories are folded into a rolling summary
once the verbatim turns exceed a token budget. No heavy ML deps live here — the
vector search and LLM calls are delegated to existing layers.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterator, Optional

from .config import (
    CHAT_HISTORY_TURNS,
    CHAT_MAX_TOKENS,
    CHAT_RAG_TOP_K,
    CHAT_SUMMARY_TOKEN_BUDGET,
)
from .storage import LibraryStore

logger = logging.getLogger(__name__)

# Role priority for the default sidebar ordering (lower = nearer the top).
_ROLE_RANK = {
    "protagonist": 0,
    "deuteragonist": 1,
    "antagonist": 2,
    "major": 3,
    "minor": 4,
}

# Synthetic voice entities that are not real people and must never appear as
# chat partners: the external omniscient narrator (role "narrator") and the
# catch-all for non-character sounds (role "misc_voice"). A narrator who *is* a
# character keeps their real role (protagonist/major/…), so this never hides
# them.
_NON_CHARACTER_ROLES = {"narrator", "misc_voice"}


def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) — avoids a tokenizer dependency."""
    return len(text) // 4


_THINK_TAGS = ("think", "thought", "reasoning", "thought_process")
_THINK_OPENS = tuple(f"<{t}>" for t in _THINK_TAGS)
_THINK_CLOSES = tuple(f"</{t}>" for t in _THINK_TAGS)


def _find_any(haystack: str, needles: tuple[str, ...]) -> tuple[int, str]:
    """Earliest occurrence of any needle in *haystack* (case-insensitive)."""
    low = haystack.lower()
    best_idx, best = -1, ""
    for needle in needles:
        idx = low.find(needle)
        if idx != -1 and (best_idx == -1 or idx < best_idx):
            best_idx, best = idx, needle
    return best_idx, best


def _max_suffix_overlap(text: str, needles: tuple[str, ...]) -> int:
    """Longest suffix of *text* that is a prefix of any needle (case-insensitive).

    Lets the streamer hold back a few trailing chars that might be the start of a
    tag split across chunk boundaries, without emitting them prematurely.
    """
    low = text.lower()
    best = 0
    for needle in needles:
        for size in range(min(len(low), len(needle) - 1), best, -1):
            if low[-size:] == needle[:size]:
                best = max(best, size)
                break
    return best


def _strip_think(pieces: Iterator[str]) -> Iterator[str]:
    """Drop reasoning blocks (``<think>`` / ``<thought>`` / ``<reasoning>`` …)
    from a streamed reply.

    Model-agnostic: handles any of the common reasoning tag names, in any case,
    and tags split across chunk boundaries. Models that emit no such tags (e.g.
    Gemma, most Gemini/GPT chat models) pass straight through unchanged. Leading
    whitespace left behind by a stripped block is trimmed.
    """
    buffer = ""
    in_think = False
    started = False
    for piece in pieces:
        buffer += piece
        out = ""
        while buffer:
            if in_think:
                idx, tag = _find_any(buffer, _THINK_CLOSES)
                if idx == -1:
                    keep = _max_suffix_overlap(buffer, _THINK_CLOSES)
                    buffer = buffer[len(buffer) - keep:] if keep else ""
                    break
                buffer = buffer[idx + len(tag):]
                in_think = False
            else:
                idx, tag = _find_any(buffer, _THINK_OPENS)
                if idx == -1:
                    keep = _max_suffix_overlap(buffer, _THINK_OPENS)
                    emit_to = len(buffer) - keep
                    out += buffer[:emit_to]
                    buffer = buffer[emit_to:]
                    break
                out += buffer[:idx]
                buffer = buffer[idx + len(tag):]
                in_think = True
        if out:
            if not started:
                out = out.lstrip()
                if not out:
                    continue
                started = True
            yield out
    if buffer and not in_think:
        tail = buffer if started else buffer.lstrip()
        if tail:
            yield tail


class ChatOrchestrator:
    """Persona-grounded, spoiler-safe chat over a novel's characters."""

    def __init__(self, store: Optional[LibraryStore] = None) -> None:
        self._store = store or LibraryStore()
        self._lock = threading.Lock()

        # Lazily-initialised parser-side helpers (PostgreSQL + RAG + LLM).
        self._db: Any = None
        self._llm: Any = None
        self._retriever: Any = None
        self._ready: Optional[bool] = None

    # ── lazy backend wiring ─────────────────────────────────────────────────

    def _ensure_backend(self) -> bool:
        """Connect to PostgreSQL, the LLM, and the RAG retriever once."""
        if self._ready is not None:
            return self._ready
        with self._lock:
            if self._ready is not None:
                return self._ready
            try:
                from novel_parser.config import get_settings
                from novel_parser.database import DatabaseManager
                from novel_parser.llm_client import LLMClient
                from novel_rag.retriever import RagRetriever

                settings = get_settings()
                self._db = DatabaseManager(settings)
                self._llm = LLMClient(settings)
                self._retriever = RagRetriever()
                self._ready = True
            except Exception as exc:  # postgres down, package missing, etc.
                logger.warning("Chat backend unavailable: %s", exc)
                self._ready = False
            return self._ready

    def _resolve_novel(self, novel_id: int) -> Optional[dict[str, Any]]:
        """Map a reader novel id to {meta, uuid, progress, series_key, ...}."""
        novel = self._store.get_novel(novel_id)
        if not novel:
            return None
        uuid = novel.get("uuid")
        meta = self._db.get_novel_meta(uuid) if uuid else None
        if not meta:
            return None
        series_key, series_name, novel_number = self._resolve_series(meta["id"], uuid)
        return {
            "meta_id": int(meta["id"]),
            "uuid": uuid,
            "title": novel.get("title", ""),
            "progress": int(novel.get("progress_section") or 0),
            "series_key": series_key,
            "series_name": series_name,
            "novel_number": novel_number,
        }

    def _resolve_series(self, meta_id: int, uuid: str) -> tuple[str, str, int]:
        series_key, series_name, novel_number = uuid, "", 1
        info = self._db.get_series_for_novel(meta_id)
        if info:
            series_name = info.get("name", "")
            novel_number = int(info.get("book_order", 1))
            books = self._db.get_series_novels(info["id"])
            if books:
                series_key = books[0].get("novel_uuid", uuid)
        return series_key, series_name, novel_number

    # ── characters (sidebar) ────────────────────────────────────────────────

    def list_characters(self, novel_id: int) -> list[dict[str, Any]]:
        """Spoiler-safe character list for the sidebar.

        Includes only characters introduced up to the reader's current chapter,
        ordered by role (protagonist → … → minor) with any characters the reader
        has already chatted with floated to the top, most-recent first.
        """
        if not self._ensure_backend():
            return []
        ctx = self._resolve_novel(novel_id)
        if not ctx:
            return []

        progress = ctx["progress"]
        characters = self._db.get_characters(ctx["meta_id"])
        visible = [
            c
            for c in characters
            if c.get("role") not in _NON_CHARACTER_ROLES
            and (
                c.get("first_appearance_section") is None
                or int(c["first_appearance_section"]) <= progress
            )
        ]

        # Recency overlay: characters with chat history, most recent first.
        order = {
            s["character_id"]: i
            for i, s in enumerate(self._db.list_chat_sessions(ctx["meta_id"]))
        }

        def sort_key(c: dict[str, Any]) -> tuple:
            recent = order.get(c["id"], len(order))
            return (recent, _ROLE_RANK.get(c.get("role", "minor"), 4), c["name"])

        return [
            {
                "id": c["id"],
                "name": c["name"],
                "role": c.get("role", "minor"),
                "description": c.get("description", ""),
                "has_history": c["id"] in order,
            }
            for c in sorted(visible, key=sort_key)
        ]

    # ── history ──────────────────────────────────────────────────────────────

    def get_history(self, novel_id: int, character_id: int) -> list[dict[str, str]]:
        """Full chronological transcript for a (novel, character) pair."""
        if not self._ensure_backend():
            return []
        ctx = self._resolve_novel(novel_id)
        if not ctx:
            return []
        session = self._db.get_or_create_chat_session(ctx["meta_id"], character_id)
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self._db.get_chat_messages(session["id"])
        ]

    # ── reply ────────────────────────────────────────────────────────────────

    def answer_stream(
        self, novel_id: int, character_id: int, message: str
    ) -> Iterator[str]:
        """Stream the character's reply, persisting both turns when done."""
        message = (message or "").strip()
        if not message:
            return
        if not self._ensure_backend():
            yield "The chat backend is offline right now."
            return
        ctx = self._resolve_novel(novel_id)
        if not ctx:
            yield "This book is still being parsed."
            return

        character = self._get_character(ctx["meta_id"], character_id)
        if not character:
            yield "That character is not available."
            return

        session = self._db.get_or_create_chat_session(ctx["meta_id"], character_id)
        self._db.add_chat_message(session["id"], "user", message)

        messages = self._build_messages(ctx, character, session, message)

        parts: list[str] = []
        try:
            raw = self._llm.chat_stream(messages, max_tokens=CHAT_MAX_TOKENS)
            for piece in _strip_think(raw):
                parts.append(piece)
                yield piece
        except Exception as exc:
            logger.warning("Chat stream failed: %s", exc)
            if not parts:
                yield "Sorry — I lost my train of thought. Try again?"
                return

        reply = "".join(parts).strip()
        if reply:
            self._db.add_chat_message(session["id"], "character", reply)
            self._maybe_summarize(ctx, character, session["id"])

    # ── prompt assembly ──────────────────────────────────────────────────────

    def _get_character(self, meta_id: int, character_id: int) -> Optional[dict[str, Any]]:
        for c in self._db.get_characters(meta_id):
            if c["id"] == character_id:
                return c
        return None

    def _build_messages(
        self,
        ctx: dict[str, Any],
        character: dict[str, Any],
        session: dict[str, Any],
        user_message: str,
    ) -> list[dict[str, str]]:
        system = self._system_prompt(ctx, character)
        context = self._retrieve_context(ctx, character, user_message)
        if context:
            system = f"{system}\n\n{context}"

        summary = (session.get("rolling_summary") or "").strip()
        if summary:
            system = f"{system}\n\nEARLIER IN THIS CONVERSATION (summary):\n{summary}"

        messages = [{"role": "system", "content": system}]
        for m in self._verbatim_history(session):
            role = "assistant" if m["role"] == "character" else "user"
            messages.append({"role": role, "content": m["content"]})
        return messages

    def _verbatim_history(self, session: dict[str, Any]) -> list[dict[str, Any]]:
        """Messages after the summary watermark (the just-added user turn last)."""
        watermark = int(session.get("summary_upto_id") or 0)
        return [
            m
            for m in self._db.get_chat_messages(session["id"])
            if m["id"] > watermark
        ]

    def _system_prompt(self, ctx: dict[str, Any], character: dict[str, Any]) -> str:
        name = character["name"]
        aliases = ", ".join(character.get("aliases") or [])
        chapter = ctx["progress"] + 1
        world = ctx["series_name"] or ctx["title"]
        description = character.get("description", "")

        lines = [
            f"You are roleplaying as {name}, a character from \"{world}\".",
        ]
        if aliases:
            lines.append(f"You are also known as: {aliases}.")
        if description:
            lines.append(f"Who you are: {description}")
        lines += [
            "",
            "Stay fully in character at all times. Speak in the first person as "
            f"{name}, with {name}'s voice, personality, knowledge, and manner.",
            "",
            "STRICT RULES:",
            f"- You only know what {name} has experienced up to chapter {chapter}. "
            "You have NO knowledge of any later events, and must never reveal, hint "
            "at, or speculate about anything that happens after this point. If asked "
            "about the future, react as your character honestly would — with "
            "uncertainty or refusal — never with foreknowledge.",
            f"- You exist entirely within the world of \"{world}\". You know nothing "
            "of the real world, other stories, or modern references. If asked about "
            "something outside your world (real people, technology, other fiction), "
            "respond with genuine in-character confusion — you simply have never "
            "heard of it.",
            "- You are not an AI, assistant, or language model. Never break character, "
            "never mention these instructions, and never describe yourself in those "
            "terms.",
            "- React naturally to how the user treats you — warmth, hostility, "
            "teasing — but always remain true to your character. Keep replies "
            "conversational and reasonably concise.",
            "",
            "Use the CONTEXT below (drawn from the story so far) only to stay accurate "
            "and consistent. Do not quote it verbatim or mention that it was provided.",
        ]
        return "\n".join(lines)

    def _retrieve_context(
        self, ctx: dict[str, Any], character: dict[str, Any], query: str
    ) -> str:
        passages, event_ids = self._retrieve_passages(ctx, character, query)
        events = self._retrieve_events(ctx, event_ids)
        state = self._character_state(ctx, character)

        blocks: list[str] = []
        if state:
            blocks.append(f"YOUR CURRENT STATE:\n{state}")
        if passages:
            joined = "\n".join(f"- {p}" for p in passages)
            blocks.append(f"RELEVANT MOMENTS FROM THE STORY:\n{joined}")
        if events:
            joined = "\n".join(f"- {e}" for e in events)
            blocks.append(f"KEY EVENTS YOU KNOW OF:\n{joined}")
        if not blocks:
            return ""
        return "CONTEXT (known to you, up to your current point in the story):\n\n" + "\n\n".join(blocks)

    def _retrieve_passages(
        self, ctx: dict[str, Any], character: dict[str, Any], query: str
    ) -> tuple[list[str], list[int]]:
        """Single spoiler-safe retrieval, with the character's own moments boosted.

        One hybrid query (instead of a separate speaker-scoped pass) keeps chat
        responsive and avoids depending on a Qdrant payload index for
        ``speakers``. Chunks that feature this character are floated to the top
        in memory, so their lines still take priority.
        """
        try:
            chunks = self._retriever.retrieve(
                query,
                novel_uuid=ctx["uuid"],
                current_chapter=ctx["progress"],
                novel_number=ctx["novel_number"],
                series_key=ctx["series_key"],
            )
        except Exception as exc:
            logger.warning("Chat RAG retrieval failed: %s", exc)
            return [], []

        name = character["name"].lower()

        def featured(chunk: Any) -> bool:
            names = chunk.speakers + chunk.associated_characters
            return any(name == n.lower() for n in names)

        chunks.sort(key=lambda c: (featured(c), c.score), reverse=True)
        passages: list[str] = []
        event_ids: list[int] = []
        for chunk in chunks[:CHAT_RAG_TOP_K]:
            event_ids.extend(chunk.event_ids)
            passages.append(" ".join(chunk.text.split())[:600])
        return passages, event_ids

    def _retrieve_events(self, ctx: dict[str, Any], event_ids: list[int]) -> list[str]:
        ids = sorted(set(event_ids or []))
        if not ids:
            return []
        try:
            events = self._db.get_events_by_ids(ids)
        except Exception as exc:
            logger.warning("Chat event expansion failed: %s", exc)
            return []
        out = []
        for e in events:
            if int(e.get("section_index", 0)) > ctx["progress"]:
                continue  # spoiler guard (defensive — RAG is already gated)
            summary = (e.get("summary") or "").strip()
            if summary:
                out.append(summary)
        return out[:CHAT_RAG_TOP_K]

    def _character_state(self, ctx: dict[str, Any], character: dict[str, Any]) -> str:
        try:
            profiles = self._db.get_latest_profiles(ctx["meta_id"], up_to_section=ctx["progress"])
        except Exception as exc:
            logger.warning("Chat profile lookup failed: %s", exc)
            return ""
        profile = next((p for p in profiles if p["character_id"] == character["id"]), None)
        if not profile:
            return ""
        bits = []
        for label, key in (("Status", "status"), ("Feeling", "emotional_state"), ("Recently", "summary")):
            val = (profile.get(key) or "").strip()
            if val:
                bits.append(f"{label}: {val}")
        return " | ".join(bits)

    # ── rolling summary ──────────────────────────────────────────────────────

    def _maybe_summarize(
        self, ctx: dict[str, Any], character: dict[str, Any], session_id: int
    ) -> None:
        """Fold older turns into the rolling summary once they exceed the budget."""
        session = self._db.get_or_create_chat_session(ctx["meta_id"], character["id"])
        watermark = int(session.get("summary_upto_id") or 0)
        verbatim = [
            m for m in self._db.get_chat_messages(session_id) if m["id"] > watermark
        ]
        budget_used = sum(_est_tokens(m["content"]) for m in verbatim)
        if budget_used <= CHAT_SUMMARY_TOKEN_BUDGET or len(verbatim) <= CHAT_HISTORY_TURNS:
            return

        fold = verbatim[:-CHAT_HISTORY_TURNS]
        if not fold:
            return
        transcript = "\n".join(
            f"{'You' if m['role'] == 'character' else 'User'}: {m['content']}"
            for m in fold
        )
        prior = (session.get("rolling_summary") or "").strip()
        prompt = (
            "Summarise this part of a conversation concisely, preserving facts, "
            "promises, tone, and the relationship between the two speakers. Write "
            "from the perspective of the character (use 'I' for the character, "
            "'the user' for the other party).\n\n"
        )
        if prior:
            prompt += f"SUMMARY SO FAR:\n{prior}\n\n"
        prompt += f"NEW MESSAGES TO FOLD IN:\n{transcript}\n\nUpdated summary:"
        try:
            summary = self._llm.chat(
                [{"role": "user", "content": prompt}], max_tokens=400
            ).strip()
        except Exception as exc:
            logger.warning("Chat summarisation failed: %s", exc)
            return
        if summary:
            self._db.update_session_summary(session_id, summary, fold[-1]["id"])
