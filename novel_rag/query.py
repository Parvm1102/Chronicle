"""Quick CLI to probe RAG retrieval quality against the live Qdrant collection.

Usage::

    # One-off query (current_chapter gates spoilers — only content up to it):
    python -m novel_rag.query <novel_uuid> --chapter 30 "who is Rue?"

    # Interactive REPL (type queries; blank line or 'q' to quit):
    python -m novel_rag.query <novel_uuid> --chapter 30

    # Restrict to a speaker, or widen/narrow the result count:
    python -m novel_rag.query <uuid> -c 30 --speaker Katniss --top 5 "the reaping"

If --chapter is omitted it defaults to a large number so the whole novel is in
scope (no spoiler gating) — handy for raw quality checks.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .retriever import RagRetriever, RetrievedChunk


def _print_chunk(i: int, ch: RetrievedChunk, max_chars: int) -> None:
    text = " ".join(ch.text.split())
    if max_chars <= 0 or len(text) <= max_chars:
        snippet = text
    else:
        snippet = text[:max_chars] + "…"
    speakers = ", ".join(ch.speakers) if ch.speakers else "-"
    chars = ", ".join(ch.associated_characters) if ch.associated_characters else "-"
    print(f"\n#{i}  score={ch.score:.4f}  ch{ch.chapter_index} «{ch.chapter_title}»  serial={ch.chunk_serial}")
    print(f"    speakers: {speakers}")
    print(f"    characters: {chars}")
    if ch.event_ids:
        print(f"    event_ids: {ch.event_ids}  (max_spoiler={ch.max_spoiler_level})")
    print(f"    {snippet}")


def _run_query(
    retriever: RagRetriever,
    query: str,
    *,
    novel_uuid: str,
    current_chapter: int,
    novel_number: int,
    series_key: str | None,
    include_series: bool,
    speaker: str | None,
    max_chars: int,
) -> None:
    results = retriever.retrieve(
        query,
        novel_uuid=novel_uuid,
        current_chapter=current_chapter,
        novel_number=novel_number,
        series_key=series_key,
        include_series=include_series,
        speaker=speaker,
    )
    if not results:
        print("  (no results)")
        return
    print(f"  {len(results)} result(s):")
    for i, ch in enumerate(results, 1):
        _print_chunk(i, ch, max_chars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe RAG retrieval quality.")
    parser.add_argument("novel_uuid", help="SQLite novel uuid to query against.")
    parser.add_argument("query", nargs="*", help="Query text (omit for interactive REPL).")
    parser.add_argument("-c", "--chapter", type=int, default=10_000,
                        help="Reader's current chapter (spoiler gate). Default: all.")
    parser.add_argument("-n", "--novel-number", type=int, default=1,
                        help="Book order within the series (default 1).")
    parser.add_argument("--series-key", default=None,
                        help="Series key (first book's uuid). Defaults to novel_uuid.")
    parser.add_argument("--no-series", action="store_true",
                        help="Restrict to this novel only (ignore earlier books).")
    parser.add_argument("--speaker", default=None,
                        help="Restrict results to chunks where this speaker appears.")
    parser.add_argument("--top", type=int, default=None,
                        help="Override final top_k results returned.")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text (no truncation).")
    parser.add_argument("--chars", type=int, default=280,
                        help="Max characters of chunk text to show (default 280; ignored with --full).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show INFO logs.")
    args = parser.parse_intermixed_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    retriever = RagRetriever()
    if args.top is not None:
        retriever._settings.rag_top_k = args.top  # noqa: SLF001 — intentional probe override

    series_key = args.series_key or args.novel_uuid
    max_chars = 0 if args.full else args.chars
    common = dict(
        novel_uuid=args.novel_uuid,
        current_chapter=args.chapter,
        novel_number=args.novel_number,
        series_key=series_key,
        include_series=not args.no_series,
        speaker=args.speaker,
        max_chars=max_chars,
    )

    if args.query:
        _run_query(retriever, " ".join(args.query), **common)
        return

    print("Interactive RAG query. Blank line or 'q' to quit.")
    while True:
        try:
            q = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q or q.lower() in {"q", "quit", "exit"}:
            break
        _run_query(retriever, q, **common)


if __name__ == "__main__":
    main()
