#!/usr/bin/env python3
"""Diagnostic script — check parse status and results in PostgreSQL.

Usage:
    python check_parsing.py                   # show all novels + status
    python check_parsing.py --init            # just init schema + seed actors
    python check_parsing.py --novel UUID      # show details for a specific novel
    python check_parsing.py --test-parse UUID # trigger parsing for a novel manually
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import json
import sys

from novel_parser.config import get_settings
from novel_parser.database import DatabaseManager
from novel_parser.voice_actors import VoiceActorManager


def check_infra():
    """Verify PostgreSQL + Ollama connectivity."""
    print("=" * 60)
    print("INFRASTRUCTURE CHECK")
    print("=" * 60)

    settings = get_settings()
    print(f"\n  LLM provider:  {settings.llm_provider.value}")
    print(f"  LLM model:     {settings.llm_model}")
    print(f"  LLM base URL:  {settings.llm_base_url}")
    print(f"  Database URL:  {settings.database_url[:40]}...")
    print(f"  Samples dir:   {settings.voice_samples_dir}")

    # PostgreSQL
    try:
        db = DatabaseManager(settings)
        db.init_schema()
        print("\n  PostgreSQL:    CONNECTED ✓ (schema initialized)")
    except Exception as e:
        print(f"\n  PostgreSQL:    FAILED ✗ — {e}")
        return None

    # Voice actors
    try:
        va = VoiceActorManager(db, settings)
        va.seed()
        actors = db.get_all_voice_actors()
        print(f"  Voice actors:  {len(actors)} seeded ✓")
    except Exception as e:
        print(f"  Voice actors:  SEED FAILED — {e}")

    # Ollama
    try:
        from novel_parser.llm_client import LLMClient
        llm = LLMClient(settings)
        response = llm.chat([
            {"role": "user", "content": "Reply with exactly: OK"}
        ], max_tokens=5)
        print(f"  LLM test:      OK ✓ (response: {response.strip()[:20]})")
    except Exception as e:
        print(f"  LLM test:      FAILED ✗ — {e}")

    return db


def show_all_novels(db: DatabaseManager):
    """List all novels in novels_meta."""
    print("\n" + "=" * 60)
    print("NOVELS IN POSTGRESQL")
    print("=" * 60)

    with db.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM novels_meta ORDER BY created_at DESC"
        ).fetchall()

    if not rows:
        print("\n  (no novels found — upload a novel via the UI first)")
        print("  Make sure you restart the app after the .env fix!")
        return

    for row in rows:
        print(f"\n  Novel: {row['novel_title']}")
        print(f"    UUID:   {row['novel_uuid']}")
        print(f"    ID:     {row['id']}")
        print(f"    Status: {row['parse_status']}")
        if row['parse_message']:
            print(f"    Msg:    {row['parse_message']}")
        print(f"    Narrator: {row['narrator_type']}")


def show_novel_details(db: DatabaseManager, novel_uuid: str):
    """Show detailed parse results for a specific novel."""
    meta = db.get_novel_meta(novel_uuid)
    if not meta:
        print(f"\n  Novel with UUID '{novel_uuid}' not found in PostgreSQL.")
        print("  Available UUIDs:")
        with db.connection() as conn:
            rows = conn.execute("SELECT novel_uuid, novel_title FROM novels_meta").fetchall()
            for r in rows:
                print(f"    {r['novel_uuid']}  ({r['novel_title']})")
        return

    novel_meta_id = meta['id']
    print(f"\n{'=' * 60}")
    print(f"NOVEL: {meta['novel_title']}")
    print(f"{'=' * 60}")
    print(f"  Status:   {meta['parse_status']}")
    print(f"  Narrator: {meta['narrator_type']}")

    # Characters
    chars = db.get_characters(novel_meta_id)
    print(f"\n  CHARACTERS ({len(chars)}):")
    for c in chars:
        actor = ""
        if c.get('voice_actor_id'):
            actor = f" → voice_actor_id={c['voice_actor_id']}"
        narrator = " [NARRATOR]" if c.get('is_narrator') else ""
        print(f"    • {c['name']} ({c['role']}, {c['gender']}, {c['age_range']}){narrator}{actor}")
        if c.get('aliases'):
            print(f"      aliases: {c['aliases']}")
        if c.get('description'):
            print(f"      desc: {c['description'][:80]}...")

    # Parse progress
    for pass_num in (1, 2):
        prog = db.get_parse_progress(novel_meta_id, pass_num)
        if prog:
            print(f"\n  PASS {pass_num} PROGRESS:")
            print(f"    Status: {prog['status']}")
            print(f"    Section: {prog['current_section']}/{prog['total_sections']}")
            if prog.get('error_message'):
                print(f"    Error: {prog['error_message']}")

    # Dialogue entries sample
    with db.connection() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS cnt FROM dialogue_entries WHERE novel_meta_id = %s",
            (novel_meta_id,)
        ).fetchone()['cnt']
        sample = conn.execute(
            """SELECT * FROM dialogue_entries
               WHERE novel_meta_id = %s
               ORDER BY section_index, sequence_number LIMIT 10""",
            (novel_meta_id,)
        ).fetchall()

    print(f"\n  DIALOGUE ENTRIES: {total} total")
    if sample:
        print(f"  (showing first {len(sample)}):")
        for e in sample:
            emo = f"{e['emotion']}_{e['emotion_intensity']}" if e['emotion'] != 'neutral' else 'neutral'
            text_preview = e['raw_text'][:60].replace('\n', ' ')
            print(f"    [{e['entry_type']:9s}] {e['speaker_name']:12s} ({emo}) {text_preview}...")

    # Events
    with db.connection() as conn:
        event_count = conn.execute(
            "SELECT COUNT(*) AS cnt FROM novel_events WHERE novel_meta_id = %s",
            (novel_meta_id,)
        ).fetchone()['cnt']
    print(f"\n  EVENTS: {event_count}")

    # Profiles
    profiles = db.get_latest_profiles(novel_meta_id)
    print(f"  CHARACTER PROFILES: {len(profiles)}")


def test_parse(db: DatabaseManager, novel_uuid: str):
    """Manually trigger parsing for a novel (blocking)."""
    from novel_reader.storage import LibraryStore
    store = LibraryStore()

    # Find the novel in SQLite
    with store.connect() as conn:
        row = conn.execute(
            "SELECT id, uuid, title FROM novels WHERE uuid = ?", (novel_uuid,)
        ).fetchone()

    if not row:
        print(f"\n  Novel '{novel_uuid}' not found in SQLite.")
        print("  Available novels:")
        with store.connect() as conn:
            rows = conn.execute("SELECT uuid, title, status FROM novels ORDER BY id").fetchall()
            for r in rows:
                print(f"    {r['uuid']}  {r['title']}  [{r['status']}]")
        return

    novel_id = row['id']
    novel_title = row['title']
    print(f"\n  Found: '{novel_title}' (SQLite id={novel_id})")

    # Get sections
    sections = store.list_sections_full(novel_id)
    print(f"  Sections: {len(sections)}")

    if not sections:
        print("  ERROR: No sections found — novel hasn't been parsed yet.")
        return

    for s in sections[:5]:
        preview = s['text'][:50].replace('\n', ' ') if s.get('text') else '[empty]'
        print(f"    [{s['section_index']:3d}] {s['title'][:30]:30s} — {preview}...")

    print(f"\n  Starting parsing pipeline (this may take a while)...")
    print(f"  Using model: {get_settings().llm_model}")
    print("-" * 60)

    from novel_parser.pipeline import NovelParsingPipeline
    from novel_parser.llm_client import LLMClient

    settings = get_settings()
    llm = LLMClient(settings)
    pipeline = NovelParsingPipeline(db, llm, settings)

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    try:
        pipeline.run(novel_uuid, sections, novel_title)
        print("\n" + "=" * 60)
        print("PARSING COMPLETE ✓")
        print("=" * 60)
        show_novel_details(db, novel_uuid)
    except Exception as e:
        print(f"\n  PARSING FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Check novel parsing status")
    parser.add_argument("--init", action="store_true", help="Just init schema + seed actors")
    parser.add_argument("--novel", type=str, help="Show details for a specific novel UUID")
    parser.add_argument("--test-parse", type=str, metavar="UUID", help="Trigger parsing for a novel")
    args = parser.parse_args()

    db = check_infra()
    if db is None:
        sys.exit(1)

    if args.init:
        print("\n  Schema initialized and actors seeded. Done.")
        db.close()
        return

    if args.test_parse:
        test_parse(db, args.test_parse)
    elif args.novel:
        show_novel_details(db, args.novel)
    else:
        show_all_novels(db)

    db.close()


if __name__ == "__main__":
    main()
