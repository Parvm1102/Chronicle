

<div align="center">

<img src="public/light_theme.jpeg" alt="Omniscient Novel Reader logo" width="160" />

# Chronicle

**Read any novel, hear every character, and talk to them — break the fourth wall.**

Live: https://build-small-hackathon-omniscient-novel-reader.hf.space/dashboard/

Demo Video: https://www.youtube.com/watch?v=6n9lhJjK10U

</div>

---

A self-hosted novel reader that turns a plain `.txt`, `.epub`, or `.pdf` into a
rich, interactive experience. Upload a book and the system:

- **Reads** it in a clean, distraction-free reader with progress, bookmarks, and auto-scroll.
- **Parses** it with an LLM — extracting characters, attributing every line of dialogue, tagging emotions, and assigning each character a voice.
- **Narrates** it with per-character text-to-speech (Chatterbox-Turbo), where each speaker has their own voice and emotion. Lines are highlighted as they play and prefetched for near real-time playback.
- **Lets you chat** with any character — answers are grounded in a **spoiler-safe** retrieval layer that never reveals anything past your current chapter.

## How to use it

1. **Upload** a `.txt`, `.epub`, or `.pdf` from the dashboard.
2. With parsing enabled, the book is analysed in the background (characters, dialogue, emotions, voice assignment).
3. **Open** the book in the reader and start reading.
4. Click the **speaker icon** to start narration — it auto-advances chapters, highlights the current line, and auto-scrolls.
5. Open the **chat** to talk to any character in-universe, with no spoilers beyond where you've read.

> See **[SETUP.md](SETUP.md)** for installation, configuration, and both Docker and cloud deployment instructions.

## Technologies

| Layer | Tech |
|-------|------|
| Web / UI | FastAPI, Gradio, vanilla-JS narration player |
| Reader storage | SQLite (sections, progress, bookmarks) |
| Parsing | LLM via Ollama / Groq / OpenRouter / Gemini |
| Parsed data | PostgreSQL (or Supabase) |
| RAG / chat | Qdrant + FastEmbed (ONNX hybrid dense + BM25, reranking) |
| Narration | Chatterbox-Turbo TTS (GPU), deployable to Modal |
| Infra | Docker Compose, optional NVIDIA Container Toolkit |

## Architecture

The system is three cooperating pieces. The **TTS service** runs as a separate
container/process because of its heavy, pinned ML dependencies (torch 2.6,
transformers 5.2, …). The main app talks to it over HTTP, so it can run locally
now and move to **Modal** later by changing only one URL.

```
                ┌──────────────────────────────────────────────┐
   Browser ───► │  Main app  (FastAPI + Gradio)  :8060          │
                │                                                │
                │   /dashboard   /reader   /chat   /tts/*        │
                │        │           │        │        │         │
                │     SQLite      SQLite    RAG    Orchestrator   │
                │   (library)   (sections) (Qdrant)  (tts.py)     │
                └──────────┬───────────┬────────┬──────┬─────────┘
                           │           │        │      │ HTTP
                  ┌────────▼─────┐ ┌───▼────┐ ┌─▼────────────────┐
                  │ PostgreSQL    │ │ Qdrant │ │ TTS service :8070 │
                  │ (parsed data) │ │ (RAG)  │ │ Chatterbox-Turbo  │
                  └───────┬───────┘ └────────┘ │ (GPU)             │
                          ▲                     └────────┬─────────┘
                  ┌───────┴──────┐              ┌────────▼─────────┐
                  │ novel_parser  │              │ voice_samples/   │
                  │ (LLM passes)  │              │ *.wav references │
                  └───────────────┘              └──────────────────┘
```

- **Reader data** (sections, progress, bookmarks) lives in SQLite at `data/reader.sqlite3`.
- **Parsed data** (characters, dialogue with speaker/emotion/voice) lives in **PostgreSQL**, linked to the reader by the novel's `uuid`.
- **RAG data** (chunked full text with spoiler metadata) lives in **Qdrant**.
- **TTS** consumes only `(text, voice_ref)` and returns a WAV — it never touches the databases.

## How it works

1. **Ingestion** (`novel_reader/ingestion.py`) — the uploaded file is split into
   chapters/sections and stored in SQLite. Reading works immediately, with or
   without parsing.
2. **Parsing** (`novel_parser/`) runs LLM passes:
   - *Pass 1* extracts the cast of characters.
   - *Pass 2* attributes each line of dialogue to a speaker, tags emotion/intensity, and injects paralinguistic cues (`[laugh]`, `[sigh]`, …).
   - Voice actors are matched to characters from `voice_samples/`.
3. **RAG** (`novel_rag/`) indexes the full text into Qdrant as overlapping chunks
   with rich metadata (chapter serials, speakers, spoiler level). The retriever
   is **temporal-aware** and never returns content past the reader's chapter.
4. **Narration** (`novel_reader/tts.py`) builds a playlist of speakable units,
   resolves each to `(text, voice_ref)`, and calls the TTS service. A sliding
   prefetch window keeps upcoming lines synthesized ahead of the playback head
   (tune the depth with `TTS_LOOKAHEAD`, default `4`).
5. **Chat** (`novel_reader/chat.py`) answers as a character using only
   spoiler-safe retrieved context, streaming the reply token-by-token.

## Project layout

```
app.py                FastAPI entrypoint — mounts Gradio apps + /tts, /chat routes
docker-compose.yml    postgres + qdrant + tts services
novel_reader/         Reading app: UI, storage, ingestion, TTS + chat orchestration
novel_parser/         LLM parsing pipeline (characters, dialogue, voices) → PostgreSQL
novel_rag/            Temporal-aware, spoiler-safe RAG over Qdrant
tts_service/          Standalone Chatterbox-Turbo microservice (Docker + Modal)
voice_samples/        Reference voice clips per actor/emotion
public/               Logo + theme assets
data/                 SQLite DB, novel sources, model + audio caches
```

## License

[MIT](LICENSE)
