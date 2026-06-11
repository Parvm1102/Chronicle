---
title: Omniscient Novel Reader
emoji: 📊
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.8.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Break the fourth wall with the Omniscient Novel Reader
---

# Novel Reader

A self-hosted novel reader with a clean reading UI, automatic novel **parsing**
(character extraction, dialogue attribution, emotion tagging, voice-actor
assignment), and per-character **text-to-speech narration** powered by
Chatterbox-Turbo.

The system is made of three cooperating pieces:

| Component | Tech | Role |
|-----------|------|------|
| **Main app** (`app.py`, `novel_reader/`) | FastAPI + Gradio | Reading UI, library, ingestion, TTS orchestration |
| **Parser** (`novel_parser/`) | PostgreSQL + LLM | Extracts characters, dialogue, emotions, assigns voices |
| **TTS service** (`tts_service/`) | FastAPI + Chatterbox-Turbo (GPU) | Turns text + a voice sample into speech |

The TTS service runs as a **separate container/process** because it has heavy,
pinned ML dependencies (torch 2.6, transformers 5.2, …). The main app talks to
it over HTTP, so you can run it locally now and move it to **Modal** later by
only changing one URL.

---

## Architecture

```
                ┌──────────────────────────────────────────────┐
   Browser ───► │  Main app  (FastAPI + Gradio)  :8060          │
                │                                                │
                │   /dashboard   /reader   /tts/*                │
                │        │            │         │                │
                │     SQLite      SQLite     Orchestrator        │
                │   (library)   (sections)   (novel_reader/tts)  │
                └──────────────────┬──────────────┬─────────────┘
                                   │              │ HTTP
                          ┌────────▼─────┐   ┌────▼──────────────┐
                          │ PostgreSQL    │   │ TTS service :8070 │
                          │ (parsed data) │   │ Chatterbox-Turbo  │
                          └───────────────┘   │ (GPU)             │
                                   ▲          └────────┬──────────┘
                                   │                   │ reads
                          ┌────────┴─────┐    ┌────────▼──────────┐
                          │ novel_parser  │    │ voice_samples/    │
                          │ (LLM passes)  │    │ *.wav references  │
                          └───────────────┘    └───────────────────┘
```

- **Reader data** (sections, reading progress, bookmarks) lives in SQLite at
  `data/reader.sqlite3`.
- **Parsed data** (characters, `dialogue_entries` with speaker/emotion/voice)
  lives in **PostgreSQL**. The two are linked by the novel's `uuid`.
- **TTS** consumes only `(text, voice_ref)` and returns a WAV — it never touches
  either database.

---

## Prerequisites

- **Python 3.10+** (the project venv uses 3.10.12)
- **Docker + Docker Compose** (for PostgreSQL and the TTS service)
- **An NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — required for the TTS container (CPU works but is very slow)
- **An LLM** for parsing — either a local [Ollama](https://ollama.com) or a cloud key (Groq / OpenRouter / Gemini). Parsing is **optional**; reading works without it.

---

## 1. Environment setup (main app)

```bash
git clone <your-repo-url>
cd "novel reader"

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the main app dependencies (lightweight — no ML libs)
pip install -r requirements.txt
```

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

Key variables (see `.env.example` for the full list):

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://novel_reader:changeme@localhost:5432/novel_reader` | PostgreSQL for the parser |
| `ENABLE_NOVEL_PARSING` | `false` | Turn on automatic parsing after upload |
| `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_MODEL` | `ollama` / local | LLM used for parsing |
| `VOICE_SAMPLES_DIR` | `./voice_samples` | Reference voice clips |
| `TTS_SERVICE_URL` | `http://localhost:8070` | Where the TTS service lives |
| `TTS_DEFAULT_NARRATOR_ACTOR` | `Sohee` | Narration voice when no narrator is assigned |
| `TTS_DEVICE` | `cuda` | TTS container device (`cuda` / `mps` / `cpu`) |
| `TTS_PORT` | `8070` | Host port the TTS container publishes |

---

## 2. PostgreSQL (parsing backend)

Only needed if you want parsing and TTS (TTS reads the parsed dialogue).

```bash
docker compose up -d postgres
```

This starts PostgreSQL 17 with the credentials from `.env` (defaults match the
default `DATABASE_URL`). The schema is created automatically by the app on first
use.

To use Supabase instead of local Postgres, set `DATABASE_URL` to your Supabase
pooler URL and `POSTGRES_SSL=require` in `.env`.

---

## 3. TTS service (Chatterbox-Turbo, GPU)

The TTS service lives in `tts_service/` and is built from the local Chatterbox
source in `chatterbox-trying/chatterbox`.

### Option A — Docker Compose (recommended)

```bash
# Build + start the GPU TTS container
docker compose up -d --build tts

# Watch the logs (first start downloads the model weights from HuggingFace)
docker compose logs -f tts

# Verify it's healthy
curl http://localhost:8070/health
# {"status":"ok","loaded":true,"device":"cuda","sample_rate":24000}
```

The compose service mounts:

- `./voice_samples` → `/voice_samples` (read-only voice references)
- `./data/tts_cache` → `/data/tts_cache` (generated audio cache, survives restarts)
- `./data/hf` → `/data/hf` (HuggingFace model cache)

> **GPU note:** the `tts` service reserves all NVIDIA GPUs via the compose
> `deploy.resources` block and requires the NVIDIA Container Toolkit. If you have
> no GPU, set `TTS_DEVICE=cpu` in `.env` and remove the `deploy:` block from the
> `tts` service in `docker-compose.yml` (generation will be slow).

### Option B — Local process (no Docker)

Use a **separate** virtual environment for the TTS service so its pinned ML
dependencies don't clash with the main app:

```bash
python3 -m venv .venv-tts
source .venv-tts/bin/activate

# Install Chatterbox (pulls torch==2.6.0, transformers==5.2.0, …) + web layer
pip install ./chatterbox-trying/chatterbox
pip install -r tts_service/requirements.txt

# Run the service
export TTS_DEVICE=cuda            # or mps / cpu
export VOICE_SAMPLES_DIR=./voice_samples
export TTS_CACHE_DIR=./data/tts_cache
uvicorn tts_service.server:app --host 0.0.0.0 --port 8070
```

---

## 4. Run the main app

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:8060** — it redirects to the dashboard.

Typical full-stack startup:

```bash
docker compose up -d postgres tts     # databases + TTS service
source .venv/bin/activate
python app.py                          # main app
```

---

## Usage

1. **Upload** a `.txt`, `.epub`, or `.pdf` from the dashboard.
2. If `ENABLE_NOVEL_PARSING=true`, parsing runs in the background: it extracts
   characters, attributes dialogue, tags emotions, and assigns each character a
   voice actor from `voice_samples/`.
3. Open the book in the reader.
4. Click the **speaker icon** in the top bar (between *Dashboard* and *Back*) to
   start narration:
   - reading begins at your current chapter and **auto-advances** to the next,
   - the **current line is highlighted** and the page **auto-scrolls**,
   - upcoming lines are **prefetched** in a sliding window that tracks the
     playback head (spilling into the next chapter near a boundary) for near
     real-time playback — tune the depth with `TTS_LOOKAHEAD` (default `4`),
   - click the icon again to stop.
5. If a chapter isn't parsed yet, a small **"Parsing in progress…"** toast
   appears and fades after a few seconds.

---

## Voice samples

Voice references live in `voice_samples/<Actor>/<Actor>_<emotion>_<intensity>.wav`,
e.g. `Vivian/Vivian_happy_high.wav`. Nine actors ship by default (Aiden, Dylan,
Eric, Ono_Anna, Ryan, Serena, Sohee, Uncle_Fu, Vivian).

> Chatterbox-Turbo requires each reference clip to be **longer than 5 seconds**.
> Shorter clips error and that line is skipped during narration.

Emotion is conveyed two ways: the parser picks the matching emotion/intensity
sample file, **and** it injects paralinguistic tags (`[laugh]`, `[sigh]`,
`[chuckle]`, …) into the text — these are native to the Turbo model.

---

## Deploying TTS to Modal (later)

The same engine and FastAPI app are reused by `tts_service/modal_app.py`:

```bash
pip install modal
modal deploy tts_service/modal_app.py
```

Modal prints a public URL. Point the main app at it with no code changes:

```bash
# in .env
TTS_SERVICE_URL=https://<your-workspace>--novel-reader-tts-ttsservice-fastapi-app.modal.run
```

Voice samples are baked into the Modal image; model weights and the audio cache
persist in Modal Volumes.

---

## Project layout

```
app.py                  FastAPI entrypoint: mounts Gradio apps + /tts routes
docker-compose.yml      postgres + tts services
requirements.txt        Main app (light) dependencies
.env.example            All configuration variables

novel_reader/           Reading app
  ui.py                 Gradio UI, reader header, player JS/CSS
  storage.py            SQLite library + sections + progress
  ingestion.py          Upload → parse → store pipeline
  tts.py                TTS orchestrator (playlist build + prefetch)
  config.py             Paths + TTS settings

novel_parser/           LLM parsing pipeline (PostgreSQL)
  pipeline.py           Orchestrates passes
  pass1_characters.py   Character extraction
  pass2_dialogue.py     Dialogue attribution + emotion + tags
  voice_actors.py       Voice-actor matching + sample resolution
  database.py           PostgreSQL schema + queries

tts_service/            Standalone Chatterbox-Turbo microservice
  engine.py             Framework-agnostic TTS engine (model + cache)
  server.py             FastAPI wrapper (/health, /synthesize)
  modal_app.py          Modal deployment (reuses engine + server)
  Dockerfile            GPU image
  requirements.txt      Thin web-layer deps

voice_samples/          Reference voice clips per actor/emotion
data/                   SQLite DB, novel sources, caches
chatterbox-trying/      Chatterbox source (installed into the TTS image)
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Speaker shows "TTS service is offline" | Ensure the `tts` container/process is up and `TTS_SERVICE_URL` is correct (`curl http://localhost:8070/health`). |
| Speaker shows "Parsing in progress…" forever | The chapter isn't parsed yet. Ensure `ENABLE_NOVEL_PARSING=true`, Postgres is up, and the LLM is reachable. |
| `docker compose up tts` fails on GPU | Install the NVIDIA Container Toolkit, or set `TTS_DEVICE=cpu` and drop the `deploy:` block. |
| First TTS request is slow | The model downloads from HuggingFace on first run and loads into VRAM; subsequent requests are cached. |
| `python: command not found` | Use `python3` (or activate the venv: `source .venv/bin/activate`). |
|quering the vector db | run `./.venv/bin/python -m novel_rag.query <uuid> -c <chapter number> --full "your query"`
