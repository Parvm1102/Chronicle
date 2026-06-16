# Setup Guide

This guide covers installing the **Omniscient Novel Reader**, configuring its
environment, and running it both **locally with Docker** and with **managed
cloud services**.

> For a high-level overview of what the project does, see [README.md](README.md).

---

## Prerequisites

- **Python 3.10+** (the project venv uses 3.10.12; the cloud image uses 3.12)
- **Docker + Docker Compose** — for PostgreSQL, Qdrant, and the TTS service
- **An NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — required for the TTS container (CPU works but is very slow)
- **An LLM** for parsing — either a local [Ollama](https://ollama.com) or a cloud API key (Groq / OpenRouter / Gemini)

> Parsing, RAG/chat, and TTS are all **optional**. Plain reading works with
> nothing but the main app.

---

## 1. Clone & install the main app

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

### Key variables

See [.env.example](.env.example) for the complete, commented list.

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql://novel_reader:changeme@localhost:5432/novel_reader` | PostgreSQL for the parser |
| `ENABLE_NOVEL_PARSING` | `false` | Turn on automatic parsing after upload |
| `LLM_PROVIDER` / `LLM_BASE_URL` / `LLM_MODEL` | `ollama` / local | LLM used for parsing |
| `ENABLE_RAG_INDEXING` | `false` | Index novels into Qdrant for spoiler-safe chat |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` | Vector DB for RAG |
| `VOICE_SAMPLES_DIR` | `./voice_samples` | Reference voice clips |
| `TTS_SERVICE_URL` | `http://localhost:8070` | Where the TTS service lives |
| `TTS_DEFAULT_NARRATOR_ACTOR` | `Sohee` | Narration voice when no narrator is assigned |
| `TTS_DEVICE` | `cuda` | TTS device (`cuda` / `mps` / `cpu`) |
| `TTS_PORT` | `8070` | Host port the TTS container publishes |

---

## 2. Choose your backends

The reader uses three optional backends. You can run each **locally with Docker**
or point it at a **managed cloud service** — the app code is identical either way.

| Backend | Local (Docker) | Cloud |
|---------|----------------|-------|
| PostgreSQL (parsed data) | `docker compose up -d postgres` | Supabase |
| Qdrant (RAG / chat) | `docker compose up -d qdrant` | Qdrant Cloud |
| TTS (narration) | `docker compose up -d --build tts` | Modal |
| LLM (parsing / chat) | Ollama | Groq / OpenRouter / Gemini |

---

## 3. Local setup (Docker)

### PostgreSQL

```bash
docker compose up -d postgres
```

Starts PostgreSQL 17 with the credentials from `.env` (defaults match the
default `DATABASE_URL`). The schema is created automatically on first use.

### Qdrant (for RAG / character chat)

```bash
docker compose up -d qdrant
```

Reachable at `http://localhost:6333`. Set `ENABLE_RAG_INDEXING=true` in `.env`
to index novels after parsing.

### TTS service (Chatterbox-Turbo, GPU)

```bash
# Build + start the GPU TTS container
docker compose up -d --build tts

# Watch the logs (first start downloads model weights from HuggingFace)
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
> `deploy.resources` block and requires the NVIDIA Container Toolkit. With no
> GPU, set `TTS_DEVICE=cpu` in `.env` and remove the `deploy:` block from the
> `tts` service in `docker-compose.yml` (generation will be slow).

#### TTS without Docker

Use a **separate** virtual environment so its pinned ML dependencies don't clash
with the main app:

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

### LLM (Ollama)

```bash
ollama serve
ollama pull qwen3.5:4b
```

Then in `.env`:

```ini
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3.5:4b
ENABLE_NOVEL_PARSING=true
```

---

## 4. Cloud setup (managed services)

Run the main app locally (or on Hugging Face Spaces) while pointing each backend
at a managed service. Only `.env` changes — no code changes.

### PostgreSQL → Supabase

```ini
DATABASE_URL=postgresql://postgres.xxxx:password@aws-0-region.pooler.supabase.com:6543/postgres
POSTGRES_SSL=require
```

### Qdrant → Qdrant Cloud

```ini
QDRANT_URL=https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.region.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-cloud-api-key
ENABLE_RAG_INDEXING=true
```

### LLM → Groq / OpenRouter / Gemini

```ini
# Groq / OpenRouter (OpenAI-compatible)
LLM_PROVIDER=cloud
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-70b-versatile
LLM_API_KEY=your-api-key-here

# Gemini
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=your-gemini-api-key-here
# LLM_MODEL=gemini-1.5-flash
```

### TTS → Modal

The same engine and FastAPI app are reused by `tts_service/modal_app.py`:

```bash
pip install modal
modal deploy tts_service/modal_app.py
```

Modal prints a public URL. Point the main app at it:

```ini
TTS_SERVICE_URL=https://<your-workspace>--novel-reader-tts-ttsservice-fastapi-app.modal.run
```

Voice samples are baked into the Modal image; model weights and the audio cache
persist in Modal Volumes.

### Whole app → Hugging Face Spaces

The repo ships a Docker Space configuration in the [README.md](README.md)
frontmatter and a [Dockerfile](Dockerfile) that runs `python app.py` on port
`7860`. Configure Supabase, Qdrant Cloud, the LLM key, and `TTS_SERVICE_URL`
(Modal) via the Space's **secrets**. The heavy GPU TTS service is **not** part of
this image — deploy it to Modal separately.

---

## 5. Run the main app

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:8060** — it redirects to the dashboard.

Typical full-stack local startup:

```bash
docker compose up -d postgres qdrant tts   # databases + TTS service
source .venv/bin/activate
python app.py                              # main app
```

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

## Useful commands

```bash
# Query the spoiler-safe vector DB directly
./.venv/bin/python -m novel_rag.query <uuid> -c <chapter number> --full "your query"
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Speaker shows "TTS service is offline" | Ensure the `tts` container/process is up and `TTS_SERVICE_URL` is correct (`curl http://localhost:8070/health`). |
| Speaker shows "Parsing in progress…" forever | The chapter isn't parsed yet. Ensure `ENABLE_NOVEL_PARSING=true`, Postgres is up, and the LLM is reachable. |
| Chat returns nothing / no characters | Ensure `ENABLE_RAG_INDEXING=true`, Qdrant is reachable, and the novel was parsed and indexed. |
| `docker compose up tts` fails on GPU | Install the NVIDIA Container Toolkit, or set `TTS_DEVICE=cpu` and drop the `deploy:` block. |
| First TTS request is slow | The model downloads from HuggingFace on first run and loads into VRAM; subsequent requests are cached. |
| `python: command not found` | Use `python3` (or activate the venv: `source .venv/bin/activate`). |
