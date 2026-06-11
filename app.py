from dotenv import load_dotenv
load_dotenv(override=True)  # must be before any module reads env vars

import logging
import os
import threading

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from novel_reader.ui import CSS, READER_CSS, READER_JS, THEME_JS, CHAT_CSS, CHAT_JS, build_dashboard_app, build_reader_app, build_chat_app

logger = logging.getLogger(__name__)

app = FastAPI()

_PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
app.mount("/public", StaticFiles(directory=_PUBLIC_DIR), name="public")


@app.on_event("startup")
def _resume_interrupted_parses() -> None:
    """Check for novels with interrupted parses and resume them in background threads."""
    if os.environ.get("ENABLE_NOVEL_PARSING", "false").lower() != "true":
        return

    def _do_resume() -> None:
        try:
            from novel_parser.config import get_settings
            from novel_parser.database import DatabaseManager
            from novel_parser.llm_client import LLMClient
            from novel_parser.pipeline import NovelParsingPipeline
            from novel_reader.storage import LibraryStore

            settings = get_settings()
            db = DatabaseManager(settings)
            db.init_schema()

            incomplete = db.get_incomplete_novels()
            if not incomplete:
                db.close()
                return

            logger.info("Found %d interrupted parses — resuming", len(incomplete))
            store = LibraryStore()

            for meta in incomplete:
                novel_uuid = meta["novel_uuid"]
                novel_title = meta["novel_title"]

                # Find matching novel in SQLite to get sections
                novels = store.list_novels(include_archived=True)
                match = next((n for n in novels if n["uuid"] == novel_uuid), None)
                if not match:
                    logger.warning("Cannot resume '%s': not found in SQLite", novel_title)
                    continue

                sections = store.list_sections_full(match["id"])
                if not sections:
                    logger.warning("Cannot resume '%s': no sections in SQLite", novel_title)
                    continue

                llm = LLMClient(settings)
                pipeline = NovelParsingPipeline(db, llm, settings)

                def _run_resume(pipe=pipeline, uuid=novel_uuid, secs=sections, title=novel_title) -> None:
                    try:
                        pipe.resume(uuid, secs, title)
                    except Exception as exc:
                        logger.error("Resume failed for '%s': %s", title, exc)

                t = threading.Thread(target=_run_resume, daemon=True, name=f"resume-{novel_uuid[:8]}")
                t.start()
                logger.info("Triggered resume for '%s' (uuid=%s)", novel_title, novel_uuid)

        except ImportError:
            logger.debug("novel_parser package not available — skipping resume")
        except Exception as exc:
            logger.warning("Failed to check for interrupted parses: %s", exc)

    # Run in background to not block server startup
    threading.Thread(target=_do_resume, daemon=True, name="startup-resume").start()


@app.get("/")
def root():
    return RedirectResponse("/dashboard/")


@app.head("/")
def root_head():
    return RedirectResponse("/dashboard/")


@app.get("/favicon.ico")
def favicon():
    return RedirectResponse("/public/light_theme.ico")


# ── TTS narration routes ────────────────────────────────────────────────────
from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse, Response

from novel_reader.tts import TTSOrchestrator

_tts = TTSOrchestrator()


@app.get("/tts/playlist")
def tts_playlist(novel_id: int = Query(...), section: int = Query(...)):
    """Return the ordered, speakable units for a chapter and warm the cache."""
    result = _tts.build_playlist(novel_id, section)
    if result.get("status") == "ready":
        # Warm only the first window; the player slides it forward via /tts/prefetch.
        _tts.prefetch_window(novel_id, section, start_seq=None)
    return JSONResponse(result)


@app.get("/tts/prefetch")
def tts_prefetch(
    novel_id: int = Query(...), section: int = Query(...), seq: int = Query(...)
):
    """Slide the prefetch window forward to track the current playback line."""
    _tts.prefetch_window(novel_id, section, start_seq=seq)
    return JSONResponse({"status": "ok"})


@app.get("/tts/audio/{novel_id}/{section}/{seq}")
def tts_audio(novel_id: int, section: int, seq: int):
    """Stream the wav for a single unit (synthesised on demand, then cached)."""
    try:
        audio = _tts.synthesize_unit(novel_id, section, seq)
    except Exception as exc:
        logger.warning("TTS audio failed: %s", exc)
        raise HTTPException(status_code=502, detail="TTS service error") from exc
    if audio is None:
        raise HTTPException(status_code=404, detail="unit not found")
    return Response(content=audio, media_type="audio/wav")


@app.post("/tts/progress")
def tts_progress(novel_id: int = Query(...), section: int = Query(...)):
    _tts.save_progress(novel_id, section)
    return JSONResponse({"status": "ok"})


# ── In-character chat routes ────────────────────────────────────────────────
from fastapi import Body
from fastapi.responses import StreamingResponse

from novel_reader.chat import ChatOrchestrator

_chat = ChatOrchestrator()


@app.get("/chat/characters")
def chat_characters(novel_id: int = Query(...)):
    """Spoiler-safe character list for the chat sidebar."""
    return JSONResponse({"characters": _chat.list_characters(novel_id)})


@app.get("/chat/history")
def chat_history(novel_id: int = Query(...), character_id: int = Query(...)):
    """Full transcript for a (novel, character) pair."""
    return JSONResponse({"messages": _chat.get_history(novel_id, character_id)})


@app.post("/chat/send")
def chat_send(payload: dict = Body(...)):
    """Stream the character's reply as plain-text chunks (SSE-style)."""
    novel_id = int(payload.get("novel_id"))
    character_id = int(payload.get("character_id"))
    message = str(payload.get("message", ""))

    def _stream():
        for piece in _chat.answer_stream(novel_id, character_id, message):
            yield piece

    return StreamingResponse(_stream(), media_type="text/plain; charset=utf-8")


gr.mount_gradio_app(app, build_dashboard_app().queue(default_concurrency_limit=4), path="/dashboard", css=CSS, js=THEME_JS, theme=gr.themes.Base())
gr.mount_gradio_app(app, build_reader_app().queue(default_concurrency_limit=4), path="/reader", css=READER_CSS, js=READER_JS, theme=gr.themes.Base())
gr.mount_gradio_app(app, build_chat_app().queue(default_concurrency_limit=4), path="/chat", css=CHAT_CSS, js=CHAT_JS, theme=gr.themes.Base())


if __name__ == "__main__":
    # On Hugging Face Spaces (docker SDK) the Dockerfile sets
    # GRADIO_SERVER_PORT=7860 and GRADIO_SERVER_NAME=0.0.0.0. Locally it falls
    # back to 8060 on all interfaces.
    port = int(os.environ.get("PORT") or os.environ.get("GRADIO_SERVER_PORT") or 8060)
    host = os.environ.get("GRADIO_SERVER_NAME") or "0.0.0.0"
    uvicorn.run(app, host=host, port=port)
