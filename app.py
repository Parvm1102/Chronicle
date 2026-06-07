from dotenv import load_dotenv
load_dotenv(override=True)  # must be before any module reads env vars

import logging
import os
import threading

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from novel_reader.ui import CSS, READER_CSS, READER_JS, THEME_JS, build_dashboard_app, build_reader_app

logger = logging.getLogger(__name__)

app = FastAPI()


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

                def _run_resume(uuid=novel_uuid, secs=sections, title=novel_title) -> None:
                    try:
                        pipeline.resume(uuid, secs, title)
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


gr.mount_gradio_app(app, build_dashboard_app().queue(default_concurrency_limit=4), path="/dashboard", css=CSS, js=THEME_JS, theme=gr.themes.Base())
gr.mount_gradio_app(app, build_reader_app().queue(default_concurrency_limit=4), path="/reader", css=READER_CSS, js=READER_JS, theme=gr.themes.Base())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8060)
