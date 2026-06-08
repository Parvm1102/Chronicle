"""Thin FastAPI wrapper around :class:`tts_service.engine.TTSEngine`.

Run locally::

    uvicorn tts_service.server:app --host 0.0.0.0 --port 8070

The same ``app`` object is reused by the Modal deployment (see ``modal_app.py``).
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .engine import get_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Novel Reader TTS", version="1.0.0")
engine = get_engine()


class SynthRequest(BaseModel):
    text: str = Field(..., description="Text to speak; may contain [laugh]/[sigh] tags.")
    voice_ref: str = Field(
        ...,
        description="Voice sample path relative to the voice samples dir, "
        "e.g. 'Vivian/Vivian_happy_high.wav'.",
    )
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 1000
    repetition_penalty: float = 1.2


@app.on_event("startup")
def _preload() -> None:
    try:
        engine.load()
    except Exception:  # pragma: no cover - best effort; retried on first request
        logger.exception("Model preload failed; will retry on first /synthesize call")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded": engine.is_loaded,
        "device": engine.device,
        "sample_rate": engine.sample_rate,
    }


@app.post("/synthesize")
def synthesize(req: SynthRequest) -> Response:
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    try:
        audio = engine.synthesize(
            req.text,
            req.voice_ref,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to client as 500
        logger.exception("Synthesis failed")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    return Response(content=audio, media_type="audio/wav")
