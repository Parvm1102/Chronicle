"""Standalone Chatterbox-Turbo TTS microservice.

This package is intentionally decoupled from the main novel-reader app: it only
knows how to turn ``(text, voice_ref)`` into a WAV. The heavy, pinned Chatterbox
dependencies (torch==2.6.0, transformers==5.2.0, ...) live only here so the main
app stays light. The same :class:`~tts_service.engine.TTSEngine` is reused by the
local FastAPI server (``server.py``) and the Modal deployment (``modal_app.py``).
"""
