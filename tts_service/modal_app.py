"""Modal deployment scaffold for the Chatterbox-Turbo TTS service.

This reuses the *exact same* engine and FastAPI app as the local server — only
the hosting wrapper differs. It is provided for when you move TTS off your local
machine; deploy with::

    modal deploy tts_service/modal_app.py

Modal will print a public URL. Point the main app at it:

    TTS_SERVICE_URL=https://<your-workspace>--novel-reader-tts-ttsservice-fastapi-app.modal.run

Notes / things you may want to tweak at deploy time:
* ``gpu="T4"`` is plenty for Turbo; bump to ``A10G``/``A100`` if you need more.
* Voice samples are baked into the image so ``voice_ref`` resolution works with
  no mount. If you prefer to update voices without rebuilding, move them to a
  ``modal.Volume`` and mount it at ``/voice_samples`` instead.
"""

from __future__ import annotations

import modal

# Upstream Chatterbox pinned to the exact commit chatterbox-trying/ tracks.
# Installed from git so no local copy of the source is needed at deploy time
# (chatterbox-trying/ is gitignored).
CHATTERBOX_GIT = (
    "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git"
    "@3f35dfc8fbe63e5b29793289dc68f1875bb317a5"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1", "ffmpeg")
    .pip_install("fastapi>=0.110", "uvicorn[standard]>=0.29", "soundfile>=0.12")
    # Install Chatterbox straight from upstream at the pinned commit
    # (pulls torch==2.6.0, transformers==5.2.0, … per its pyproject).
    .pip_install(CHATTERBOX_GIT)
    .env(
        {
            "TTS_DEVICE": "cuda",
            "VOICE_SAMPLES_DIR": "/voice_samples",
            "TTS_CACHE_DIR": "/cache",
        }
    )
    # add_local_* steps must come last: Modal forbids build steps (env/run/pip)
    # after local files are added. Bake voice samples so voice_ref resolution
    # needs no runtime mount, and add our service package.
    .add_local_dir("voice_samples", "/voice_samples", copy=True)
    .add_local_python_source("tts_service")
)

app = modal.App("novel-reader-tts")

# Persisted across runs so model weights and generated audio survive restarts.
hf_cache = modal.Volume.from_name("novel-reader-hf-cache", create_if_missing=True)
tts_cache = modal.Volume.from_name("novel-reader-tts-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": hf_cache, "/cache": tts_cache},
    scaledown_window=300,
    timeout=600,
)
class TTSService:
    @modal.enter()
    def _load(self) -> None:
        from tts_service.engine import get_engine

        get_engine().load()

    @modal.asgi_app()
    def fastapi_app(self):
        # Same FastAPI app as the local server — its startup hook also ensures
        # the (already-loaded) engine is ready.
        from tts_service.server import app as fastapi_app

        return fastapi_app
