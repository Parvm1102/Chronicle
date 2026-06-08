"""Framework-agnostic Chatterbox-Turbo TTS engine.

No web framework is imported here on purpose — the exact same engine runs under
the local FastAPI server and under Modal. The engine:

* lazily loads ``ChatterboxTurboTTS`` once,
* clones a speaker voice from a reference wav (``voice_ref``) and caches the
  prepared conditionals so repeated lines for the same voice are fast,
* honours the paralinguistic tags (``[laugh]``, ``[sigh]`` …) that the novel
  parser already injects into the text — these are native to the *Turbo* model,
* caches generated audio on disk keyed by ``sha256(text + voice_ref + params)``.

We use the Turbo model (not standard Chatterbox) specifically because Turbo is
the only variant that supports the paralinguistic tags. Turbo ignores
``exaggeration``/``cfg_weight``, so those are never used here.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _auto_device() -> str:
    """Pick the best available torch device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # torch not importable yet — caller will surface the real error
        pass
    return "cpu"


class TTSEngine:
    """Loads Chatterbox-Turbo once and synthesises cloned, emotion-aware speech."""

    def __init__(
        self,
        device: Optional[str] = None,
        voice_samples_dir: Optional[os.PathLike[str] | str] = None,
        cache_dir: Optional[os.PathLike[str] | str] = None,
    ) -> None:
        self.device = device or os.environ.get("TTS_DEVICE") or _auto_device()
        self.voice_samples_dir = Path(
            voice_samples_dir or os.environ.get("VOICE_SAMPLES_DIR", "./voice_samples")
        ).resolve()
        self.cache_dir = Path(
            cache_dir or os.environ.get("TTS_CACHE_DIR", "./data/tts_cache")
        ).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model: Any = None
        self._conds_cache: dict[str, Any] = {}
        # A single lock guards both model loading and generation: the underlying
        # torch model is not thread-safe and the GPU serialises work anyway.
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the Turbo model once (idempotent, thread-safe)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            logger.info("Loading ChatterboxTurboTTS on device=%s …", self.device)
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            logger.info("ChatterboxTurboTTS loaded (sample_rate=%s)", self._model.sr)

    @property
    def sample_rate(self) -> int:
        return int(self._model.sr) if self._model is not None else 24000

    # ── helpers ─────────────────────────────────────────────────────────────

    def _resolve_voice(self, voice_ref: str) -> Path:
        """Resolve a relative ``voice_ref`` to a wav inside the samples dir.

        Guards against path traversal so a malicious ``voice_ref`` cannot read
        files outside ``voice_samples_dir``.
        """
        if not voice_ref:
            raise ValueError("voice_ref is required")
        candidate = (self.voice_samples_dir / voice_ref).resolve()
        if os.path.commonpath([str(candidate), str(self.voice_samples_dir)]) != str(
            self.voice_samples_dir
        ):
            raise ValueError(f"voice_ref escapes voice samples dir: {voice_ref!r}")
        if not candidate.is_file():
            raise FileNotFoundError(f"voice sample not found: {voice_ref!r}")
        return candidate

    @staticmethod
    def _cache_key(text: str, voice_ref: str, params: dict[str, Any]) -> str:
        payload = json.dumps(
            {"text": text, "voice_ref": voice_ref, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ensure_conditionals(self, voice_ref: str, voice_path: Path) -> None:
        """Prepare (and cache) speaker conditionals for a voice reference."""
        cached = self._conds_cache.get(voice_ref)
        if cached is not None:
            self._model.conds = cached
            return
        # exaggeration is irrelevant for Turbo; pass 0.0 to avoid the warning path.
        self._model.prepare_conditionals(str(voice_path), exaggeration=0.0)
        self._conds_cache[voice_ref] = self._model.conds

    # ── synthesis ───────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_ref: str,
        *,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
    ) -> bytes:
        """Return 24 kHz mono WAV bytes for ``text`` spoken in voice ``voice_ref``.

        Results are cached on disk; identical requests return instantly.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("text is required")

        params = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        key = self._cache_key(text, voice_ref, params)
        out_path = self.cache_dir / f"{key}.wav"
        if out_path.exists():
            return out_path.read_bytes()

        voice_path = self._resolve_voice(voice_ref)
        self.load()

        import soundfile as sf

        with self._lock:
            self._ensure_conditionals(voice_ref, voice_path)
            wav = self._model.generate(
                text,
                audio_prompt_path=None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )

        data = wav.squeeze(0).detach().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, data, self.sample_rate, format="WAV", subtype="PCM_16")
        audio = buf.getvalue()

        tmp_path = out_path.with_suffix(".wav.tmp")
        tmp_path.write_bytes(audio)
        tmp_path.replace(out_path)  # atomic publish so prefetch readers never see a partial file
        return audio


# Process-global singleton so the FastAPI server and the Modal entrypoint share
# one loaded model.
_ENGINE: Optional[TTSEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> TTSEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = TTSEngine()
    return _ENGINE
