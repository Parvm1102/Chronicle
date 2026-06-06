"""OpenAI-compatible LLM client for Ollama and cloud providers (Groq / OpenRouter).

Provides a single `LLMClient` abstraction that handles:
- JSON-mode activation per provider
- Retry with exponential back-off
- Rate-limit awareness (Groq)
- Request/response logging
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from .config import LLMProvider, Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion API."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client = OpenAI(
            base_url=self._settings.llm_base_url,
            api_key=self._settings.llm_api_key or "not-needed",
        )
        self._model = self._settings.llm_model
        self._temperature = self._settings.llm_temperature
        self._max_retries = self._settings.llm_max_retries
        self._provider = self._settings.llm_provider

    # ── public API ─────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        """Send a chat completion and return the assistant's text response.

        Args:
            messages: Standard OpenAI message list (role/content dicts).
            json_mode: If True, instruct the model to return valid JSON.
            max_tokens: Override default max_tokens for this call.

        Returns:
            The assistant message content as a string.
        """
        kwargs = self._build_kwargs(messages, json_mode, max_tokens)
        return self._call_with_retry(kwargs)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Chat completion with JSON output, parsed into a dict."""
        raw = self.chat(messages, json_mode=True, max_tokens=max_tokens)
        return self._parse_json(raw)

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: Type[BaseModel],
        *,
        max_tokens: int | None = None,
    ) -> BaseModel:
        """Chat completion parsed into a Pydantic model.

        Sends in JSON mode and validates the response against *response_model*.
        """
        raw_dict = self.chat_json(messages, max_tokens=max_tokens)
        return response_model.model_validate(raw_dict)

    # ── internals ──────────────────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if json_mode:
            # Both Groq and Ollama support this format;
            # Ollama also accepts format="json" but the OpenAI-compat
            # endpoint handles response_format fine.
            kwargs["response_format"] = {"type": "json_object"}

        return kwargs

    def _call_with_retry(self, kwargs: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "LLM request attempt %d/%d  model=%s",
                    attempt, self._max_retries, self._model,
                )
                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                logger.debug("LLM response length: %d chars", len(content))
                return content

            except Exception as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self._max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM output, handling markdown fences."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].startswith("```"):
                lines = lines[1:]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", exc, text[:500])
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
