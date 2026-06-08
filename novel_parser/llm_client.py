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
        self._provider = self._settings.llm_provider

        # Determine the API key (Gemini can use gemini_api_key or fallback to llm_api_key)
        if self._provider == LLMProvider.GEMINI:
            api_key = self._settings.gemini_api_key or self._settings.llm_api_key or "not-needed"
        else:
            api_key = self._settings.llm_api_key or "not-needed"

        self._client = OpenAI(
            base_url=self._settings.llm_base_url,
            api_key=api_key,
            timeout=120.0,
        )
        self._model = self._settings.llm_model
        self._temperature = self._settings.llm_temperature
        self._max_retries = self._settings.llm_max_retries

    # ── public API ─────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a chat completion and return the assistant's text response.

        Args:
            messages: Standard OpenAI message list (role/content dicts).
            json_mode: If True, instruct the model to return valid JSON.
            max_tokens: Override default max_tokens for this call.
            timeout: Override default client timeout for this request.

        Returns:
            The assistant message content as a string.
        """
        kwargs = self._build_kwargs(messages, json_mode, max_tokens)
        return self._call_with_retry(kwargs, timeout=timeout)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Chat completion with JSON output, parsed into a dict, retrying on parse errors."""
        import time
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                raw = self.chat(messages, json_mode=True, max_tokens=max_tokens, timeout=timeout)
                return self._parse_json(raw)
            except ValueError as exc:
                last_exc = exc
                logger.warning(
                    "JSON parsing failed on attempt %d/3: %s. Retrying...",
                    attempt, exc
                )
                time.sleep(1)
        raise last_exc or ValueError("Failed to obtain valid JSON after 3 attempts")

    def chat_structured(
        self,
        messages: list[dict[str, str]],
        response_model: Type[BaseModel],
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> BaseModel:
        """Chat completion parsed into a Pydantic model.

        Sends in JSON mode and validates the response against *response_model*, retrying on errors.
        """
        import time
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                raw = self.chat(
                    messages,
                    json_mode=True,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                raw_dict = self._parse_json(raw)
                return response_model.model_validate(raw_dict)
            except (ValueError, Exception) as exc:
                last_exc = exc
                logger.warning(
                    "Structured validation failed on attempt %d/3: %s. Retrying...",
                    attempt, exc
                )
                time.sleep(1)
        raise last_exc or ValueError("Failed to obtain valid structured response after 3 attempts")

    # ── internals ──────────────────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, str]],
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        if json_mode and self._provider == LLMProvider.OLLAMA:
            messages = self._with_no_think(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.0 if json_mode else self._temperature,
        }
        # Use a generous default of 4096 to prevent response truncation in Ollama / cloud providers
        kwargs["max_tokens"] = max_tokens if max_tokens is not None else 4096

        if json_mode:
            # Both Groq and Ollama support this format;
            # Ollama also accepts format="json" but the OpenAI-compat
            # endpoint handles response_format fine.
            kwargs["response_format"] = {"type": "json_object"}

        if self._provider == LLMProvider.OLLAMA:
            # Ollama expects options like num_ctx in extra_body when using the OpenAI-compatible endpoint
            kwargs["extra_body"] = {
                "think": False,
                "options": {
                    "num_ctx": self._settings.llm_max_context,
                    "num_predict": max_tokens if max_tokens is not None else 4096,
                    "temperature": 0.0 if json_mode else self._temperature,
                }
            }
            # Disable thinking for Ollama models to prevent conflicts with JSON mode and slowdowns.
            kwargs["reasoning_effort"] = "none"

        return kwargs

    def _call_with_retry(self, kwargs: dict[str, Any], timeout: float | None = None) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "LLM request attempt %d/%d  model=%s",
                    attempt, self._max_retries, self._model,
                )
                call_kwargs = dict(kwargs)
                if timeout is not None:
                    call_kwargs["timeout"] = timeout
                response = self._client.chat.completions.create(**call_kwargs)
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
        """Parse JSON from LLM output, handling thinking tags, code fences, and conversational wrappers."""
        import re
        text = raw.strip()

        # Remove thinking blocks if present (e.g. <think>...</think>, <thought>...</thought>, etc.)
        text = re.sub(r"<(think|thought|reasoning|thought_process)>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        # Handle unclosed thinking blocks at the start of the text
        text = re.sub(r"^<(think|thought|reasoning|thought_process)>[^{]*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        extracted = LLMClient._extract_first_json_object(text)
        if extracted is not None:
            return extracted

        # Try to locate the JSON boundaries using brace matching
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as exc:
                try:
                    repaired = LLMClient._repair_json(json_str)
                    return json.loads(repaired)
                except Exception:
                    logger.debug(
                        "Brace-extracted substring is not valid JSON even after repair: %s\nSubstring: %s",
                        exc, json_str[:500]
                    )

        # Fallback to standard code fence stripping
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].startswith("```"):
                lines = lines[1:]
            text = "\n".join(lines).strip()
            # If the fence was like ```json, strip the 'json' line if it remains
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            try:
                repaired = LLMClient._repair_json(text)
                return json.loads(repaired)
            except Exception:
                logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", exc, raw[:500])
                raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    @staticmethod
    def _with_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        """Add Qwen/Ollama no-thinking hint to JSON calls without mutating caller data."""
        if not messages:
            return messages

        copied = [dict(message) for message in messages]
        last = copied[-1]
        content = last.get("content", "")
        if "/no_think" not in content:
            last["content"] = f"/no_think\n\n{content}"
        return copied

    @staticmethod
    def _extract_first_json_object(text: str) -> dict[str, Any] | None:
        """Find and parse the first balanced JSON object embedded in text."""
        starts = [idx for idx, char in enumerate(text) if char == "{"]
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            for idx in range(start, len(text)):
                char = text[idx]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:idx + 1]
                        try:
                            parsed = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if isinstance(parsed, dict):
                            return parsed
                        break
        return None

    @staticmethod
    def _repair_json(json_str: str) -> str:
        """Repair truncated JSON by closing open strings and matching brackets/braces."""
        json_str = json_str.strip()
        if not json_str:
            return json_str

        stack = []
        in_string = False
        escaped = False
        
        for i, char in enumerate(json_str):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    stack.append('{')
                elif char == '[':
                    stack.append('[')
                elif char == '}':
                    if stack and stack[-1] == '{':
                        stack.pop()
                elif char == ']':
                    if stack and stack[-1] == '[':
                        stack.pop()

        repaired = json_str
        if in_string:
            repaired += '"'
        
        # Close open containers in reverse order
        while stack:
            opener = stack.pop()
            if opener == "{":
                repaired += "}"
            elif opener == "[":
                repaired += "]"
                
        return repaired
