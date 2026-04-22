from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    cached: bool = False
    latency_ms: int = 0


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        cache_enabled: Optional[bool] = None,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model = model or settings.LLM_MODEL
        self.temperature = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        self.max_tokens = max_tokens or settings.LLM_MAX_TOKENS
        self.timeout = timeout or settings.LLM_TIMEOUT
        self.cache_enabled = (
            cache_enabled if cache_enabled is not None else settings.LLM_CACHE_ENABLED
        )
        self.cache_dir = cache_dir or settings.LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # litellm reads os.environ, not our Settings object.
        if settings.GEMINI_API_KEY:
            os.environ.setdefault("GEMINI_API_KEY", settings.GEMINI_API_KEY)
        if settings.OLLAMA_API_BASE:
            os.environ.setdefault("OLLAMA_API_BASE", settings.OLLAMA_API_BASE)

        logger.info(
            "LLMClient ready: model=%s temp=%.2f cache=%s",
            self.model,
            self.temperature,
            "on" if self.cache_enabled else "off",
        )

    def complete(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[str] = None,
    ) -> LLMResponse:
        cache_key = self._cache_key(user_prompt, system_prompt, response_format)

        if self.cache_enabled:
            hit = self._cache_read(cache_key)
            if hit is not None:
                logger.debug("LLM cache HIT %s", cache_key[:12])
                return LLMResponse(
                    text=hit["text"],
                    model=hit.get("model", self.model),
                    usage=hit.get("usage", {}),
                    cached=True,
                    latency_ms=0,
                )

        text, usage, latency_ms = self._call_provider(
            user_prompt, system_prompt, response_format
        )

        if self.cache_enabled:
            self._cache_write(
                cache_key,
                {"text": text, "model": self.model, "usage": usage},
            )

        return LLMResponse(
            text=text,
            model=self.model,
            usage=usage,
            cached=False,
            latency_ms=latency_ms,
        )

    def complete_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        max_retries: int = 1,
    ) -> Any:
        prompt = user_prompt
        for attempt in range(max_retries + 1):
            resp = self.complete(prompt, system_prompt, response_format="json")
            try:
                return _parse_json_loose(resp.text)
            except ValueError as e:
                logger.warning(
                    "LLM returned non-JSON on attempt %d/%d: %s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                if attempt == max_retries:
                    raise
                prompt = (
                    user_prompt
                    + "\n\nIMPORTANT: return ONLY a JSON array/object. "
                    "No markdown fences, no commentary."
                )
        raise RuntimeError("unreachable")

    def is_ollama(self) -> bool:
        return self.model.startswith("ollama/")

    def _ollama_options(self) -> Dict[str, Any]:
        """Ollama-native sampler block. Forwarded verbatim to /api/chat by
        litellm when the provider slug starts with ``ollama/``."""
        return {
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": settings.OLLAMA_NUM_PREDICT,
            "temperature": self.temperature,
            "top_p": settings.OLLAMA_TOP_P,
            "top_k": settings.OLLAMA_TOP_K,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
        }

    def _preflight_tokens(self, prompt: str) -> int:
        return int(len(prompt) / settings.PREFLIGHT_CHARS_PER_TOKEN)

    def _warn_if_overflow(self, user_prompt: str, system_prompt: Optional[str]) -> None:
        """Warn when an Ollama prompt approaches ``num_ctx``. Ollama would
        otherwise truncate the head silently — the response can still parse
        as valid JSON while missing half the input."""
        if not self.is_ollama():
            return
        est = self._preflight_tokens((system_prompt or "") + user_prompt)
        budget = int(settings.OLLAMA_NUM_CTX * settings.PREFLIGHT_CTX_RATIO)
        if est > budget:
            logger.warning(
                "Ollama preflight: prompt ~%d tokens > %d%% of num_ctx=%d. "
                "Ollama will truncate silently. Raise OLLAMA_NUM_CTX in .env "
                "or shorten the prompt.",
                est,
                int(settings.PREFLIGHT_CTX_RATIO * 100),
                settings.OLLAMA_NUM_CTX,
            )
        else:
            logger.debug(
                "Ollama preflight OK: ~%d tokens / num_ctx=%d",
                est,
                settings.OLLAMA_NUM_CTX,
            )

    def _call_provider(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        response_format: Optional[str],
    ) -> tuple[str, Dict[str, int], int]:
        # Lazy import so unit tests don't need the network to construct an
        # LLMClient.
        import litellm  # noqa: WPS433

        self._warn_if_overflow(user_prompt, system_prompt)

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        if self.is_ollama():
            # Ollama uses a top-level ``format`` field, not OpenAI's
            # ``response_format`` (which some proxies reject outright).
            if response_format == "json":
                kwargs["format"] = "json"
            kwargs.pop("response_format", None)
            kwargs.setdefault("api_base", settings.OLLAMA_API_BASE)
            kwargs["options"] = self._ollama_options()
        elif response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.time()
        raw = litellm.completion(**kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        text = raw["choices"][0]["message"]["content"] or ""
        usage_raw = dict(raw.get("usage", {}) or {})
        usage = {
            k: int(v)
            for k, v in usage_raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        logger.info(
            "LLM call model=%s latency=%dms tokens=%s",
            self.model,
            latency_ms,
            usage.get("total_tokens", "?"),
        )
        return text, usage, latency_ms

    def _cache_key(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
        response_format: Optional[str],
    ) -> str:
        payload = "||".join(
            [
                self.model,
                f"temp={self.temperature:.3f}",
                f"rf={response_format or ''}",
                "SYS:" + (system_prompt or ""),
                "USR:" + user_prompt,
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_read(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_write(self, key: str, payload: Dict[str, Any]) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("LLM cache write failed: %s", e)


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_json_loose(text: str) -> Any:
    stripped = _JSON_FENCE.sub("", text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first JSON-looking substring.
    for open_c, close_c in (("[", "]"), ("{", "}")):
        start = stripped.find(open_c)
        end = stripped.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            candidate = stripped[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not parse JSON from LLM output: {text[:200]!r}")
