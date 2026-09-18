
"""Groq AI Provider — Primary LLM Provider.

Integration with Groq via its official OpenAI-compatible REST API:
  Base URL: https://api.groq.com/openai/v1
  Auth: Bearer GROQ_API_KEY
  Default Model: openai/gpt-oss-120b
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage

logger = logging.getLogger("ai.groq_provider")

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqAIProvider(AIProvider):
    """
    LLM Provider backed by Groq's high-speed inference engine.

    Connects via Groq's official OpenAI-compatible endpoint. Supports
    fast open-weight models with sub-second latency and JSON object mode.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self._api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        self._base_url = (
            base_url
            or os.getenv("GROQ_BASE_URL")
            or DEFAULT_GROQ_BASE_URL
        )
        self._model = (
            model
            or os.getenv("GROQ_MODEL")
            or DEFAULT_GROQ_MODEL
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key == "your_groq_api_key_placeholder":
            raise RuntimeError(
                "GROQ_API_KEY is not configured. Obtain an API key from "
                "https://console.groq.com/keys and set it in backend/.env to enable Groq AI."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'openai' client package is required to connect to Groq's "
                "OpenAI-compatible endpoint. Install with: pip install openai"
            ) from exc

        return self._client

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> AITextResponse:
        client = self._get_client()
        last_error: Optional[Exception] = None

        kwargs: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self._max_retries + 1):
            try:
                from openai import APIConnectionError, InternalServerError, RateLimitError

                response = client.chat.completions.create(**kwargs)
                raw_text = response.choices[0].message.content or ""
                usage_raw = response.usage

                usage = TokenUsage(
                    prompt_tokens=usage_raw.prompt_tokens if usage_raw else 0,
                    completion_tokens=usage_raw.completion_tokens if usage_raw else 0,
                    total_tokens=usage_raw.total_tokens if usage_raw else 0,
                )

                logger.info(
                    "Groq text generated. model=%s tokens=%d",
                    self._model,
                    usage.total_tokens,
                )

                return AITextResponse(
                    text=raw_text,
                    model=self._model,
                    usage=usage,
                )

            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    logger.error("Groq max retries exceeded: %s", exc.__class__.__name__)
                    raise
                backoff = (2 ** attempt) * 1.5
                logger.warning(
                    "Retryable Groq error (%s). Backing off %.1fs (attempt %d/%d)...",
                    exc.__class__.__name__, backoff, attempt + 1, self._max_retries,
                )
                time.sleep(backoff)

            except Exception as exc:
                # Log without exposing any credentials
                logger.error("Groq call failed: %s", exc.__class__.__name__)
                raise

        if last_error:
            raise last_error

        return AITextResponse(text="", model=self._model)

    def classify_intent(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 400,
    ) -> Optional[Dict[str, Any]]:
        client = self._get_client()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Employee message: {text}"},
        ]

        intent_model = os.getenv("GROQ_INTENT_MODEL") or self._model
        if intent_model == "llama-3.1-8b-instant":
            intent_model = self._model

        try:
            response = client.chat.completions.create(
                model=intent_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max(max_tokens, 300),
            )
            raw = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            logger.warning("Groq returned non-JSON intent response: %r", raw[:100])
        except Exception as exc:
            logger.warning("Groq intent classification direct call failed: %s (%s)", exc.__class__.__name__, exc)

        # Fallback: try via generate_text
        try:
            resp = self.generate_text(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            json_match = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as exc:
            logger.error("Groq fallback intent classification failed: %s", exc)

        return None
