"""OpenAI AI Provider — Module 5 Extension.

Direct integration with the OpenAI Chat Completions API.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage

logger = logging.getLogger("ai.openai_provider")


class OpenAIProvider(AIProvider):
    """LLM Provider implementation backed by the official OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self._model = (
            model
            or os.getenv("OPENAI_CHAT_MODEL")
            or os.getenv("LLM_MODEL")
            or "gpt-4o-mini"
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key == "your_openai_api_key_placeholder":
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it in backend/.env to enable OpenAI."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
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

                return AITextResponse(
                    text=raw_text,
                    model=self._model,
                    usage=usage,
                )

            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    logger.error("OpenAI max retries exceeded: %s", exc)
                    raise
                backoff = (2 ** attempt) * 1.5
                logger.warning(
                    "Retryable OpenAI error (%s). Backing off %.1fs (attempt %d/%d)...",
                    exc.__class__.__name__, backoff, attempt + 1, self._max_retries,
                )
                time.sleep(backoff)

            except Exception as exc:
                logger.error("Non-retryable OpenAI error: %s", exc)
                raise

        if last_error:
            raise last_error

        return AITextResponse(text="", model=self._model)

    def classify_intent(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> Optional[Dict[str, Any]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Employee message: {text}"},
        ]
        try:
            resp = self.generate_text(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw = resp.text
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not json_match:
                logger.warning("OpenAI returned non-JSON intent response: %r", raw[:100])
                return None
            return json.loads(json_match.group())
        except Exception as exc:
            logger.warning("OpenAI intent classification failed: %s", exc)
            return None
