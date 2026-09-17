"""Puter AI Provider — Module 5 Extension.

Integration with Puter AI via its official OpenAI-compatible REST API:
  Base URL: https://api.puter.com/puterai/openai/v1/
  Auth: Puter Auth Token (Bearer token from https://puter.com/dashboard)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage

logger = logging.getLogger("ai.puter_provider")

DEFAULT_PUTER_BASE_URL = "https://api.puter.com/puterai/openai/v1/"
DEFAULT_PUTER_MODEL = "gpt-4o-mini"


class PuterAIProvider(AIProvider):
    """
    LLM Provider backed by Puter AI.

    Connects to Puter's official OpenAI-compatible gateway. Supports hundreds
    of models (GPT, Claude, Gemini, etc.) routed via Puter without individual
    provider API keys.
    """

    def __init__(
        self,
        auth_token: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self._auth_token = auth_token if auth_token is not None else os.getenv("PUTER_AUTH_TOKEN", "")
        self._base_url = (
            base_url
            or os.getenv("PUTER_BASE_URL")
            or DEFAULT_PUTER_BASE_URL
        )
        self._model = (
            model
            or os.getenv("PUTER_MODEL")
            or os.getenv("LLM_MODEL")
            or DEFAULT_PUTER_MODEL
        )
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[Any] = None

    @property
    def name(self) -> str:
        return "puter"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._auth_token or self._auth_token == "your_puter_auth_token_placeholder":
            raise RuntimeError(
                "PUTER_AUTH_TOKEN is not configured. Obtain your auth token from "
                "https://puter.com/dashboard and set it in backend/.env to enable Puter AI."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._auth_token,
                timeout=self._timeout,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'openai' client package is required to connect to Puter's "
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
                    "Puter AI text generated successfully. model=%s tokens=%d",
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
                    logger.error("Puter AI max retries exceeded: %s", exc.__class__.__name__)
                    raise
                backoff = (2 ** attempt) * 1.5
                logger.warning(
                    "Retryable Puter AI error (%s). Backing off %.1fs (attempt %d/%d)...",
                    exc.__class__.__name__, backoff, attempt + 1, self._max_retries,
                )
                time.sleep(backoff)

            except Exception as exc:
                # Log without exposing any credentials
                logger.error("Puter AI call failed: %s", exc.__class__.__name__)
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
                logger.warning("Puter AI returned non-JSON intent response: %r", raw[:100])
                return None
            return json.loads(json_match.group())
        except Exception as exc:
            logger.warning("Puter AI intent classification failed: %s", exc.__class__.__name__)
            return None
