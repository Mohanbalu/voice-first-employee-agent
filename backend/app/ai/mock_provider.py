"""Mock AI Provider — Module 5 Extension.

Provides deterministic responses for testing without external API calls.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage


class MockAIProvider(AIProvider):
    """Deterministic in-memory AI provider for unit and integration testing."""

    def __init__(
        self,
        canned_response: str = "Mock answer generated from company policy.",
        canned_intent: Optional[Dict[str, Any]] = None,
        model: str = "mock-llm-model",
        custom_generator: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ):
        self._canned_response = canned_response
        self._canned_intent = canned_intent or {
            "intent": "knowledge_query",
            "confidence": 0.92,
            "reasoning": "Mock classification",
        }
        self._model = model
        self._custom_generator = custom_generator
        self.call_count: int = 0
        self.last_messages: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return self._model

    def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> AITextResponse:
        self.call_count += 1
        self.last_messages = messages

        if self._custom_generator is not None:
            text = self._custom_generator(messages)
        else:
            text = self._canned_response

        return AITextResponse(
            text=text,
            model=self._model,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=25, total_tokens=75),
        )

    def classify_intent(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> Optional[Dict[str, Any]]:
        self.call_count += 1
        return self._canned_intent
