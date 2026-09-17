"""AI Provider Base Abstraction — Module 5 Extension.

Defines the common interface and response models for LLM operations across
different providers (Puter AI, OpenAI, Mock).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """Token consumption statistics for an AI completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AITextResponse:
    """Standardized text generation response."""

    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "usage": self.usage.to_dict(),
            "error": self.error,
        }


class AIProvider(ABC):
    """
    Abstract base class for LLM providers.

    Decouples the application (RAG answer generator, intent classifier)
    from specific backend AI services (Puter, OpenAI, Local, Mock).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier string (e.g. 'puter', 'openai', 'mock')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Configured model name for this provider."""
        pass

    @abstractmethod
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> AITextResponse:
        """
        Generates a text completion for the provided list of chat messages.

        Args:
            messages: List of message dicts: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Optional upper bound on completion tokens.

        Returns:
            AITextResponse with normalized content and token counts.
        """
        pass

    @abstractmethod
    def classify_intent(
        self,
        text: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a structured intent classification JSON dictionary.

        Args:
            text: The employee request message.
            system_prompt: Instructions defining the classification categories and output JSON.
            temperature: Low temperature for deterministic classification.
            max_tokens: Token limit for the JSON response.

        Returns:
            Parsed JSON dict if successful, or None if malformed/unparseable.
        """
        pass
