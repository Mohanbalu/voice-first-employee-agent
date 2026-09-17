"""AI Provider Layer — Module 5 Extension.

Provides a unified interface for LLM operations (Groq, Puter AI, OpenAI, Mock).
"""

from __future__ import annotations

from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage
from backend.app.ai.groq_provider import GroqAIProvider
from backend.app.ai.openai_provider import OpenAIProvider
from backend.app.ai.puter_provider import PuterAIProvider
from backend.app.ai.mock_provider import MockAIProvider
from backend.app.ai.factory import (
    get_ai_provider,
    set_global_provider,
    reset_global_provider,
)

__all__ = [
    "AIProvider",
    "AITextResponse",
    "TokenUsage",
    "GroqAIProvider",
    "OpenAIProvider",
    "PuterAIProvider",
    "MockAIProvider",
    "get_ai_provider",
    "set_global_provider",
    "reset_global_provider",
]
