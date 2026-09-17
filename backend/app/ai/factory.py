"""AI Provider Factory — Module 5 Extension.

Factory function to instantiate and retrieve configured AI providers.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    from backend.app.config import config
except ImportError:
    from app.config import config

from backend.app.ai.provider import AIProvider
from backend.app.ai.groq_provider import GroqAIProvider
from backend.app.ai.openai_provider import OpenAIProvider
from backend.app.ai.puter_provider import PuterAIProvider
from backend.app.ai.mock_provider import MockAIProvider

logger = logging.getLogger("ai.factory")

_GLOBAL_OVERRIDE_PROVIDER: Optional[AIProvider] = None


def set_global_provider(provider: Optional[AIProvider]) -> None:
    """Sets a global AI provider instance (useful for unit tests)."""
    global _GLOBAL_OVERRIDE_PROVIDER
    _GLOBAL_OVERRIDE_PROVIDER = provider


def reset_global_provider() -> None:
    """Clears any test provider override."""
    global _GLOBAL_OVERRIDE_PROVIDER
    _GLOBAL_OVERRIDE_PROVIDER = None


def get_ai_provider(
    provider_type: Optional[str] = None,
    **kwargs,
) -> AIProvider:
    """
    Returns an AIProvider instance according to application configuration.

    Resolution order:
      1. Injected global override (if set via set_global_provider)
      2. Explicit provider_type argument
      3. AI_PROVIDER environment variable
      4. config.ai.provider
      5. Default fallback to "groq"
    """
    if _GLOBAL_OVERRIDE_PROVIDER is not None and provider_type is None:
        return _GLOBAL_OVERRIDE_PROVIDER

    target = (
        provider_type
        or os.getenv("AI_PROVIDER")
        or (config.ai.provider if hasattr(config, "ai") else None)
        or "groq"
    ).strip().lower()

    if target == "groq":
        logger.info("Instantiating Groq AI Provider")
        return GroqAIProvider(**kwargs)

    if target == "puter":
        logger.info("Instantiating Puter AI Provider")
        return PuterAIProvider(**kwargs)

    if target == "openai":
        logger.info("Instantiating OpenAI Provider")
        return OpenAIProvider(**kwargs)

    if target == "mock":
        logger.info("Instantiating Mock AI Provider")
        return MockAIProvider(**kwargs)

    logger.warning("Unrecognized AI_PROVIDER '%s' — falling back to OpenAI", target)
    return OpenAIProvider(**kwargs)
