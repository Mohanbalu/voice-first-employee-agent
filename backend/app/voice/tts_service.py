"""TTS Service — Module 6.3.

High-level business service coordinating Text-to-Speech synthesis, input validation,
provider execution, and error handling.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from backend.app.config import config
    from backend.app.schemas.voice import SynthesizeRequest
    from backend.app.voice.text_to_speech import (
        DEFAULT_TTS_MAX_TEXT_LENGTH,
        SUPPORTED_TTS_FORMATS,
        TTSError,
        TTSSynthesisResult,
        TextToSpeechProvider,
        get_tts_provider,
    )
except ImportError:
    from app.config import config
    from app.schemas.voice import SynthesizeRequest
    from app.voice.text_to_speech import (
        DEFAULT_TTS_MAX_TEXT_LENGTH,
        SUPPORTED_TTS_FORMATS,
        TTSError,
        TTSSynthesisResult,
        TextToSpeechProvider,
        get_tts_provider,
    )

logger = logging.getLogger("voice.tts_service")


class TTSService:
    """Service layer for speech synthesis operations."""

    def __init__(
        self,
        provider: Optional[TextToSpeechProvider] = None,
        max_text_length: Optional[int] = None,
    ):
        self._provider = provider or get_tts_provider()
        try:
            self._max_text_length = max_text_length or config.tts.max_text_length or DEFAULT_TTS_MAX_TEXT_LENGTH
        except Exception:
            self._max_text_length = max_text_length or DEFAULT_TTS_MAX_TEXT_LENGTH

    @property
    def provider(self) -> TextToSpeechProvider:
        return self._provider

    def synthesize_speech(self, request: SynthesizeRequest) -> TTSSynthesisResult:
        """Validates synthesis request and delegates to configured TTS provider."""
        # 1. Text validation
        if not request.text or not request.text.strip():
            raise TTSError("Text content must not be empty or whitespace-only.", status_code=400)

        cleaned_text = request.text.strip()
        if len(cleaned_text) > self._max_text_length:
            raise TTSError(
                f"Text length ({len(cleaned_text)} characters) exceeds maximum limit of {self._max_text_length} characters.",
                status_code=413,
            )

        # 2. Format validation if provided
        if request.format:
            fmt = request.format.strip().lower()
            if fmt not in SUPPORTED_TTS_FORMATS:
                raise TTSError(
                    f"Unsupported audio format: '{fmt}'. Supported formats: {sorted(SUPPORTED_TTS_FORMATS)}",
                    status_code=422,
                )

        # 3. Call provider
        logger.info(
            "Synthesizing text (%d chars, model=%s, voice=%s, format=%s)",
            len(cleaned_text),
            request.model or getattr(self._provider, "model_name", "default"),
            request.voice or getattr(self._provider, "default_voice", "default"),
            request.format or "default",
        )

        try:
            result = self._provider.synthesize(
                text=cleaned_text,
                voice=request.voice,
                model=request.model,
                response_format=request.format,
                speed=request.speed,
            )
            return result
        except TTSError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error in TTSService during speech synthesis: %s", exc)
            raise TTSError(f"Speech synthesis failed: {exc}", status_code=502) from exc
