"""Text-to-Speech Module — Groq & Orpheus TTS (Module 6.3).

Converts agent text responses into natural speech audio:
  Model: canopylabs/orpheus-v1-english
  Voices: autumn, diana, hannah, austin, troy
  Format: wav (browser compatible)
  Provider abstraction supporting Mock (offline tests) and Groq (production).
"""

from __future__ import annotations

import io
import logging
import os
import struct
import wave
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger("voice.text_to_speech")

DEFAULT_TTS_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_TTS_VOICE = "autumn"
DEFAULT_TTS_FORMAT = "wav"
DEFAULT_TTS_MAX_TEXT_LENGTH = 5000
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SUPPORTED_TTS_FORMATS = {"wav", "mp3", "flac", "ogg", "mulaw"}
SUPPORTED_ORPHEUS_VOICES = {"autumn", "diana", "hannah", "austin", "troy"}


class TTSError(Exception):
    """Raised when Text-to-Speech synthesis fails."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TTSSynthesisResult:
    """Structured result containing synthesized audio bytes and metadata."""

    audio_bytes: bytes
    content_type: str = "audio/wav"
    format: str = "wav"
    duration: Optional[float] = None
    provider: str = "groq"
    model: str = DEFAULT_TTS_MODEL
    voice: str = DEFAULT_TTS_VOICE

    def to_dict(self) -> Dict[str, Any]:
        """Returns non-sensitive synthesis metadata dictionary."""
        return {
            "success": True,
            "content_type": self.content_type,
            "format": self.format,
            "size_bytes": len(self.audio_bytes),
            "duration": self.duration,
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
        }


@runtime_checkable
class TextToSpeechProvider(Protocol):
    """Abstract protocol for text-to-speech providers."""

    @property
    def model_name(self) -> str:
        """Returns active TTS model identifier."""
        ...

    @property
    def default_voice(self) -> str:
        """Returns the default voice identifier."""
        ...

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSSynthesisResult:
        """Synthesizes text into speech audio bytes."""
        ...


def generate_synthetic_wav_bytes(duration_seconds: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Generates deterministic, valid PCM WAV audio bytes in memory without external tools."""
    buffer = io.BytesIO()
    total_samples = int(sample_rate * duration_seconds)
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        # Generate silence / deterministic low-amplitude sample data
        pcm_data = struct.pack("<" + "h" * total_samples, *([0] * total_samples))
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


class GroqTextToSpeechProvider(TextToSpeechProvider):
    """Production Text-to-Speech provider backed by Groq Orpheus API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        format: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_text_length: Optional[int] = None,
    ):
        if api_key is not None:
            self._api_key = api_key
        else:
            try:
                from backend.app.config import config
                self._api_key = config.ai.groq_api_key or os.getenv("GROQ_API_KEY", "")
            except Exception:
                self._api_key = os.getenv("GROQ_API_KEY", "")

        if max_text_length is not None:
            self._max_text_length = max_text_length
        else:
            try:
                from backend.app.config import config
                self._max_text_length = config.tts.max_text_length or DEFAULT_TTS_MAX_TEXT_LENGTH
            except Exception:
                self._max_text_length = DEFAULT_TTS_MAX_TEXT_LENGTH

        try:
            from backend.app.config import config
            self._model = model or config.tts.model or DEFAULT_TTS_MODEL
            self._voice = voice or config.tts.voice or DEFAULT_TTS_VOICE
            self._format = (format or config.tts.format or DEFAULT_TTS_FORMAT).lower()
        except Exception:
            self._model = model or os.getenv("TTS_MODEL") or DEFAULT_TTS_MODEL
            self._voice = voice or os.getenv("TTS_VOICE") or DEFAULT_TTS_VOICE
            self._format = (format or os.getenv("TTS_FORMAT") or DEFAULT_TTS_FORMAT).lower()

        self._base_url = base_url or os.getenv("GROQ_BASE_URL") or DEFAULT_GROQ_BASE_URL
        self._timeout = timeout
        self._client: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def default_voice(self) -> str:
        return self._voice

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key == "your_groq_api_key_placeholder":
            raise TTSError(
                "GROQ_API_KEY is not configured. Set GROQ_API_KEY in backend/.env "
                "to enable Groq Text-to-Speech.",
                status_code=500,
            )

        try:
            from groq import Groq

            groq_url = self._base_url
            if groq_url and groq_url.rstrip("/").endswith("/openai/v1"):
                groq_url = groq_url.rstrip("/")[:-len("/openai/v1")]

            self._client = Groq(
                api_key=self._api_key,
                base_url=groq_url if groq_url else None,
                timeout=self._timeout,
            )
            return self._client
        except ImportError as exc:
            raise TTSError(
                "'groq' client package is not installed. Install with: pip install groq",
                status_code=500,
            ) from exc

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSSynthesisResult:
        """Synthesizes text to speech audio using Groq API."""
        # 1. Text input validation
        if not text or not text.strip():
            raise TTSError("Text content must not be empty or whitespace-only.", status_code=400)

        clean_text = text.strip()
        if len(clean_text) > self._max_text_length:
            raise TTSError(
                f"Text length ({len(clean_text)} characters) exceeds maximum allowed length of {self._max_text_length} characters.",
                status_code=413,
            )

        target_model = model or self._model
        target_voice = voice or self._voice
        target_format = (response_format or self._format).lower()

        if target_format not in SUPPORTED_TTS_FORMATS:
            raise TTSError(
                f"Unsupported audio format: '{target_format}'. Supported formats: {sorted(SUPPORTED_TTS_FORMATS)}",
                status_code=422,
            )

        client = self._get_client()

        create_kwargs: Dict[str, Any] = {
            "model": target_model,
            "input": clean_text,
            "voice": target_voice,
            "response_format": target_format,
        }
        if speed is not None:
            create_kwargs["speed"] = speed

        try:
            logger.info("Sending TTS request to Groq (model=%s, voice=%s, format=%s)", target_model, target_voice, target_format)
            response = client.audio.speech.create(**create_kwargs)

            # Extract raw binary content from BinaryAPIResponse
            if hasattr(response, "read"):
                audio_bytes = response.read()
            elif hasattr(response, "content"):
                audio_bytes = response.content
            elif isinstance(response, bytes):
                audio_bytes = response
            else:
                audio_bytes = bytes(response)

            if not audio_bytes:
                raise TTSError("Groq TTS returned an empty audio response.", status_code=502)

            return TTSSynthesisResult(
                audio_bytes=audio_bytes,
                content_type=f"audio/{target_format}",
                format=target_format,
                duration=None,
                provider="groq",
                model=target_model,
                voice=target_voice,
            )

        except TTSError:
            raise
        except Exception as exc:
            error_str = str(exc)
            logger.error("Groq TTS synthesis failed: %s", error_str)

            # Auto-recovery for Groq free-tier TPM limit (1,200 TPM): retry with concise spoken passage
            if ("Limit 1200" in error_str or "Request too large" in error_str or "rate_limit_exceeded" in error_str) and len(clean_text) > 350:
                logger.warning("Groq TTS TPM limit hit (%d chars). Truncating to concise spoken text and retrying...", len(clean_text))
                truncated_text = clean_text[:450].rsplit(". ", 1)[0]
                if len(truncated_text) < 100:
                    truncated_text = clean_text[:350]
                if not truncated_text.endswith("."):
                    truncated_text += "."
                create_kwargs["input"] = truncated_text
                try:
                    retry_resp = client.audio.speech.create(**create_kwargs)
                    retry_bytes = retry_resp.read() if hasattr(retry_resp, "read") else (retry_resp.content if hasattr(retry_resp, "content") else bytes(retry_resp))
                    if retry_bytes:
                        logger.info("Groq TTS retry succeeded with %d bytes of audio", len(retry_bytes))
                        return TTSSynthesisResult(
                            audio_bytes=retry_bytes,
                            content_type=f"audio/{target_format}",
                            format=target_format,
                            duration=None,
                            provider="groq",
                            model=target_model,
                            voice=target_voice,
                        )
                except Exception as retry_err:
                    logger.error("Groq TTS auto-recovery retry failed: %s", retry_err)

            # Scrub API key from error strings
            scrubbed = error_str
            if self._api_key and len(self._api_key) > 8:
                scrubbed = scrubbed.replace(self._api_key, "gsk_***")

            if "model_terms_required" in error_str:
                raise TTSError(
                    f"The Groq TTS model '{target_model}' requires terms acceptance in the Groq console. "
                    f"Please have the org admin accept terms at: https://console.groq.com/playground?model={target_model.replace('/', '%2F')}",
                    status_code=502,
                ) from exc

            if "model_decommissioned" in error_str:
                raise TTSError(
                    f"The configured Groq TTS model '{target_model}' has been decommissioned. "
                    f"Please configure a supported model such as 'canopylabs/orpheus-v1-english'.",
                    status_code=502,
                ) from exc

            if "rate limit" in error_str.lower() or "429" in error_str:
                raise TTSError(f"Groq TTS rate limit exceeded: {scrubbed}", status_code=429) from exc

            if "authentication" in error_str.lower() or "401" in error_str or "unauthorized" in error_str.lower():
                raise TTSError("Groq TTS authentication failed. Check GROQ_API_KEY.", status_code=502) from exc

            raise TTSError(f"Groq TTS provider error: {scrubbed}", status_code=502) from exc


class MockTextToSpeechProvider(TextToSpeechProvider):
    """Deterministic Mock TTS provider generating valid PCM WAV audio for fast offline tests."""

    def __init__(
        self,
        mock_audio_bytes: Optional[bytes] = None,
        model: str = "mock-orpheus-tts",
        voice: str = DEFAULT_TTS_VOICE,
        format: str = "wav",
        should_fail: bool = False,
        max_text_length: Optional[int] = None,
    ):
        self._mock_audio_bytes = mock_audio_bytes or generate_synthetic_wav_bytes(duration_seconds=0.2)
        self._model = model
        self._voice = voice
        self._format = format
        self._should_fail = should_fail
        if max_text_length is not None:
            self._max_text_length = max_text_length
        else:
            try:
                from backend.app.config import config
                self._max_text_length = config.tts.max_text_length or DEFAULT_TTS_MAX_TEXT_LENGTH
            except Exception:
                self._max_text_length = DEFAULT_TTS_MAX_TEXT_LENGTH

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def default_voice(self) -> str:
        return self._voice

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSSynthesisResult:
        if self._should_fail:
            raise TTSError("Mock TTS provider configured failure", status_code=502)

        if not text or not text.strip():
            raise TTSError("Text content must not be empty or whitespace-only.", status_code=400)

        clean_text = text.strip()
        if len(clean_text) > self._max_text_length:
            raise TTSError(
                f"Text length ({len(clean_text)} characters) exceeds maximum allowed length of {self._max_text_length} characters.",
                status_code=413,
            )

        target_format = (response_format or self._format).lower()
        if target_format not in SUPPORTED_TTS_FORMATS:
            raise TTSError(
                f"Unsupported audio format: '{target_format}'. Supported formats: {sorted(SUPPORTED_TTS_FORMATS)}",
                status_code=422,
            )

        return TTSSynthesisResult(
            audio_bytes=self._mock_audio_bytes,
            content_type=f"audio/{target_format}",
            format=target_format,
            duration=0.2,
            provider="mock",
            model=model or self._model,
            voice=voice or self._voice,
        )


def get_tts_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    voice: Optional[str] = None,
) -> TextToSpeechProvider:
    """Factory resolver for text-to-speech providers."""
    try:
        from backend.app.config import config
        resolved_type = (provider_type or config.tts.provider or "groq").strip().lower()
    except Exception:
        resolved_type = (provider_type or os.getenv("TTS_PROVIDER") or "groq").strip().lower()

    if resolved_type == "mock":
        return MockTextToSpeechProvider()

    if resolved_type == "groq":
        return GroqTextToSpeechProvider(api_key=api_key, model=model, voice=voice)

    logger.warning("Unrecognized TTS provider '%s', defaulting to Groq", resolved_type)
    return GroqTextToSpeechProvider(api_key=api_key, model=model, voice=voice)
