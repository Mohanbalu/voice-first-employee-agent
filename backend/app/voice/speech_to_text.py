"""Speech-to-Text Module — Groq Whisper Large V3 Turbo (Module 6.1).

Converts voice recordings into text transcripts using Groq's high-speed Whisper API:
  Base URL: https://api.groq.com/openai/v1/audio/transcriptions
  Model: whisper-large-v3-turbo
  Auth: Bearer GROQ_API_KEY
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Protocol, Union, runtime_checkable

logger = logging.getLogger("voice.speech_to_text")

DEFAULT_STT_MODEL = "whisper-large-v3-turbo"
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

SUPPORTED_AUDIO_EXTENSIONS = {
    "wav",
    "mp3",
    "m4a",
    "ogg",
    "webm",
    "mp4",
    "flac",
    "mpeg",
    "mpga",
}


class STTError(Exception):
    """Raised when Speech-to-Text transcription fails."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class STTTranscriptionResult:
    """Internal transcription result data structure."""

    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "success": True,
            "text": self.text,
        }
        if self.language is not None:
            res["language"] = self.language
        if self.duration is not None:
            res["duration"] = self.duration
        if self.segments is not None:
            res["segments"] = self.segments
        return res


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """Abstract protocol for speech-to-text providers."""

    @property
    def model_name(self) -> str:
        """Returns active STT model name."""
        ...

    def transcribe(
        self,
        audio_file: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> STTTranscriptionResult:
        """Transcribes audio file to text."""
        ...


class GroqSpeechToTextProvider(SpeechToTextProvider):
    """Production Speech-to-Text provider backed by Groq Whisper Large V3 Turbo."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        # Resolve API key from argument, config, or environment
        if api_key is not None:
            self._api_key = api_key
        else:
            try:
                from backend.app.config import config
                self._api_key = config.ai.groq_api_key or os.getenv("GROQ_API_KEY", "")
            except Exception:
                self._api_key = os.getenv("GROQ_API_KEY", "")

        self._model = model or os.getenv("STT_MODEL") or DEFAULT_STT_MODEL
        self._base_url = (
            base_url
            or os.getenv("GROQ_BASE_URL")
            or DEFAULT_GROQ_BASE_URL
        )
        self._timeout = timeout
        self._client: Optional[Any] = None

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key == "your_groq_api_key_placeholder":
            raise STTError(
                "GROQ_API_KEY is not configured. Set GROQ_API_KEY in backend/.env "
                "to enable Groq Speech-to-Text.",
                status_code=500,
            )

        # First attempt official groq SDK, then fallback to openai SDK
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
        except ImportError:
            pass

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
            return self._client
        except ImportError as exc:
            raise STTError(
                "Neither 'groq' nor 'openai' client package is installed. "
                "Install with: pip install groq",
                status_code=500,
            ) from exc

    def transcribe(
        self,
        audio_file: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> STTTranscriptionResult:
        """Transcribes audio using Groq Whisper.

        Accepts filesystem paths, Path objects, or binary file-like objects.
        """
        client = self._get_client()

        # Handle path vs file-like
        opened_file: Optional[BinaryIO] = None
        file_payload: Any = None

        try:
            if isinstance(audio_file, (str, Path)):
                file_path = Path(audio_file)
                if not file_path.exists():
                    raise STTError(f"Audio file not found: {file_path}", status_code=400)
                fname = filename or file_path.name
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in SUPPORTED_AUDIO_EXTENSIONS:
                    raise STTError(
                        f"Unsupported audio format: '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}",
                        status_code=415,
                    )
                opened_file = open(file_path, "rb")
                file_payload = (fname, opened_file)
            else:
                # File-like object
                fname = filename or "audio.wav"
                ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
                if ext not in SUPPORTED_AUDIO_EXTENSIONS:
                    raise STTError(
                        f"Unsupported audio format: '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}",
                        status_code=415,
                    )
                file_payload = (fname, audio_file)

            transcribe_kwargs: Dict[str, Any] = {
                "model": self._model,
                "file": file_payload,
                "temperature": temperature,
                "response_format": response_format,
            }
            if language:
                transcribe_kwargs["language"] = language
            if prompt:
                transcribe_kwargs["prompt"] = prompt

            logger.info("Sending audio transcription request to Groq STT (model=%s)", self._model)
            response = client.audio.transcriptions.create(**transcribe_kwargs)

            # Process response format
            if isinstance(response, str):
                return STTTranscriptionResult(text=response.strip())

            # Attribute or dict-based response
            text_val = getattr(response, "text", "") or ""
            if not text_val and isinstance(response, dict):
                text_val = response.get("text", "")

            lang_val = getattr(response, "language", None)
            duration_val = getattr(response, "duration", None)
            segments_val = getattr(response, "segments", None)

            return STTTranscriptionResult(
                text=text_val.strip(),
                language=lang_val,
                duration=duration_val,
                segments=segments_val if segments_val else [],
            )

        except STTError:
            raise
        except Exception as exc:
            error_str = str(exc)
            logger.error("Groq STT transcription failed: %s", error_str)
            # Scrub API keys or Authorization headers from message
            scrubbed = error_str
            if self._api_key and len(self._api_key) > 8:
                scrubbed = scrubbed.replace(self._api_key, "gsk_***")

            if "rate limit" in error_str.lower() or "429" in error_str:
                raise STTError(f"Groq STT rate limit exceeded: {scrubbed}", status_code=429) from exc
            if "authentication" in error_str.lower() or "401" in error_str or "unauthorized" in error_str.lower():
                raise STTError("Groq STT authentication failed. Check GROQ_API_KEY.", status_code=502) from exc
            raise STTError(f"Groq STT provider error: {scrubbed}", status_code=502) from exc

        finally:
            if opened_file is not None and not opened_file.closed:
                opened_file.close()


class MockSpeechToTextProvider(SpeechToTextProvider):
    """Deterministic Mock STT provider for fast offline unit tests."""

    def __init__(
        self,
        mock_transcript: str = "What is the annual leave and sick leave policy?",
        mock_language: str = "en",
        mock_duration: float = 3.5,
        should_fail: bool = False,
    ):
        self._model = "mock-whisper-turbo"
        self._mock_transcript = mock_transcript
        self._mock_language = mock_language
        self._mock_duration = mock_duration
        self._should_fail = should_fail

    @property
    def model_name(self) -> str:
        return self._model

    def transcribe(
        self,
        audio_file: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
    ) -> STTTranscriptionResult:
        if self._should_fail:
            raise STTError("Mock STT provider configured failure", status_code=502)

        fname = filename or (Path(audio_file).name if isinstance(audio_file, (str, Path)) else "audio.wav")
        ext = fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext and ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise STTError(
                f"Unsupported audio format: '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}",
                status_code=415,
            )

        # Optional customization via prompt/language for testing
        out_text = self._mock_transcript
        if prompt and "custom:" in prompt.lower():
            out_text = prompt.split("custom:", 1)[-1].strip()

        return STTTranscriptionResult(
            text=out_text,
            language=language or self._mock_language,
            duration=self._mock_duration,
            segments=[],
        )


def get_stt_provider(
    provider_type: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> SpeechToTextProvider:
    """Factory resolver for speech-to-text providers."""
    try:
        from backend.app.config import config
        resolved_type = (provider_type or config.stt.provider or "groq").strip().lower()
    except Exception:
        resolved_type = (provider_type or os.getenv("STT_PROVIDER") or "groq").strip().lower()

    if resolved_type == "mock":
        return MockSpeechToTextProvider()

    if resolved_type == "groq":
        return GroqSpeechToTextProvider(api_key=api_key, model=model)

    logger.warning("Unrecognized STT provider '%s', defaulting to Groq", resolved_type)
    return GroqSpeechToTextProvider(api_key=api_key, model=model)
