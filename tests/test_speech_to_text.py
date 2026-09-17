"""Unit tests for Speech-to-Text Module 6.1 (Groq Whisper Large V3 Turbo).

Verifies:
1. Successful transcription via provider and FastAPI endpoint.
2. Empty transcript handling.
3. Unsupported file extension rejection (415).
4. Oversized file rejection (413).
5. Empty audio file rejection (400).
6. Upstream provider failure handling (502).
7. Missing API key handling (500).
8. Language parameter propagation.
9. Prompt context parameter propagation.
10. Response schema validation.
11. Temporary file cleanup guaranteed even on provider failure.
12. MockSpeechToTextProvider deterministic output.
13. Provider factory resolution (groq, mock, fallback).
14. Zero credential leakage in repr or error messages.
"""

import io
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.voice import TranscriptionResponse
from backend.app.voice.speech_to_text import (
    DEFAULT_STT_MODEL,
    GroqSpeechToTextProvider,
    MockSpeechToTextProvider,
    STTError,
    STTTranscriptionResult,
    get_stt_provider,
)
from backend.app.routes.voice import set_stt_provider


@pytest.fixture(autouse=True)
def reset_voice_provider():
    """Ensures each test gets a clean STT provider environment."""
    set_stt_provider(None)
    yield
    set_stt_provider(None)


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


# ── Provider Unit Tests ────────────────────────────────────────────────────────


class TestGroqSpeechToTextProvider:
    """Tests for GroqSpeechToTextProvider without external network calls."""

    def test_provider_initialization_defaults(self):
        provider = GroqSpeechToTextProvider(api_key="gsk_test1234567890", model="whisper-large-v3-turbo")
        assert provider.model_name == "whisper-large-v3-turbo"

    def test_missing_api_key_raises_error(self):
        provider = GroqSpeechToTextProvider(api_key="")
        with pytest.raises(STTError) as exc_info:
            provider.transcribe(io.BytesIO(b"fake_audio"), filename="test.wav")
        assert exc_info.value.status_code == 500
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)

    def test_placeholder_api_key_raises_error(self):
        provider = GroqSpeechToTextProvider(api_key="your_groq_api_key_placeholder")
        with pytest.raises(STTError) as exc_info:
            provider.transcribe(io.BytesIO(b"fake_audio"), filename="test.wav")
        assert exc_info.value.status_code == 500

    def test_unsupported_audio_extension_raises_415(self):
        provider = GroqSpeechToTextProvider(api_key="gsk_valid")
        with pytest.raises(STTError) as exc_info:
            provider.transcribe(io.BytesIO(b"data"), filename="document.pdf")
        assert exc_info.value.status_code == 415
        assert "Unsupported audio format" in str(exc_info.value)

    @patch("backend.app.voice.speech_to_text.GroqSpeechToTextProvider._get_client")
    def test_successful_transcription_mocked_client(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello, this is a test audio transcription."
        mock_response.language = "en"
        mock_response.duration = 4.2
        mock_response.segments = [{"id": 0, "start": 0.0, "end": 4.2, "text": "Hello, this is a test audio transcription."}]
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        provider = GroqSpeechToTextProvider(api_key="gsk_valid")
        result = provider.transcribe(io.BytesIO(b"RIFFdummy"), filename="sample.wav", language="en", prompt="Employee query")

        assert isinstance(result, STTTranscriptionResult)
        assert result.text == "Hello, this is a test audio transcription."
        assert result.language == "en"
        assert result.duration == 4.2
        assert len(result.segments) == 1

        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["prompt"] == "Employee query"
        assert call_kwargs["model"] == DEFAULT_STT_MODEL

    @patch("backend.app.voice.speech_to_text.GroqSpeechToTextProvider._get_client")
    def test_provider_handles_rate_limit(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("Rate limit reached (429)")
        mock_get_client.return_value = mock_client

        provider = GroqSpeechToTextProvider(api_key="gsk_valid")
        with pytest.raises(STTError) as exc_info:
            provider.transcribe(io.BytesIO(b"audio"), filename="test.mp3")
        assert exc_info.value.status_code == 429

    @patch("backend.app.voice.speech_to_text.GroqSpeechToTextProvider._get_client")
    def test_zero_api_key_leakage_in_exception(self, mock_get_client):
        secret_key = "gsk_super_secret_key_123456789"
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = RuntimeError(f"Error with key {secret_key}")
        mock_get_client.return_value = mock_client

        provider = GroqSpeechToTextProvider(api_key=secret_key)
        with pytest.raises(STTError) as exc_info:
            provider.transcribe(io.BytesIO(b"audio"), filename="test.wav")
        assert secret_key not in str(exc_info.value)
        assert "gsk_***" in str(exc_info.value)


class TestMockSpeechToTextProvider:
    """Tests for the offline mock STT provider."""

    def test_mock_transcription(self):
        mock = MockSpeechToTextProvider(mock_transcript="Testing audio", mock_language="fr")
        res = mock.transcribe(io.BytesIO(b"audio"), filename="test.wav")
        assert res.text == "Testing audio"
        assert res.language == "fr"

    def test_mock_transcription_prompt_override(self):
        mock = MockSpeechToTextProvider()
        res = mock.transcribe(io.BytesIO(b"audio"), filename="test.wav", prompt="custom: Dynamic output")
        assert res.text == "Dynamic output"

    def test_mock_provider_failure_flag(self):
        mock = MockSpeechToTextProvider(should_fail=True)
        with pytest.raises(STTError) as exc_info:
            mock.transcribe(io.BytesIO(b"audio"), filename="test.wav")
        assert exc_info.value.status_code == 502


class TestSTTFactory:
    """Tests for the STT provider factory."""

    def test_factory_resolves_mock(self):
        provider = get_stt_provider("mock")
        assert isinstance(provider, MockSpeechToTextProvider)

    def test_factory_resolves_groq(self):
        provider = get_stt_provider("groq")
        assert isinstance(provider, GroqSpeechToTextProvider)

    def test_factory_fallback_to_groq(self):
        provider = get_stt_provider("unknown_provider")
        assert isinstance(provider, GroqSpeechToTextProvider)


# ── FastAPI Endpoint Unit Tests ────────────────────────────────────────────────


class TestTranscribeEndpoint:
    """Tests for POST /api/voice/transcribe."""

    def test_successful_transcribe_endpoint(self, client):
        mock_provider = MockSpeechToTextProvider(mock_transcript="What is the sick leave policy?")
        set_stt_provider(mock_provider)

        audio_bytes = b"RIFFfake_wav_content_for_test"
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("policy_query.wav", audio_bytes, "audio/wav")},
            data={"language": "en", "prompt": "Policy inquiry"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == "What is the sick leave policy?"
        assert data["language"] == "en"
        assert "duration" in data

    def test_empty_transcript_handled_cleanly(self, client):
        mock_provider = MockSpeechToTextProvider(mock_transcript="")
        set_stt_provider(mock_provider)

        audio_bytes = b"RIFFfake_wav_content_silence"
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("silence.wav", audio_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["text"] == ""

    def test_unsupported_file_extension_returns_415(self, client):
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("malicious.exe", b"binarycontent", "application/octet-stream")},
        )
        assert response.status_code == 415
        assert "Unsupported audio format" in response.json()["detail"]

    def test_unsupported_text_file_returns_415(self, client):
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("notes.txt", b"plain text", "text/plain")},
        )
        assert response.status_code == 415

    def test_empty_audio_file_returns_400(self, client):
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "Audio file is empty" in response.json()["detail"]

    def test_oversized_audio_file_returns_413(self, client):
        # Limit is 25MB
        with patch("backend.app.config.config.stt.max_upload_size_mb", 1):
            large_audio = b"0" * (2 * 1024 * 1024)  # 2MB > 1MB
            response = client.post(
                "/api/voice/transcribe",
                files={"audio": ("large.wav", large_audio, "audio/wav")},
            )
            assert response.status_code == 413
            assert "exceeds maximum limit" in response.json()["detail"]

    def test_provider_failure_returns_502(self, client):
        mock_provider = MockSpeechToTextProvider(should_fail=True)
        set_stt_provider(mock_provider)

        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("query.wav", b"RIFFdummydata", "audio/wav")},
        )
        assert response.status_code == 502
        assert "Mock STT provider configured failure" in response.json()["detail"]

    def test_temporary_file_cleanup_on_success_and_failure(self, client):
        """Verifies that no temporary files are left behind on disk."""
        mock_provider = MockSpeechToTextProvider(should_fail=False)
        set_stt_provider(mock_provider)

        import tempfile
        temp_dir = Path(tempfile.gettempdir())
        before_wavs = set(temp_dir.glob("*.wav"))

        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("clean_test.wav", b"RIFFtempcheck", "audio/wav")},
        )
        assert response.status_code == 200

        after_wavs = set(temp_dir.glob("*.wav"))
        # Any temporary file created for clean_test.wav must have been deleted
        new_wavs = after_wavs - before_wavs
        assert len(new_wavs) == 0

        # Now test failure condition cleanup
        mock_provider._should_fail = True
        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("fail_test.wav", b"RIFFfailcheck", "audio/wav")},
        )
        assert response.status_code == 502

        after_fail_wavs = set(temp_dir.glob("*.wav"))
        new_fail_wavs = after_fail_wavs - before_wavs
        assert len(new_fail_wavs) == 0

    def test_health_probe_updated_for_module_6_1(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "6.1" in response.json()["module"]
