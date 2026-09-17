"""Unit tests for Text-to-Speech Subsystem — Module 6.3.

All tests run strictly offline with zero external network calls and zero credit consumption.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.routes.voice import set_tts_service
from backend.app.schemas.voice import SynthesizeRequest
from backend.app.voice.text_to_speech import (
    DEFAULT_TTS_FORMAT,
    DEFAULT_TTS_MAX_TEXT_LENGTH,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    GroqTextToSpeechProvider,
    MockTextToSpeechProvider,
    TTSError,
    TTSSynthesisResult,
    generate_synthetic_wav_bytes,
    get_tts_provider,
)
from backend.app.voice.tts_service import TTSService


@pytest.fixture(autouse=True)
def reset_dependencies():
    """Ensure clean dependency injection state before and after each test."""
    set_tts_service(None)
    yield
    set_tts_service(None)


# ===========================================================================
# 1. GroqTextToSpeechProvider Offline Tests
# ===========================================================================

class TestGroqTextToSpeechProvider:
    """Offline unit tests for GroqTextToSpeechProvider."""

    def test_provider_initialization_defaults(self):
        provider = GroqTextToSpeechProvider(
            api_key="gsk_test1234567890abcdef",
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
        )
        assert provider.model_name == "canopylabs/orpheus-v1-english"
        assert provider.default_voice == "autumn"

    def test_missing_api_key_raises_error(self):
        provider = GroqTextToSpeechProvider(api_key="")
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Hello world")
        assert exc_info.value.status_code == 500
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)

    def test_placeholder_api_key_raises_error(self):
        provider = GroqTextToSpeechProvider(api_key="your_groq_api_key_placeholder")
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Hello world")
        assert exc_info.value.status_code == 500
        assert "GROQ_API_KEY is not configured" in str(exc_info.value)

    def test_empty_text_raises_400(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("")
        assert exc_info.value.status_code == 400
        assert "must not be empty or whitespace-only" in str(exc_info.value)

    def test_whitespace_text_raises_400(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("    \n\t   ")
        assert exc_info.value.status_code == 400
        assert "must not be empty or whitespace-only" in str(exc_info.value)

    def test_oversized_text_raises_413(self):
        provider = GroqTextToSpeechProvider(
            api_key="gsk_test1234567890abcdef",
            max_text_length=50,
        )
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("A" * 51)
        assert exc_info.value.status_code == 413
        assert "exceeds maximum allowed length" in str(exc_info.value)

    def test_unsupported_format_raises_422(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Hello world", response_format="invalid_codec")
        assert exc_info.value.status_code == 422
        assert "Unsupported audio format" in str(exc_info.value)

    def test_successful_synthesis_mocked_client(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_wav_bytes = generate_synthetic_wav_bytes(duration_seconds=0.1)
        mock_response.read.return_value = mock_wav_bytes
        mock_client.audio.speech.create.return_value = mock_response

        provider._client = mock_client

        result = provider.synthesize(
            text="Hello from workplace assistant.",
            voice="troy",
            model="canopylabs/orpheus-v1-english",
            response_format="wav",
        )

        assert isinstance(result, TTSSynthesisResult)
        assert result.audio_bytes == mock_wav_bytes
        assert result.content_type == "audio/wav"
        assert result.format == "wav"
        assert result.voice == "troy"
        assert result.model == "canopylabs/orpheus-v1-english"
        assert result.provider == "groq"
        mock_client.audio.speech.create.assert_called_once_with(
            model="canopylabs/orpheus-v1-english",
            input="Hello from workplace assistant.",
            voice="troy",
            response_format="wav",
        )

    def test_provider_handles_rate_limit_429(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        mock_client = MagicMock()
        mock_client.audio.speech.create.side_effect = Exception("Rate limit reached 429: Too Many Requests")
        provider._client = mock_client

        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Test rate limit")
        assert exc_info.value.status_code == 429
        assert "rate limit exceeded" in str(exc_info.value).lower()

    def test_provider_handles_model_terms_required_502(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        mock_client = MagicMock()
        mock_client.audio.speech.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'The model canopylabs/orpheus-v1-english requires terms acceptance', 'code': 'model_terms_required'}}"
        )
        provider._client = mock_client

        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Test terms acceptance")
        assert exc_info.value.status_code == 502
        assert "requires terms acceptance" in str(exc_info.value)
        assert "https://console.groq.com/playground" in str(exc_info.value)

    def test_provider_handles_model_decommissioned_502(self):
        provider = GroqTextToSpeechProvider(api_key="gsk_test1234567890abcdef")
        mock_client = MagicMock()
        mock_client.audio.speech.create.side_effect = Exception(
            "Error code: 400 - {'error': {'message': 'The model playai-tts has been decommissioned', 'code': 'model_decommissioned'}}"
        )
        provider._client = mock_client

        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Test decommissioned")
        assert exc_info.value.status_code == 502
        assert "decommissioned" in str(exc_info.value)

    def test_zero_api_key_leakage_in_exception(self):
        secret_key = "gsk_super_secret_unshared_token_999"
        provider = GroqTextToSpeechProvider(api_key=secret_key)
        mock_client = MagicMock()
        mock_client.audio.speech.create.side_effect = Exception(f"Failed authenticating with {secret_key}")
        provider._client = mock_client

        with pytest.raises(TTSError) as exc_info:
            provider.synthesize("Testing secret masking")
        assert secret_key not in str(exc_info.value)
        assert "gsk_***" in str(exc_info.value)


# ===========================================================================
# 2. MockTextToSpeechProvider Offline Tests
# ===========================================================================

class TestMockTextToSpeechProvider:
    """Unit tests for MockTextToSpeechProvider generating synthetic WAV."""

    def test_mock_synthesis_valid_wav_header(self):
        provider = MockTextToSpeechProvider()
        result = provider.synthesize("Sample mock synthesis text")
        assert isinstance(result, TTSSynthesisResult)
        assert len(result.audio_bytes) > 44  # WAV header is 44 bytes
        assert result.audio_bytes[:4] == b"RIFF"
        assert result.audio_bytes[8:12] == b"WAVE"
        assert result.content_type == "audio/wav"
        assert result.format == "wav"
        assert result.provider == "mock"

    def test_mock_parameter_propagation(self):
        provider = MockTextToSpeechProvider(model="mock-tts", voice="diana")
        result = provider.synthesize(
            text="Testing custom parameters",
            voice="troy",
            model="custom-model",
            response_format="wav",
        )
        assert result.voice == "troy"
        assert result.model == "custom-model"
        assert result.format == "wav"

    def test_mock_empty_and_whitespace_rejection(self):
        provider = MockTextToSpeechProvider()
        with pytest.raises(TTSError) as exc_empty:
            provider.synthesize("")
        assert exc_empty.value.status_code == 400

        with pytest.raises(TTSError) as exc_ws:
            provider.synthesize("   \t\n  ")
        assert exc_ws.value.status_code == 400

    def test_mock_oversized_text_rejection(self):
        provider = MockTextToSpeechProvider(max_text_length=10)
        with pytest.raises(TTSError) as exc:
            provider.synthesize("More than ten characters")
        assert exc.value.status_code == 413

    def test_mock_failure_flag(self):
        provider = MockTextToSpeechProvider(should_fail=True)
        with pytest.raises(TTSError) as exc:
            provider.synthesize("Valid text")
        assert exc.value.status_code == 502
        assert "Mock TTS provider configured failure" in str(exc.value)


# ===========================================================================
# 3. Factory Resolution Tests
# ===========================================================================

class TestTTSFactory:
    """Tests for get_tts_provider factory."""

    def test_factory_resolves_mock(self):
        provider = get_tts_provider(provider_type="mock")
        assert isinstance(provider, MockTextToSpeechProvider)

    def test_factory_resolves_groq(self):
        provider = get_tts_provider(
            provider_type="groq",
            api_key="gsk_test1234567890abcdef",
            model="canopylabs/orpheus-v1-english",
        )
        assert isinstance(provider, GroqTextToSpeechProvider)
        assert provider.model_name == "canopylabs/orpheus-v1-english"

    def test_factory_fallback(self):
        provider = get_tts_provider(provider_type="unknown_vendor")
        assert isinstance(provider, GroqTextToSpeechProvider)


# ===========================================================================
# 4. TTSService Layer Tests
# ===========================================================================

class TestTTSService:
    """Unit tests for TTSService coordinating validation and provider execution."""

    def test_service_validation_and_delegation(self):
        mock_provider = MockTextToSpeechProvider()
        service = TTSService(provider=mock_provider)
        req = SynthesizeRequest(text="Hello workplace assistant", voice="autumn")
        result = service.synthesize_speech(req)
        assert result.audio_bytes[:4] == b"RIFF"
        assert result.voice == "autumn"

    def test_service_empty_text_rejection(self):
        service = TTSService(provider=MockTextToSpeechProvider())
        with pytest.raises(TTSError) as exc:
            service.synthesize_speech(SynthesizeRequest(text="    "))
        assert exc.value.status_code == 400

    def test_service_oversized_text_rejection(self):
        service = TTSService(provider=MockTextToSpeechProvider(), max_text_length=15)
        with pytest.raises(TTSError) as exc:
            service.synthesize_speech(SynthesizeRequest(text="This string is too long for the service limit."))
        assert exc.value.status_code == 413

    def test_service_unsupported_format_rejection(self):
        service = TTSService(provider=MockTextToSpeechProvider())
        with pytest.raises(TTSError) as exc:
            service.synthesize_speech(SynthesizeRequest(text="Valid text", format="unsupported_format"))
        assert exc.value.status_code == 422


# ===========================================================================
# 5. FastAPI POST /api/voice/synthesize Endpoint Tests
# ===========================================================================

class TestSynthesizeEndpoint:
    """API endpoint integration tests using TestClient."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_successful_synthesize_endpoint_returns_wav(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider())
        set_tts_service(mock_service)

        response = client.post(
            "/api/voice/synthesize",
            json={"text": "According to company policy, employees receive 18 days of annual leave."},
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert "attachment" in response.headers.get("content-disposition", "") or "inline" in response.headers.get("content-disposition", "")
        assert response.headers.get("x-tts-provider") == "mock"
        assert response.content[:4] == b"RIFF"
        assert response.content[8:12] == b"WAVE"

    def test_synthesize_endpoint_custom_voice_and_format(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider())
        set_tts_service(mock_service)

        response = client.post(
            "/api/voice/synthesize",
            json={
                "text": "Meeting scheduled for 3 PM.",
                "voice": "diana",
                "format": "wav",
                "speed": 1.25,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.headers.get("x-tts-voice") == "diana"
        assert len(response.content) > 0

    def test_synthesize_endpoint_empty_text_returns_400(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider())
        set_tts_service(mock_service)

        response = client.post("/api/voice/synthesize", json={"text": ""})
        assert response.status_code == 400
        assert "empty or whitespace" in response.json().get("detail", "").lower()

    def test_synthesize_endpoint_whitespace_text_returns_400(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider())
        set_tts_service(mock_service)

        response = client.post("/api/voice/synthesize", json={"text": "   \n\t  "})
        assert response.status_code == 400
        assert "empty or whitespace" in response.json().get("detail", "").lower()

    def test_synthesize_endpoint_missing_text_returns_422(self, client):
        response = client.post("/api/voice/synthesize", json={})
        assert response.status_code == 422

    def test_synthesize_endpoint_oversized_text_returns_413(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider(), max_text_length=20)
        set_tts_service(mock_service)

        response = client.post(
            "/api/voice/synthesize",
            json={"text": "This text is definitely longer than twenty characters."},
        )
        assert response.status_code == 413
        assert "exceeds maximum limit" in response.json().get("detail", "").lower()

    def test_synthesize_endpoint_unsupported_format_returns_422(self, client):
        mock_service = TTSService(provider=MockTextToSpeechProvider())
        set_tts_service(mock_service)

        response = client.post(
            "/api/voice/synthesize",
            json={"text": "Hello assistant", "format": "invalid_format_xyz"},
        )
        assert response.status_code == 422
        assert "unsupported audio format" in response.json().get("detail", "").lower()

    def test_synthesize_endpoint_provider_failure_returns_502(self, client):
        failing_provider = MockTextToSpeechProvider(should_fail=True)
        set_tts_service(TTSService(provider=failing_provider))

        response = client.post(
            "/api/voice/synthesize",
            json={"text": "This should fail at provider level."},
        )
        assert response.status_code == 502
        assert "configured failure" in response.json().get("detail", "").lower()

    def test_existing_transcribe_and_agent_endpoints_remain_functional(self, client):
        """Verify that adding Module 6.3 TTS did not affect Module 6.1 or 6.2 endpoints."""
        openapi_res = client.get("/openapi.json")
        assert openapi_res.status_code == 200
        paths = openapi_res.json()["paths"]
        assert "/api/voice/transcribe" in paths
        assert "/api/voice/agent" in paths
        assert "/api/voice/synthesize" in paths
        assert "/api/agent" in paths
        assert "/api/chat" in paths
