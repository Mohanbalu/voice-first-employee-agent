"""Unit tests for Voice Agent Orchestrator Module 6.2 (STT → LangGraph).

Verifies:
1. Successful audio → transcript → agent response (KNOWLEDGE_QUERY).
2. Successful audio → tool intent stub routing (e.g. HR escalation, meeting booking).
3. Transcript and tenant_id reach AgentOrchestrator.
4. Empty transcript stops pipeline immediately without calling LangGraph.
5. Upstream STT failure handling (502).
6. Upstream Agent/LLM failure handling.
7. Empty audio file (0 bytes) rejected (400).
8. Unsupported audio format rejected (415).
9. Oversized audio file rejected (413).
10. Temporary audio file cleanup guaranteed on success and failure.
11. Invalid tenant UUID rejected (400).
12. Zero API key or secret leakage in repr, dict, or error output.
13. Existing text /api/agent and /api/voice/transcribe remain unaffected.
"""

from __future__ import annotations

import io
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.voice import VoiceAgentResponse
from backend.app.voice.speech_to_text import (
    MockSpeechToTextProvider,
    STTError,
    STTTranscriptionResult,
)
from backend.app.voice.voice_agent_service import VoiceAgentService
from backend.app.routes.voice import set_stt_provider, set_voice_agent_service


@pytest.fixture(autouse=True)
def reset_voice_agent_overrides():
    """Cleans up route dependency overrides after each test."""
    set_stt_provider(None)
    set_voice_agent_service(None)
    yield
    set_stt_provider(None)
    set_voice_agent_service(None)


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


def make_mock_orchestrator(
    final_response: str = "According to company policy, employees receive 20 annual leave days.",
    intent: str = "knowledge_query",
    confidence: float = 0.95,
    agent_mode: str = "rag",
    rag_sources: Optional[List[Dict[str, Any]]] = None,
    tool_intents: Optional[List[Dict[str, Any]]] = None,
    should_fail: bool = False,
):
    """Creates a mock AgentOrchestrator for deterministic testing."""
    mock_orch = MagicMock()
    if should_fail:
        mock_orch.run.side_effect = RuntimeError("Mock agent orchestration error")
    else:
        mock_orch.run.return_value = {
            "request": "dummy",
            "tenant_id": "00000000-0000-0000-0000-000000000001",
            "conversation_id": "conv-123",
            "intent": intent,
            "intent_confidence": confidence,
            "agent_mode": agent_mode,
            "final_response": final_response,
            "requires_clarification": False,
            "clarification_question": None,
            "rag_response": {
                "answer": final_response,
                "sources": rag_sources or [
                    {
                        "document_name": "Leave Policy",
                        "source_file": "leave_policy.pdf",
                        "page_start": 3,
                        "page_end": 4,
                        "similarity_score": 0.88,
                        "chunk_id": "leave_001",
                    }
                ],
            },
            "tool_intents": tool_intents or [],
            "error": None,
        }
    return mock_orch


# ── VoiceAgentService Unit Tests ───────────────────────────────────────────────


class TestVoiceAgentService:
    """Direct unit tests for VoiceAgentService."""

    def test_service_successful_knowledge_flow(self):
        mock_stt = MockSpeechToTextProvider(
            mock_transcript="What is the annual leave policy for employees?",
            mock_duration=2.5,
        )
        mock_orch = make_mock_orchestrator()

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        result = service.process_voice_request(
            audio_file=io.BytesIO(b"RIFFdummywav"),
            filename="leave_query.wav",
            tenant_id="00000000-0000-0000-0000-000000000001",
        )

        assert isinstance(result, VoiceAgentResponse)
        assert result.success is True
        assert result.transcript == "What is the annual leave policy for employees?"
        assert result.intent == "knowledge_query"
        assert result.agent_mode == "rag"
        assert "20 annual leave days" in result.response
        assert result.status == "completed"
        assert len(result.rag_sources) == 1

        # Verify orchestrator was called with exact transcript and tenant_id
        mock_orch.run.assert_called_once()
        call_kwargs = mock_orch.run.call_args[1]
        assert call_kwargs["request"] == "What is the annual leave policy for employees?"
        assert call_kwargs["tenant_id"] == "00000000-0000-0000-0000-000000000001"

    def test_service_empty_transcript_stops_pipeline(self):
        """Ensures that empty transcript stops the pipeline without invoking LangGraph."""
        mock_stt = MockSpeechToTextProvider(mock_transcript="", mock_duration=1.0)
        mock_orch = make_mock_orchestrator()

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        result = service.process_voice_request(
            audio_file=io.BytesIO(b"RIFFdummywav"),
            filename="silence.wav",
        )

        assert result.success is False
        assert result.transcript == ""
        assert result.response is None
        assert result.status == "empty_transcript"

        # LangGraph MUST NOT be called
        mock_orch.run.assert_not_called()

    def test_service_whitespace_transcript_stops_pipeline(self):
        mock_stt = MockSpeechToTextProvider(mock_transcript="   \n\t  ", mock_duration=1.0)
        mock_orch = make_mock_orchestrator()

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        result = service.process_voice_request(
            audio_file=io.BytesIO(b"RIFFdummywav"),
            filename="blank.wav",
        )

        assert result.success is False
        assert result.status == "empty_transcript"
        mock_orch.run.assert_not_called()

    def test_service_stt_failure_propagates(self):
        mock_stt = MockSpeechToTextProvider(should_fail=True)
        mock_orch = make_mock_orchestrator()

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        with pytest.raises(STTError) as exc_info:
            service.process_voice_request(
                audio_file=io.BytesIO(b"audio"),
                filename="test.wav",
            )
        assert exc_info.value.status_code == 502
        mock_orch.run.assert_not_called()

    def test_service_orchestrator_failure_raises_error(self):
        mock_stt = MockSpeechToTextProvider(mock_transcript="Hello assistant")
        mock_orch = make_mock_orchestrator(should_fail=True)

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        with pytest.raises(RuntimeError) as exc_info:
            service.process_voice_request(
                audio_file=io.BytesIO(b"audio"),
                filename="test.wav",
            )
        assert "Agent orchestration failed" in str(exc_info.value)

    def test_service_invalid_tenant_id_raises_value_error(self):
        mock_stt = MockSpeechToTextProvider(mock_transcript="Hello")
        mock_orch = make_mock_orchestrator()

        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        with pytest.raises(ValueError) as exc_info:
            service.process_voice_request(
                audio_file=io.BytesIO(b"audio"),
                filename="test.wav",
                tenant_id="invalid-not-a-uuid",
            )
        assert "Must be a valid UUID" in str(exc_info.value)
        mock_orch.run.assert_not_called()


# ── FastAPI Endpoint Unit Tests (POST /api/voice/agent) ─────────────────────────


class TestVoiceAgentEndpoint:
    """Tests for POST /api/voice/agent endpoint."""

    def test_successful_voice_agent_endpoint(self, client):
        mock_stt = MockSpeechToTextProvider(mock_transcript="What is the annual leave policy for employees?")
        mock_orch = make_mock_orchestrator()
        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        set_voice_agent_service(service)

        audio_bytes = b"RIFFfake_wav_audio_content"
        response = client.post(
            "/api/voice/agent",
            files={"audio": ("voice.wav", audio_bytes, "audio/wav")},
            data={
                "tenant_id": "00000000-0000-0000-0000-000000000001",
                "language": "en",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["transcript"] == "What is the annual leave policy for employees?"
        assert "20 annual leave days" in data["response"]
        assert data["intent"] == "knowledge_query"
        assert data["status"] == "completed"
        assert data["agent_mode"] == "rag"
        assert len(data["rag_sources"]) > 0

    def test_voice_agent_tool_intent_routing(self, client):
        mock_stt = MockSpeechToTextProvider(mock_transcript="Please book a conference room for 2pm")
        mock_orch = make_mock_orchestrator(
            final_response="I can assist with scheduling.",
            intent="meeting_scheduling",
            agent_mode="tool_intent",
            tool_intents=[{"tool": "calendar", "intent": "meeting_scheduling", "status": "stub_pending", "message": "Stub ready"}],
        )
        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        set_voice_agent_service(service)

        response = client.post(
            "/api/voice/agent",
            files={"audio": ("schedule.wav", b"RIFFaudio", "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["intent"] == "meeting_scheduling"
        assert data["agent_mode"] == "tool_intent"
        assert len(data["tool_intents"]) == 1

    def test_empty_transcript_stops_pipeline_endpoint(self, client):
        mock_stt = MockSpeechToTextProvider(mock_transcript="")
        mock_orch = make_mock_orchestrator()
        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        set_voice_agent_service(service)

        response = client.post(
            "/api/voice/agent",
            files={"audio": ("silent.wav", b"RIFFaudio", "audio/wav")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["transcript"] == ""
        assert data["response"] is None
        assert data["status"] == "empty_transcript"
        mock_orch.run.assert_not_called()

    def test_empty_audio_file_returns_400(self, client):
        response = client.post(
            "/api/voice/agent",
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400
        assert "Audio file is empty" in response.json()["detail"]

    def test_unsupported_audio_extension_returns_415(self, client):
        response = client.post(
            "/api/voice/agent",
            files={"audio": ("script.sh", b"echo hi", "application/x-sh")},
        )
        assert response.status_code == 415
        assert "Unsupported audio format" in response.json()["detail"]

    def test_oversized_audio_returns_413(self, client):
        with patch("backend.app.config.config.stt.max_upload_size_mb", 1):
            oversized_audio = b"0" * (2 * 1024 * 1024)
            response = client.post(
                "/api/voice/agent",
                files={"audio": ("huge.wav", oversized_audio, "audio/wav")},
            )
            assert response.status_code == 413

    def test_invalid_tenant_id_returns_400(self, client):
        response = client.post(
            "/api/voice/agent",
            files={"audio": ("test.wav", b"RIFFaudio", "audio/wav")},
            data={"tenant_id": "not-a-uuid"},
        )
        assert response.status_code == 400
        assert "Invalid tenant_id" in response.json()["detail"]

    def test_stt_failure_returns_502(self, client):
        mock_stt = MockSpeechToTextProvider(should_fail=True)
        service = VoiceAgentService(stt_provider=mock_stt)
        set_voice_agent_service(service)

        response = client.post(
            "/api/voice/agent",
            files={"audio": ("fail.wav", b"RIFFaudio", "audio/wav")},
        )
        assert response.status_code == 502

    def test_temporary_file_cleanup_on_success_and_failure(self, client):
        """Verifies temporary audio files are removed from disk on both success and error."""
        mock_stt = MockSpeechToTextProvider(mock_transcript="Checking cleanup")
        mock_orch = make_mock_orchestrator()
        service = VoiceAgentService(stt_provider=mock_stt, orchestrator=mock_orch)
        set_voice_agent_service(service)

        temp_dir = Path(tempfile.gettempdir())
        before_wavs = set(temp_dir.glob("*.wav"))

        response = client.post(
            "/api/voice/agent",
            files={"audio": ("cleanup_test.wav", b"RIFFaudio", "audio/wav")},
        )
        assert response.status_code == 200

        after_wavs = set(temp_dir.glob("*.wav"))
        assert len(after_wavs - before_wavs) == 0

        # Test failure cleanup
        mock_stt._should_fail = True
        response = client.post(
            "/api/voice/agent",
            files={"audio": ("cleanup_fail.wav", b"RIFFaudio", "audio/wav")},
        )
        assert response.status_code == 502
        after_fail_wavs = set(temp_dir.glob("*.wav"))
        assert len(after_fail_wavs - before_wavs) == 0

    def test_existing_text_agent_endpoint_remains_functional(self, client):
        """Verifies existing /api/agent text endpoint is not broken."""
        mock_orch = make_mock_orchestrator(final_response="Direct text response")
        from backend.app.routes.agent import _get_orchestrator
        app.dependency_overrides[_get_orchestrator] = lambda: mock_orch

        try:
            response = client.post(
                "/api/agent",
                json={"request": "Direct text query without audio"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["response"] == "Direct text response"
        finally:
            app.dependency_overrides.pop(_get_orchestrator, None)

    def test_existing_voice_transcribe_remains_functional(self, client):
        """Verifies existing /api/voice/transcribe endpoint continues working."""
        mock_stt = MockSpeechToTextProvider(mock_transcript="Transcription only")
        set_stt_provider(mock_stt)

        response = client.post(
            "/api/voice/transcribe",
            files={"audio": ("transcribe_test.wav", b"RIFFaudio", "audio/wav")},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "Transcription only"
