"""Voice Agent Service — Module 6.2.

Connects the Module 6.1 Speech-to-Text subsystem to the Module 5 LangGraph
Agent Orchestrator:
  Audio Upload → STT (Whisper) → Transcript → AgentOrchestrator → LangGraph → Response

Responsibilities:
- Transcribe uploaded audio using the configured STT provider.
- Validate transcript text (short-circuit on empty/blank audio).
- Forward transcript to AgentOrchestrator with explicit tenant scoping.
- Extract safe, sanitized response metadata without exposing secrets.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

try:
    from backend.app.agents.agent_state import AgentState
    from backend.app.agents.orchestrator import AgentOrchestrator
    from backend.app.config import config
    from backend.app.schemas.voice import VoiceAgentResponse
    from backend.app.voice.speech_to_text import (
        SpeechToTextProvider,
        STTError,
        STTTranscriptionResult,
        get_stt_provider,
    )
except ImportError:
    from app.agents.agent_state import AgentState
    from app.agents.orchestrator import AgentOrchestrator
    from app.config import config
    from app.schemas.voice import VoiceAgentResponse
    from app.voice.speech_to_text import (
        SpeechToTextProvider,
        STTError,
        STTTranscriptionResult,
        get_stt_provider,
    )

logger = logging.getLogger("voice.voice_agent_service")


class VoiceAgentService:
    """Service orchestrating Speech-to-Text and LangGraph Agent execution."""

    def __init__(
        self,
        stt_provider: Optional[SpeechToTextProvider] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
    ):
        self._stt_provider = stt_provider or get_stt_provider()
        self._orchestrator = orchestrator

    def _get_orchestrator(self) -> AgentOrchestrator:
        if self._orchestrator is not None:
            return self._orchestrator
        # Lazily create default orchestrator
        self._orchestrator = AgentOrchestrator()
        return self._orchestrator

    def process_voice_request(
        self,
        audio_file: Union[str, Path, BinaryIO],
        filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> VoiceAgentResponse:
        """Processes an audio recording through STT and LangGraph agent orchestration.

        1. Transcribes audio via STT provider.
        2. Validates transcript (safely stops on empty transcript without LLM calls).
        3. Forwards transcript to LangGraph AgentOrchestrator with tenant isolation.
        4. Returns sanitized VoiceAgentResponse.
        """
        # Resolve and validate tenant ID format
        resolved_tenant_id = tenant_id or config.tenant.default_id
        try:
            uuid.UUID(resolved_tenant_id)
        except (ValueError, AttributeError) as exc:
            logger.warning("Invalid tenant_id format: %r", resolved_tenant_id)
            raise ValueError(f"Invalid tenant_id format: '{resolved_tenant_id}'. Must be a valid UUID.") from exc

        conv_id = conversation_id or str(uuid.uuid4())

        logger.info(
            "Starting VoiceAgent pipeline. tenant=%s conversation=%s filename=%s",
            resolved_tenant_id,
            conv_id,
            filename or "audio",
        )

        # 1. Speech-to-Text Transcription
        try:
            stt_result: STTTranscriptionResult = self._stt_provider.transcribe(
                audio_file=audio_file,
                filename=filename,
                language=language,
                prompt=prompt,
            )
        except STTError:
            raise
        except Exception as exc:
            logger.error("STT provider execution failed: %s", exc)
            raise STTError(f"STT transcription failed: {exc}", status_code=502) from exc

        raw_transcript = (stt_result.text or "").strip()

        # 2. Empty Transcript Safety Check
        if not raw_transcript:
            logger.info("STT returned empty transcript. Short-circuiting pipeline.")
            return VoiceAgentResponse(
                success=False,
                transcript="",
                response=None,
                intent=None,
                status="empty_transcript",
                duration=stt_result.duration,
                error="No speech detected in the audio file.",
            )

        logger.info(
            "STT succeeded. transcript_len=%d, dispatching to LangGraph AgentOrchestrator",
            len(raw_transcript),
        )

        # 3. LangGraph Agent Execution
        orchestrator = self._get_orchestrator()
        try:
            final_state: AgentState = orchestrator.run(
                request=raw_transcript,
                tenant_id=resolved_tenant_id,
                conversation_id=conv_id,
            )
        except Exception as exc:
            logger.error("Agent orchestrator execution failed: %s", exc)
            raise RuntimeError(f"Agent orchestration failed: {exc}") from exc

        # 4. Extract safe metadata
        rag_sources: List[Dict[str, Any]] = []
        rag_resp = final_state.get("rag_response")
        if rag_resp and isinstance(rag_resp, dict):
            rag_sources = rag_resp.get("sources", [])

        tool_intents: List[Dict[str, Any]] = final_state.get("tool_intents") or []

        resp_text = final_state.get("final_response") or ""
        agent_mode = final_state.get("agent_mode")
        requires_clarification = bool(final_state.get("requires_clarification") or False)
        suggest_ticket = False
        if agent_mode in ("clarify", "decline") or requires_clarification:
            suggest_ticket = True
        else:
            resp_lower = resp_text.lower()
            fallback_phrases = [
                "could not find",
                "unable to find",
                "cannot find",
                "not find relevant information",
                "no relevant information",
                "not available in the company knowledge",
                "not mentioned in the available",
                "do not have information",
                "don't have information",
                "not found in the records",
                "not specified in current company records",
                "not specified in the provided records",
                "please raise a ticket",
                "contact it support",
                "contact hr",
                "outside the scope of what i can help with",
            ]
            if any(phrase in resp_lower for phrase in fallback_phrases):
                suggest_ticket = True

        return VoiceAgentResponse(
            success=True,
            transcript=raw_transcript,
            response=resp_text,
            intent=final_state.get("intent"),
            status="completed",
            agent_mode=agent_mode,
            intent_confidence=float(final_state.get("intent_confidence") or 0.0),
            requires_clarification=requires_clarification,
            clarification_question=final_state.get("clarification_question"),
            rag_sources=rag_sources,
            tool_intents=tool_intents,
            duration=stt_result.duration,
            error=final_state.get("error"),
            suggest_ticket=suggest_ticket,
        )
