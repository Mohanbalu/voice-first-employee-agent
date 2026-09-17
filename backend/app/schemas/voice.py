"""Voice API Schemas — Module 6.1.

Pydantic models for speech-to-text transcription endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    """Successful transcription response schema."""

    success: bool = Field(default=True, description="Whether the transcription succeeded.")
    text: str = Field(..., description="The full transcribed text.")
    language: Optional[str] = Field(default=None, description="Detected or specified ISO-639-1 language code.")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds if provided by provider.")
    segments: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional timestamped segments.")


class TranscriptionErrorResponse(BaseModel):
    """Error response schema for transcription endpoints."""

    success: bool = Field(default=False, description="Whether the operation succeeded (false).")
    error: str = Field(..., description="Error category or summary.")
    detail: Optional[str] = Field(default=None, description="Detailed explanation.")


class VoiceAgentResponse(BaseModel):
    """Unified response schema for POST /api/voice/agent (Module 6.2)."""

    success: bool = Field(default=True, description="Whether the voice agent pipeline completed successfully.")
    transcript: str = Field(..., description="The recognized speech transcript from STT.")
    response: Optional[str] = Field(default=None, description="The final text response from the Agent Orchestrator.")
    intent: Optional[str] = Field(default=None, description="Classified intent (e.g. KNOWLEDGE_QUERY, GREETING, etc.).")
    status: str = Field(default="completed", description="Status code: completed | empty_transcript | failed.")
    agent_mode: Optional[str] = Field(default=None, description="Which path was taken: rag | tool_intent | clarify | decline.")
    intent_confidence: float = Field(default=0.0, description="Classifier confidence score [0.0–1.0].")
    requires_clarification: bool = Field(default=False, description="Whether follow-up clarification was requested.")
    clarification_question: Optional[str] = Field(default=None, description="Clarification question if asked.")
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Policy sources cited if RAG mode was used.")
    tool_intents: List[Dict[str, Any]] = Field(default_factory=list, description="Tool intent stubs if tool path was taken.")
    duration: Optional[float] = Field(default=None, description="Audio duration in seconds if provided by STT provider.")
    suggest_ticket: bool = Field(
        default=False,
        description="Whether raising a support ticket is suggested when the assistant is unable to answer.",
    )
    error: Optional[str] = Field(default=None, description="Optional error description if any step failed.")


class SynthesizeRequest(BaseModel):
    """Request payload for text-to-speech synthesis (Module 6.3)."""

    text: str = Field(..., description="Text content to synthesize into speech.")
    voice: Optional[str] = Field(default=None, description="Optional voice persona name (e.g. 'autumn', 'diana', 'hannah', 'austin', 'troy').")
    model: Optional[str] = Field(default=None, description="Optional TTS model name (e.g. 'canopylabs/orpheus-v1-english').")
    format: Optional[str] = Field(default=None, description="Optional audio output format (e.g. 'wav').")
    speed: Optional[float] = Field(default=None, description="Optional speech rate multiplier (e.g. 1.0).")


class SynthesizeErrorResponse(BaseModel):
    """Error response schema for speech synthesis endpoints."""

    success: bool = Field(default=False, description="Whether the operation succeeded (false).")
    error: str = Field(..., description="Error category or summary.")
    detail: Optional[str] = Field(default=None, description="Detailed explanation.")

