"""Agent API Schemas — Module 5.

Pydantic v2 request/response models for POST /api/agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ToolIntentRef(BaseModel):
    """A single tool intent stub result."""

    tool: str
    intent: str
    status: str           # "stub_pending" | future: "success" | "error"
    message: str
    requires_action: bool = False
    action_description: Optional[str] = None
    module_planned: Optional[str] = None


class AgentRequest(BaseModel):
    """
    Request body for POST /api/agent.

    NOTE (Development-only):
    `tenant_id` is accepted directly in the request body for development and testing.
    Module 8 (Authentication) will derive this from the authenticated JWT token.
    """

    request: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The employee's request or question.",
        examples=["How many days of annual leave can I take?"],
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description=(
            "[DEVELOPMENT ONLY] Explicit tenant UUID. "
            "Will be replaced by authenticated identity in Module 8."
        ),
        examples=["00000000-0000-0000-0000-000000000001"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional session/conversation ID for future multi-turn support.",
    )

    @field_validator("request")
    @classmethod
    def request_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Request must not be blank.")
        return v.strip()


class AgentResponse(BaseModel):
    """
    Structured response from the Agent Orchestrator.

    Fields:
    - response:          The final text response to the employee.
    - agent_mode:        Which path was taken: rag | tool_intent | clarify | decline
    - intent:            Classified intent name.
    - intent_confidence: Classifier confidence score [0.0–1.0].
    - requires_clarification: True if a follow-up question was asked.
    - clarification_question: The follow-up question (if any).
    - rag_sources:       Sources used (if agent_mode == "rag").
    - tool_intents:      Tool stub results (if agent_mode == "tool_intent").
    - error:             Optional error description.
    """

    response: str
    agent_mode: str = Field(description="rag | tool_intent | clarify | decline")
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    requires_clarification: bool = False
    clarification_question: Optional[str] = None
    rag_sources: List[Dict[str, Any]] = Field(default_factory=list)
    tool_intents: List[ToolIntentRef] = Field(default_factory=list)
    error: Optional[str] = None
    suggest_ticket: bool = Field(
        default=False,
        description="Whether raising a support ticket is suggested when the assistant is unable to answer.",
    )


class AgentErrorResponse(BaseModel):
    """Error response for non-2xx agent responses."""

    detail: str
    code: Optional[str] = None
