"""Agent State — Module 5.

Defines the LangGraph AgentState TypedDict and IntentType enum shared across
all nodes in the orchestrator graph.
"""

from __future__ import annotations

from enum import Enum
import uuid
from typing import Any, Dict, List, Optional

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class IntentType(str, Enum):
    """All recognised employee request intents.

    The classifier assigns exactly one intent to every incoming message.
    Intents drive conditional routing in the LangGraph StateGraph.
    """

    # → Answered by RAG (Module 4)
    KNOWLEDGE_QUERY = "knowledge_query"

    # → Safe tool stubs (real execution in Module 6+)
    LEAVE_REQUEST = "leave_request"
    IT_SUPPORT = "it_support"
    CALENDAR_QUERY = "calendar_query"
    TASK_MANAGEMENT = "task_management"
    TIMESHEET = "timesheet"
    NAVIGATION = "navigation"

    # → Special routing
    CLARIFY_NEEDED = "clarify_needed"   # intent too ambiguous → ask follow-up
    OUT_OF_SCOPE = "out_of_scope"       # outside assistant capability → safe decline


# All tool-backed intents (not RAG, not special routing)
TOOL_INTENTS = {
    IntentType.LEAVE_REQUEST,
    IntentType.IT_SUPPORT,
    IntentType.CALENDAR_QUERY,
    IntentType.TASK_MANAGEMENT,
    IntentType.TIMESHEET,
    IntentType.NAVIGATION,
}


class AgentState(TypedDict, total=False):
    """
    LangGraph graph state — passed between every node.

    All fields are Optional (total=False) except the three required inputs
    that must be present at graph entry: request, tenant_id, conversation_id.

    Fields are immutable once set by a node — nodes return partial dicts
    with only the keys they write.
    """

    # ── Inputs (required at graph entry) ─────────────────────────────────────
    request: str                        # original user message (never modified)
    tenant_id: str                      # UUID string — tenant isolation key
    conversation_id: str                # for future multi-turn session tracking

    # ── Classification layer ──────────────────────────────────────────────────
    intent: Optional[str]               # IntentType value string
    intent_confidence: float            # classifier confidence [0.0–1.0]
    intent_reasoning: Optional[str]     # short explanation from classifier

    # ── Clarification layer ───────────────────────────────────────────────────
    requires_clarification: bool
    clarification_question: Optional[str]

    # ── RAG layer (knowledge_query) ───────────────────────────────────────────
    rag_response: Optional[Dict[str, Any]]   # full RAGResponse.to_dict()

    # ── Tool intent layer (non-RAG intents) ──────────────────────────────────
    tool_intents: List[Dict[str, Any]]  # list of ToolIntentResponse dicts

    # ── Output layer ─────────────────────────────────────────────────────────
    final_response: Optional[str]       # text sent back to the employee
    agent_mode: str                     # "rag" | "tool_intent" | "clarify" | "decline"
    error: Optional[str]                # pipeline error, if any


def make_initial_state(
    request: str,
    tenant_id: str,
    conversation_id: Optional[str] = None,
) -> AgentState:
    """Returns a fully initialised AgentState for graph entry."""
    return AgentState(
        request=request,
        tenant_id=tenant_id,
        conversation_id=conversation_id or str(uuid.uuid4()),
        intent=None,
        intent_confidence=0.0,
        intent_reasoning=None,
        requires_clarification=False,
        clarification_question=None,
        rag_response=None,
        tool_intents=[],
        final_response=None,
        agent_mode="",
        error=None,
    )
