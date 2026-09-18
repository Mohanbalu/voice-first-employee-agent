"""Agent Orchestrator — Module 5.

LangGraph StateGraph that routes employee requests to:
  1. RAG (knowledge_query)    → KnowledgeAgentNode
  2. Tool intent stub          → ToolRouterNode
  3. Clarification             → ClarifyNode
  4. Safe decline              → DeclineNode

Architecture:
  START → classify_intent → [conditional router] → handler_node → END

The orchestrator is stateless and thread-safe. It accepts dependency injection
of all sub-components for deterministic testing without any API calls.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

try:
    from backend.app.agents.agent_state import (
        AgentState,
        IntentType,
        TOOL_INTENTS,
        make_initial_state,
    )
    from backend.app.agents.intent_classifier import IntentClassifier, IntentClassification
    from backend.app.agents.knowledge_agent import KnowledgeAgentNode
    from backend.app.agents.tool_router import ToolRouterNode
    from backend.app.agents.clarify_node import ClarifyNode
    from backend.app.agents.decline_node import DeclineNode
except ImportError:
    from app.agents.agent_state import (
        AgentState,
        IntentType,
        TOOL_INTENTS,
        make_initial_state,
    )
    from app.agents.intent_classifier import IntentClassifier, IntentClassification
    from app.agents.knowledge_agent import KnowledgeAgentNode
    from app.agents.tool_router import ToolRouterNode
    from app.agents.clarify_node import ClarifyNode
    from app.agents.decline_node import DeclineNode

from langgraph.graph import StateGraph, END, START

logger = logging.getLogger("agents.orchestrator")

# ── Routing constants ─────────────────────────────────────────────────────────
_NODE_CLASSIFY = "classify_intent"
_NODE_KNOWLEDGE = "knowledge_agent"
_NODE_TOOL = "tool_router"
_NODE_CLARIFY = "clarify"
_NODE_DECLINE = "decline"


def _route_after_classify(state: AgentState) -> str:
    """
    Conditional edge function — reads the classified intent from state
    and returns the name of the next node.

    This function is called by LangGraph after the classify_intent node runs.
    """
    intent_str = state.get("intent", IntentType.CLARIFY_NEEDED.value)
    error = state.get("error")

    # If classification itself failed, ask for clarification
    if error:
        return _NODE_CLARIFY

    try:
        intent = IntentType(intent_str)
    except ValueError:
        logger.warning("Unknown intent from classifier: %r — routing to clarify", intent_str)
        return _NODE_CLARIFY

    if intent == IntentType.KNOWLEDGE_QUERY:
        return _NODE_KNOWLEDGE
    if intent in TOOL_INTENTS or intent == IntentType.NAVIGATION:
        return _NODE_TOOL
    if intent == IntentType.CLARIFY_NEEDED:
        return _NODE_CLARIFY
    if intent == IntentType.OUT_OF_SCOPE:
        return _NODE_DECLINE

    # Fallback — never leave the graph stuck
    return _NODE_CLARIFY


class AgentOrchestrator:
    """
    Production-oriented LangGraph StateGraph agent orchestrator.

    All sub-components are injectable:
      - classifier: IntentClassifier (or mock)
      - knowledge_node: KnowledgeAgentNode (or mock)
      - tool_router: ToolRouterNode (or mock)
      - clarify_node: ClarifyNode (or mock)
      - decline_node: DeclineNode (or mock)

    This makes the orchestrator fully unit-testable without any API calls.
    """

    def __init__(
        self,
        classifier: Optional[IntentClassifier] = None,
        knowledge_node: Optional[KnowledgeAgentNode] = None,
        tool_router: Optional[ToolRouterNode] = None,
        clarify_node: Optional[ClarifyNode] = None,
        decline_node: Optional[DeclineNode] = None,
        db_session: Optional[Session] = None,
    ):
        self._classifier = classifier or IntentClassifier()
        self._db_session = db_session

        # Build nodes with injected session if provided
        if knowledge_node is not None:
            self._knowledge_node = knowledge_node
        else:
            self._knowledge_node = KnowledgeAgentNode(session=db_session)

        self._tool_router = tool_router or ToolRouterNode()
        self._clarify_node = clarify_node or ClarifyNode()
        self._decline_node = decline_node or DeclineNode()

        # Compile the LangGraph StateGraph once on init
        self._graph = self._build_graph()

    def _build_classify_node(self):
        """Returns the classify_intent node function (closure over self._classifier)."""
        classifier = self._classifier

        def classify_intent(state: AgentState) -> Dict[str, Any]:
            request = state.get("request", "")
            try:
                result: IntentClassification = classifier.classify(request)
                logger.info(
                    "Classified intent=%s confidence=%.2f",
                    result.intent.value,
                    result.confidence,
                )
                return {
                    "intent": result.intent.value,
                    "intent_confidence": result.confidence,
                    "intent_reasoning": result.reasoning,
                }
            except Exception as exc:
                logger.error("Intent classification failed: %s", exc)
                return {
                    "intent": IntentType.CLARIFY_NEEDED.value,
                    "intent_confidence": 0.0,
                    "intent_reasoning": "",
                    "error": f"Classification error: {exc}",
                }

        return classify_intent

    def _build_graph(self):
        """Builds and compiles the LangGraph StateGraph."""
        graph = StateGraph(AgentState)

        # ── Nodes ─────────────────────────────────────────────────────────────
        graph.add_node(_NODE_CLASSIFY, self._build_classify_node())
        graph.add_node(_NODE_KNOWLEDGE, self._knowledge_node)
        graph.add_node(_NODE_TOOL, self._tool_router)
        graph.add_node(_NODE_CLARIFY, self._clarify_node)
        graph.add_node(_NODE_DECLINE, self._decline_node)

        # ── Edges ─────────────────────────────────────────────────────────────
        graph.add_edge(START, _NODE_CLASSIFY)

        graph.add_conditional_edges(
            _NODE_CLASSIFY,
            _route_after_classify,
            {
                _NODE_KNOWLEDGE: _NODE_KNOWLEDGE,
                _NODE_TOOL: _NODE_TOOL,
                _NODE_CLARIFY: _NODE_CLARIFY,
                _NODE_DECLINE: _NODE_DECLINE,
            },
        )

        graph.add_edge(_NODE_KNOWLEDGE, END)
        graph.add_edge(_NODE_TOOL, END)
        graph.add_edge(_NODE_CLARIFY, END)
        graph.add_edge(_NODE_DECLINE, END)

        return graph.compile()

    def run(
        self,
        request: str,
        tenant_id: str,
        conversation_id: Optional[str] = None,
        current_user: Optional[Any] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> AgentState:
        """
        Executes the full agent graph for a single user request.

        Args:
            request:         The employee's message.
            tenant_id:       UUID string — mandatory for tenant isolation.
            conversation_id: Optional session ID for future multi-turn support.
            current_user:    Optional authenticated User object.
            latitude:        Optional device GPS latitude.
            longitude:       Optional device GPS longitude.
            accuracy:        Optional GPS horizontal accuracy in meters.

        Returns:
            The final AgentState after all nodes have run.
        """
        if not request or not request.strip():
            # Short-circuit for empty input — don't enter the graph
            state = make_initial_state(
                request=request,
                tenant_id=tenant_id,
                conversation_id=conversation_id or str(uuid.uuid4()),
                latitude=latitude,
                longitude=longitude,
                accuracy=accuracy,
            )
            state["intent"] = IntentType.CLARIFY_NEEDED.value
            state["agent_mode"] = "clarify"
            state["requires_clarification"] = True
            state["clarification_question"] = "Please enter your question or request."
            state["final_response"] = "Please enter your question or request."
            return state

        initial = make_initial_state(
            request=request.strip(),
            tenant_id=tenant_id,
            conversation_id=conversation_id or str(uuid.uuid4()),
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
        )
        if self._db_session is not None:
            initial["db_session"] = self._db_session
        if current_user is not None:
            initial["current_user"] = current_user

        start = time.monotonic()
        final_state: AgentState = self._graph.invoke(initial)
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "Agent run complete. mode=%s intent=%s latency=%.1fms",
            final_state.get("agent_mode"),
            final_state.get("intent"),
            elapsed_ms,
        )

        return final_state
