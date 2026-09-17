"""Unit and Integration Tests for Module 5 — Agent Orchestrator.

Covers:
1. AgentState data structures and initialization
2. IntentClassifier (heuristics, edge cases, provider injection)
3. Individual graph nodes (Knowledge, Clarify, Decline, HR stub, Tool router)
4. AgentOrchestrator LangGraph execution (all routing paths, edge cases, isolation)
5. FastAPI /api/agent endpoint (validation, error handling, responses)

Zero OpenAI API calls or expenditures during any test.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.agents.agent_state import (
    AgentState,
    IntentType,
    TOOL_INTENTS,
    make_initial_state,
)
from backend.app.agents.intent_classifier import (
    IntentClassifier,
    IntentClassification,
    _heuristic_classify,
)
from backend.app.agents.clarify_node import ClarifyNode
from backend.app.agents.decline_node import DeclineNode
from backend.app.agents.hr_agent import HRAgentNode
from backend.app.agents.tool_router import ToolRouterNode
from backend.app.agents.knowledge_agent import KnowledgeAgentNode
from backend.app.agents.orchestrator import AgentOrchestrator, _route_after_classify
from backend.app.routes.agent import _get_orchestrator
from backend.app.schemas.agent import AgentRequest, AgentResponse


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_tenant_id() -> str:
    return "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def mock_classifier() -> IntentClassifier:
    """Classifier with a mock provider for predictable testing."""
    def stub_provider(text: str) -> IntentClassification:
        lowered = text.lower()
        if "leave" in lowered:
            return IntentClassification(IntentType.LEAVE_REQUEST, 0.95, "Leave keyword match")
        if "policy" in lowered:
            return IntentClassification(IntentType.KNOWLEDGE_QUERY, 0.90, "Policy keyword match")
        if "weather" in lowered:
            return IntentClassification(IntentType.OUT_OF_SCOPE, 0.85, "Out of scope keyword match")
        if "unclear" in lowered:
            return IntentClassification(IntentType.CLARIFY_NEEDED, 0.50, "Clarification match")
        return IntentClassification(IntentType.KNOWLEDGE_QUERY, 0.80, "Default test match")

    return IntentClassifier(provider=stub_provider)


@pytest.fixture
def mock_rag_service():
    """Mock RAGService returning predictable answer."""
    service = MagicMock()
    mock_response = MagicMock()
    mock_response.answer = "Company policy allows 25 days of annual leave."
    mock_response.confidence = 0.92
    mock_response.sources = [
        {"document_name": "Leave and Holiday Policy.pdf", "chunk_id": "chunk-123"}
    ]
    mock_response.has_sufficient_context = True
    mock_response.error = None
    mock_response.used_context_count = 1
    mock_response.to_dict.return_value = {
        "answer": mock_response.answer,
        "confidence": mock_response.confidence,
        "sources": mock_response.sources,
        "has_sufficient_context": True,
        "error": None,
    }
    service.answer_question.return_value = mock_response
    return service


@pytest.fixture
def test_orchestrator(sample_tenant_id, mock_rag_service) -> AgentOrchestrator:
    """Configured orchestrator with mock classifier and mock RAG service."""
    classifier = IntentClassifier(
        provider=lambda text: IntentClassification(
            IntentType.KNOWLEDGE_QUERY if "policy" in text.lower()
            else IntentType.LEAVE_REQUEST if "leave" in text.lower()
            else IntentType.OUT_OF_SCOPE if "joke" in text.lower()
            else IntentType.CLARIFY_NEEDED,
            0.90,
            "Rule-based test classification",
        )
    )
    knowledge_node = KnowledgeAgentNode(rag_service=mock_rag_service)
    return AgentOrchestrator(
        classifier=classifier,
        knowledge_node=knowledge_node,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT STATE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_make_initial_state_defaults(sample_tenant_id):
    """Initial state creates expected default structure."""
    state = make_initial_state(
        request="What is the leave policy?",
        tenant_id=sample_tenant_id,
    )
    assert state["request"] == "What is the leave policy?"
    assert state["tenant_id"] == sample_tenant_id
    assert isinstance(state["conversation_id"], str)
    assert state["intent"] is None
    assert state["intent_confidence"] == 0.0
    assert state["requires_clarification"] is False
    assert state["clarification_question"] is None
    assert state["rag_response"] is None
    assert state["tool_intents"] == []
    assert state["final_response"] is None
    assert state["agent_mode"] == ""
    assert state["error"] is None


def test_tool_intents_constant():
    """Verify tool intents set contains expected action types."""
    assert IntentType.LEAVE_REQUEST in TOOL_INTENTS
    assert IntentType.IT_SUPPORT in TOOL_INTENTS
    assert IntentType.CALENDAR_QUERY in TOOL_INTENTS
    assert IntentType.TASK_MANAGEMENT in TOOL_INTENTS
    assert IntentType.TIMESHEET in TOOL_INTENTS
    assert IntentType.NAVIGATION in TOOL_INTENTS
    assert IntentType.KNOWLEDGE_QUERY not in TOOL_INTENTS
    assert IntentType.OUT_OF_SCOPE not in TOOL_INTENTS


# ─────────────────────────────────────────────────────────────────────────────
# 2. INTENT CLASSIFIER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_classify_knowledge_query():
    """Heuristic recognizes company policy & knowledge questions."""
    queries = [
        "What is the company policy on remote work?",
        "Tell me about health insurance benefits",
        "What are the disciplinary guidelines?",
        "Can I get travel reimbursement under the expense policy?",
    ]
    for q in queries:
        result = _heuristic_classify(q)
        assert result.intent == IntentType.KNOWLEDGE_QUERY, f"Failed for: {q}"
        assert result.confidence >= 0.70


def test_heuristic_classify_leave_request():
    """Heuristic recognizes leave and PTO requests."""
    queries = [
        "I want to apply for annual leave next week",
        "How do I submit sick leave?",
        "Request time off for vacation",
        "Book 3 days of pto",
    ]
    for q in queries:
        result = _heuristic_classify(q)
        assert result.intent == IntentType.LEAVE_REQUEST, f"Failed for: {q}"


def test_heuristic_classify_it_support():
    """Heuristic recognizes IT support requests."""
    queries = [
        "My laptop screen is broken, please raise an IT ticket",
        "How do I connect to the office VPN?",
        "I need a password reset for my account",
    ]
    for q in queries:
        result = _heuristic_classify(q)
        assert result.intent == IntentType.IT_SUPPORT, f"Failed for: {q}"


def test_heuristic_classify_out_of_scope():
    """Heuristic recognizes non-work out of scope queries."""
    queries = [
        "What's the weather in Tokyo today?",
        "Should I buy Bitcoin crypto stock right now?",
        "Tell me a funny joke",
        "Give me a recipe for chocolate cake",
    ]
    for q in queries:
        result = _heuristic_classify(q)
        assert result.intent == IntentType.OUT_OF_SCOPE, f"Failed for: {q}"


def test_heuristic_classify_clarify_needed():
    """Heuristic marks short or vague requests as clarify_needed."""
    vague_queries = ["help", "something", "idk", "stuff"]
    for q in vague_queries:
        result = _heuristic_classify(q)
        assert result.intent == IntentType.CLARIFY_NEEDED, f"Failed for: {q}"


def test_intent_classifier_empty_string():
    """Empty or whitespace-only inputs immediately flag clarify_needed."""
    classifier = IntentClassifier()
    res1 = classifier.classify("")
    res2 = classifier.classify("   \n\t ")
    assert res1.intent == IntentType.CLARIFY_NEEDED
    assert res1.confidence == 1.0
    assert res2.intent == IntentType.CLARIFY_NEEDED


def test_intent_classifier_injected_provider_returns_dict():
    """Injected provider returning a dictionary is converted to IntentClassification."""
    classifier = IntentClassifier(
        provider=lambda text: {
            "intent": "calendar_query",
            "confidence": 0.88,
            "reasoning": "Injected dict provider",
        }
    )
    result = classifier.classify("Schedule a meeting")
    assert result.intent == IntentType.CALENDAR_QUERY
    assert result.confidence == 0.88
    assert result.reasoning == "Injected dict provider"


def test_intent_classifier_injected_provider_invalid_type():
    """Injected provider returning invalid type raises TypeError."""
    classifier = IntentClassifier(provider=lambda text: 12345)
    with pytest.raises(TypeError):
        classifier.classify("Test")


# ─────────────────────────────────────────────────────────────────────────────
# 3. INDIVIDUAL NODES TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_clarify_node(sample_tenant_id):
    """ClarifyNode sets requires_clarification=True and generates a friendly prompt."""
    node = ClarifyNode()
    state = make_initial_state("help me", sample_tenant_id)
    state["intent"] = IntentType.CLARIFY_NEEDED.value

    result = node(state)
    assert result["agent_mode"] == "clarify"
    assert result["requires_clarification"] is True
    assert isinstance(result["clarification_question"], str)
    assert len(result["clarification_question"]) > 10
    assert result["final_response"] == result["clarification_question"]


def test_clarify_node_custom_question(sample_tenant_id):
    """ClarifyNode respects custom question override."""
    node = ClarifyNode(custom_question="Can you specify which department?")
    state = make_initial_state("need contact", sample_tenant_id)
    result = node(state)
    assert result["clarification_question"] == "Can you specify which department?"
    assert result["final_response"] == "Can you specify which department?"


def test_decline_node(sample_tenant_id):
    """DeclineNode sets agent_mode='decline' and returns polite scope refusal."""
    node = DeclineNode()
    state = make_initial_state("buy tesla stocks", sample_tenant_id)
    state["intent"] = IntentType.OUT_OF_SCOPE.value

    result = node(state)
    assert result["agent_mode"] == "decline"
    assert "workplace assistant" in result["final_response"].lower()
    assert result["requires_clarification"] is False


def test_hr_agent_node_stub(sample_tenant_id):
    """HRAgentNode returns safe stub pending response without executing leave."""
    node = HRAgentNode()
    state = make_initial_state("I want to apply for 2 days leave", sample_tenant_id)
    state["intent"] = IntentType.LEAVE_REQUEST.value

    result = node(state)
    assert result["agent_mode"] == "tool_intent"
    assert len(result["tool_intents"]) == 1
    intent_ref = result["tool_intents"][0]
    assert intent_ref["tool"] == "hr_leave_system"
    assert intent_ref["intent"] == "leave_request"
    assert intent_ref["status"] == "stub_pending"
    assert "Module 6" in intent_ref["module_planned"]
    assert "leave request" in result["final_response"].lower()


def test_tool_router_node_routes_all_tool_types(sample_tenant_id):
    """ToolRouterNode handles all supported tool intents as safe stubs."""
    router = ToolRouterNode()

    tool_tests = [
        (IntentType.LEAVE_REQUEST, "hr_leave_system"),
        (IntentType.IT_SUPPORT, "it_helpdesk_system"),
        (IntentType.CALENDAR_QUERY, "calendar_system"),
        (IntentType.TASK_MANAGEMENT, "task_management_system"),
        (IntentType.TIMESHEET, "timesheet_system"),
        (IntentType.NAVIGATION, "indoor_navigation_system"),
    ]

    for intent_type, expected_tool in tool_tests:
        state = make_initial_state(f"Perform {intent_type.value}", sample_tenant_id)
        state["intent"] = intent_type.value

        result = router(state)
        assert result["agent_mode"] == "tool_intent"
        assert len(result["tool_intents"]) == 1
        ref = result["tool_intents"][0]
        assert ref["tool"] == expected_tool
        assert ref["status"] == "stub_pending"


def test_knowledge_agent_node_success(sample_tenant_id, mock_rag_service):
    """KnowledgeAgentNode queries RAGService and populates rag_response & final_response."""
    node = KnowledgeAgentNode(rag_service=mock_rag_service)
    state = make_initial_state("What is the annual leave allowance?", sample_tenant_id)
    state["intent"] = IntentType.KNOWLEDGE_QUERY.value

    result = node(state)
    assert result["agent_mode"] == "rag"
    assert "25 days" in result["final_response"]
    assert result["rag_response"]["confidence"] == 0.92
    assert len(result["rag_response"]["sources"]) == 1


def test_knowledge_agent_node_handles_invalid_tenant(mock_rag_service):
    """KnowledgeAgentNode safely handles non-UUID tenant without crashing."""
    node = KnowledgeAgentNode(rag_service=mock_rag_service)
    state = make_initial_state("Question", tenant_id="not-a-uuid")
    state["intent"] = IntentType.KNOWLEDGE_QUERY.value

    result = node(state)
    assert result["agent_mode"] == "rag"
    assert "Invalid tenant_id" in result["error"]
    assert "could not find" in result["final_response"].lower()


def test_knowledge_agent_node_handles_service_exception(sample_tenant_id):
    """KnowledgeAgentNode safely catches RAGService runtime errors."""
    failing_service = MagicMock()
    failing_service.answer_question.side_effect = RuntimeError("Database timeout")
    node = KnowledgeAgentNode(rag_service=failing_service)

    state = make_initial_state("Question", sample_tenant_id)
    state["intent"] = IntentType.KNOWLEDGE_QUERY.value

    result = node(state)
    assert result["agent_mode"] == "rag"
    assert "error while searching the knowledge base" in result["final_response"].lower()
    assert "Database timeout" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONDITIONAL ROUTING & ORCHESTRATOR GRAPH TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_route_after_classify_routing_logic(sample_tenant_id):
    """_route_after_classify maps IntentType strings to the correct graph node."""
    state = make_initial_state("test", sample_tenant_id)

    state["intent"] = IntentType.KNOWLEDGE_QUERY.value
    assert _route_after_classify(state) == "knowledge_agent"

    state["intent"] = IntentType.LEAVE_REQUEST.value
    assert _route_after_classify(state) == "tool_router"

    state["intent"] = IntentType.IT_SUPPORT.value
    assert _route_after_classify(state) == "tool_router"

    state["intent"] = IntentType.CLARIFY_NEEDED.value
    assert _route_after_classify(state) == "clarify"

    state["intent"] = IntentType.OUT_OF_SCOPE.value
    assert _route_after_classify(state) == "decline"

    # Fallback on unknown intent
    state["intent"] = "completely_unknown_future_intent"
    assert _route_after_classify(state) == "clarify"

    # Route to clarify if error exists in state
    state["error"] = "Classification failed"
    assert _route_after_classify(state) == "clarify"


def test_orchestrator_full_flow_knowledge_query(test_orchestrator, sample_tenant_id):
    """Orchestrator correctly processes a knowledge policy query through RAG."""
    final_state = test_orchestrator.run(
        request="What is the policy on annual leave?",
        tenant_id=sample_tenant_id,
    )
    assert final_state["agent_mode"] == "rag"
    assert final_state["intent"] == IntentType.KNOWLEDGE_QUERY.value
    assert "25 days" in final_state["final_response"]
    assert final_state["requires_clarification"] is False


def test_orchestrator_full_flow_tool_intent(test_orchestrator, sample_tenant_id):
    """Orchestrator correctly routes leave requests to safe tool stub."""
    final_state = test_orchestrator.run(
        request="I want to apply for leave tomorrow",
        tenant_id=sample_tenant_id,
    )
    assert final_state["agent_mode"] == "tool_intent"
    assert final_state["intent"] == IntentType.LEAVE_REQUEST.value
    assert len(final_state["tool_intents"]) == 1
    assert final_state["tool_intents"][0]["status"] == "stub_pending"


def test_orchestrator_full_flow_decline(test_orchestrator, sample_tenant_id):
    """Orchestrator safely declines out-of-scope requests."""
    final_state = test_orchestrator.run(
        request="Tell me a funny joke",
        tenant_id=sample_tenant_id,
    )
    assert final_state["agent_mode"] == "decline"
    assert final_state["intent"] == IntentType.OUT_OF_SCOPE.value
    assert "workplace assistant" in final_state["final_response"].lower()


def test_orchestrator_full_flow_clarify(test_orchestrator, sample_tenant_id):
    """Orchestrator prompts for clarification on vague requests."""
    final_state = test_orchestrator.run(
        request="something unclear",
        tenant_id=sample_tenant_id,
    )
    assert final_state["agent_mode"] == "clarify"
    assert final_state["intent"] == IntentType.CLARIFY_NEEDED.value
    assert final_state["requires_clarification"] is True
    assert final_state["clarification_question"] is not None


def test_orchestrator_empty_request_short_circuit(test_orchestrator, sample_tenant_id):
    """Empty request string short-circuits with clarification without graph failure."""
    final_state = test_orchestrator.run(
        request="   ",
        tenant_id=sample_tenant_id,
    )
    assert final_state["agent_mode"] == "clarify"
    assert final_state["requires_clarification"] is True
    assert "Please enter your question" in final_state["final_response"]


def test_orchestrator_preserves_tenant_id(test_orchestrator):
    """Tenant ID is strictly preserved across the entire orchestrator lifecycle."""
    custom_tenant = "11111111-2222-3333-4444-555555555555"
    final_state = test_orchestrator.run(
        request="Company policy details",
        tenant_id=custom_tenant,
    )
    assert final_state["tenant_id"] == custom_tenant


# ─────────────────────────────────────────────────────────────────────────────
# 5. FASTAPI /api/agent ENDPOINT INTEGRATION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_api_agent_knowledge_success(test_orchestrator, sample_tenant_id):
    """POST /api/agent returns 200 and formatted AgentResponse for knowledge query."""
    app.dependency_overrides[_get_orchestrator] = lambda: test_orchestrator
    client = TestClient(app)

    payload = {
        "request": "What is the policy on vacation?",
        "tenant_id": sample_tenant_id,
    }
    response = client.post("/api/agent", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["agent_mode"] == "rag"
    assert data["intent"] == "knowledge_query"
    assert "25 days" in data["response"]
    assert len(data["rag_sources"]) == 1
    assert data["requires_clarification"] is False


def test_api_agent_tool_intent_success(test_orchestrator, sample_tenant_id):
    """POST /api/agent returns 200 and formatted stub references for tool actions."""
    app.dependency_overrides[_get_orchestrator] = lambda: test_orchestrator
    client = TestClient(app)

    payload = {
        "request": "Apply for leave tomorrow",
        "tenant_id": sample_tenant_id,
    }
    response = client.post("/api/agent", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["agent_mode"] == "tool_intent"
    assert data["intent"] == "leave_request"
    assert len(data["tool_intents"]) == 1
    assert data["tool_intents"][0]["status"] == "stub_pending"


def test_api_agent_clarification_response(test_orchestrator, sample_tenant_id):
    """POST /api/agent returns clarification fields when query is unclear."""
    app.dependency_overrides[_get_orchestrator] = lambda: test_orchestrator
    client = TestClient(app)

    payload = {
        "request": "unclear request",
        "tenant_id": sample_tenant_id,
    }
    response = client.post("/api/agent", json=payload)
    app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["agent_mode"] == "clarify"
    assert data["requires_clarification"] is True
    assert data["clarification_question"] is not None


def test_api_agent_invalid_tenant_format():
    """POST /api/agent returns 400 Bad Request if tenant_id is not a valid UUID."""
    client = TestClient(app)
    payload = {
        "request": "Any request",
        "tenant_id": "not-a-valid-uuid-string",
    }
    response = client.post("/api/agent", json=payload)
    assert response.status_code == 400
    assert "Invalid tenant_id" in response.json()["detail"]


def test_api_agent_blank_request_validation():
    """POST /api/agent returns 422 Unprocessable Entity if request text is blank."""
    client = TestClient(app)
    payload = {
        "request": "   ",
    }
    response = client.post("/api/agent", json=payload)
    assert response.status_code == 422
