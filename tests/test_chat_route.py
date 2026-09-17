"""Tests for Chat API Route — Module 4.

Uses FastAPI TestClient with mocked RAGService.
No real database or OpenAI calls.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Create a minimal FastAPI app for testing without loading the full app
from fastapi import FastAPI

from backend.app.rag.rag_service import RAGResponse
from backend.app.rag.context_builder import SourceReference
from backend.app.rag.answer_generator import NO_CONTEXT_ANSWER


# ── Helpers ───────────────────────────────────────────────────────────────────

DEV_TENANT_ID = "00000000-0000-0000-0000-000000000001"

_SOURCE = SourceReference(
    document_name="Leave and Holiday Policy",
    source_file="Leave_Holiday_Policy.pdf",
    page_start=12,
    page_end=13,
    section="Annual Leave",
    chunk_id="leave_policy_annual_0001",
    similarity_score=0.92,
)

def _success_rag_response() -> RAGResponse:
    return RAGResponse(
        answer="You are entitled to 20 days of annual leave.",
        sources=[_SOURCE],
        retrieved_count=4,
        used_context_count=1,
        confidence="HIGH",
        mode="production",
        model="gpt-4o-mini",
        latency_ms=320.5,
    )

def _no_match_rag_response() -> RAGResponse:
    return RAGResponse(
        answer=NO_CONTEXT_ANSWER,
        sources=[],
        retrieved_count=0,
        used_context_count=0,
        confidence="NO_MATCH",
        mode="production",
        model="gpt-4o-mini",
        latency_ms=80.0,
    )


def _make_client():
    """Creates a TestClient with the chat router and mocked dependencies."""
    from backend.app.routes.chat import router, _get_rag_service
    from backend.app.database import get_db

    app = FastAPI()

    # Override DB dependency — return a mock session
    mock_db = MagicMock()
    app.dependency_overrides[get_db] = lambda: mock_db

    app.include_router(router)
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChatRouteSuccess:
    """POST /api/chat happy-path tests."""

    def test_successful_response_structure(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={
                "question": "How many leave days do I get?",
                "tenant_id": DEV_TENANT_ID,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "retrieved_count" in data
        assert "used_context_count" in data
        assert "confidence" in data
        assert "mode" in data
        assert "model" in data
        assert "latency_ms" in data

    def test_answer_populated(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={"question": "Leave days?", "tenant_id": DEV_TENANT_ID})

        assert resp.status_code == 200
        assert "20 days" in resp.json()["answer"]

    def test_sources_contain_expected_fields(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={"question": "Leave?", "tenant_id": DEV_TENANT_ID})

        sources = resp.json()["sources"]
        assert len(sources) == 1
        src = sources[0]
        assert src["document_name"] == "Leave and Holiday Policy"
        assert src["source_file"] == "Leave_Holiday_Policy.pdf"
        assert src["page_start"] == 12
        assert src["page_end"] == 13
        assert src["section"] == "Annual Leave"
        assert src["chunk_id"] == "leave_policy_annual_0001"
        assert "similarity_score" in src

    def test_no_embeddings_in_response(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={"question": "Leave?", "tenant_id": DEV_TENANT_ID})

        data = resp.json()
        assert "embedding" not in data
        for src in data.get("sources", []):
            assert "embedding" not in src

    def test_no_secret_in_response(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={"question": "Leave?", "tenant_id": DEV_TENANT_ID})

        text = resp.text
        assert "sk-" not in text
        assert "password" not in text.lower()


class TestChatRouteValidation:
    """Request validation tests."""

    def test_empty_question_returns_422(self):
        client = _make_client()
        resp = client.post("/api/chat", json={"question": "", "tenant_id": DEV_TENANT_ID})
        assert resp.status_code == 422

    def test_whitespace_question_returns_422(self):
        client = _make_client()
        resp = client.post("/api/chat", json={"question": "   ", "tenant_id": DEV_TENANT_ID})
        assert resp.status_code == 422

    def test_missing_question_returns_422(self):
        client = _make_client()
        resp = client.post("/api/chat", json={"tenant_id": DEV_TENANT_ID})
        assert resp.status_code == 422

    def test_invalid_tenant_uuid_returns_400(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_factory.return_value = MagicMock()
            resp = client.post("/api/chat", json={
                "question": "Leave?",
                "tenant_id": "not-a-valid-uuid",
            })

        assert resp.status_code == 400

    def test_no_tenant_uses_default(self):
        """Omitting tenant_id should use the configured default."""
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _success_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={"question": "Leave?"})

        assert resp.status_code == 200


class TestNoMatchBehavior:
    """Verifies unknown questions return controlled no-match response."""

    def test_no_match_does_not_hallucinate(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _no_match_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={
                "question": "What is the policy for flying to Mars?",
                "tenant_id": DEV_TENANT_ID,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] == "NO_MATCH"
        assert data["retrieved_count"] == 0
        assert data["sources"] == []
        assert "could not find" in data["answer"].lower() or NO_CONTEXT_ANSWER in data["answer"]

    def test_no_match_sources_empty(self):
        client = _make_client()

        with patch("backend.app.routes.chat._get_rag_service") as mock_factory:
            mock_service = MagicMock()
            mock_service.answer_question.return_value = _no_match_rag_response()
            mock_factory.return_value = mock_service

            resp = client.post("/api/chat", json={
                "question": "Company policy on time travel?",
                "tenant_id": DEV_TENANT_ID,
            })

        assert resp.json()["sources"] == []
