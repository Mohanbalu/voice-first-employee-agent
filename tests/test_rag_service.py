"""Tests for RAG Service — Module 4.

Uses fully injected mocks for retriever, context builder, and answer generator.
No real OpenAI calls. No live database required.

Synthetic test fixtures:
- Tenant A: Leave Policy chunks (Annual Leave, Sick Leave)
- Tenant B: Travel Policy chunks (Flight Booking)
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from backend.app.rag.rag_service import RAGService, RAGResponse, _derive_confidence
from backend.app.rag.retriever import RAGRetriever, RetrievalConfig
from backend.app.rag.context_builder import ContextBuilder, BuiltContext, SourceReference
from backend.app.rag.answer_generator import AnswerGenerator, GeneratedAnswer, NO_CONTEXT_ANSWER
from backend.app.rag.vector_store import SearchResult


# ── Synthetic Fixtures ────────────────────────────────────────────────────────

TENANT_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
TENANT_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")

ANNUAL_LEAVE_CHUNK = SearchResult(
    chunk_id="leave_policy_annual_0001",
    document_id="leave_policy",
    document_name="Leave and Holiday Policy",
    source_file="Leave_Holiday_Policy.pdf",
    text="Employees are entitled to 20 days of annual leave per calendar year.",
    page_start=12,
    page_end=13,
    section="Annual Leave",
    similarity_score=0.92,
    distance=0.08,
    metadata={"tenant_id": str(TENANT_A_ID)},
)

SICK_LEAVE_CHUNK = SearchResult(
    chunk_id="leave_policy_sick_0002",
    document_id="leave_policy",
    document_name="Leave and Holiday Policy",
    source_file="Leave_Holiday_Policy.pdf",
    text="Employees may take up to 10 days of paid sick leave per year.",
    page_start=14,
    page_end=14,
    section="Sick Leave",
    similarity_score=0.81,
    distance=0.19,
    metadata={"tenant_id": str(TENANT_A_ID)},
)

TENANT_B_TRAVEL_CHUNK = SearchResult(
    chunk_id="travel_policy_flights_0001",
    document_id="travel_policy",
    document_name="Travel Policy",
    source_file="Travel_Policy.pdf",
    text="Employees must book economy class flights unless pre-approved.",
    page_start=5,
    page_end=6,
    section="Flight Booking",
    similarity_score=0.88,
    distance=0.12,
    metadata={"tenant_id": str(TENANT_B_ID)},
)


def _make_tenant(tenant_id: uuid.UUID, name: str = "Test Org") -> MagicMock:
    """Returns a lightweight tenant mock — no SQLAlchemy ORM instrumentation needed."""
    t = MagicMock()
    t.id = tenant_id
    t.name = name
    t.slug = name.lower().replace(" ", "-")
    return t


def _make_session(tenant=None):
    session = MagicMock()
    if tenant is not None:
        session.execute.return_value.scalar_one_or_none.return_value = tenant
    else:
        session.execute.return_value.scalar_one_or_none.return_value = None
    return session


def _stub_retriever(results):
    retriever = MagicMock(spec=RAGRetriever)
    retriever.retrieve.return_value = results
    return retriever


def _stub_answer(answer_text: str = "You have 20 days of annual leave."):
    generator = MagicMock(spec=AnswerGenerator)
    generator.generate.return_value = GeneratedAnswer(
        answer=answer_text,
        model="gpt-4o-mini",
        prompt_tokens=80,
        completion_tokens=40,
        total_tokens=120,
    )
    return generator


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRAGServicePipeline:
    """Full pipeline integration with injected mocks."""

    def test_successful_answer_with_sources(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=_stub_retriever([ANNUAL_LEAVE_CHUNK]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("You get 20 days annual leave."),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="How many annual leave days?",
            session=session,
        )

        assert isinstance(resp, RAGResponse)
        assert resp.error is None
        assert "20 days" in resp.answer
        assert len(resp.sources) == 1
        assert resp.sources[0].chunk_id == "leave_policy_annual_0001"
        assert resp.sources[0].document_name == "Leave and Holiday Policy"
        assert resp.sources[0].page_start == 12
        assert resp.retrieved_count == 1
        assert resp.used_context_count == 1

    def test_no_match_returns_no_context_answer(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=_stub_retriever([]),  # empty retrieval
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=MagicMock(spec=AnswerGenerator),
        )
        service._answer_generator.generate.return_value = GeneratedAnswer(
            answer=NO_CONTEXT_ANSWER,
            model="gpt-4o-mini",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="What is our policy for flying to Mars?",
            session=session,
        )

        assert resp.retrieved_count == 0
        assert resp.used_context_count == 0
        assert resp.confidence == "NO_MATCH"
        assert NO_CONTEXT_ANSWER in resp.answer or "could not find" in resp.answer.lower()

    def test_multiple_sources_in_response(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=_stub_retriever([ANNUAL_LEAVE_CHUNK, SICK_LEAVE_CHUNK]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("You get 20 annual + 10 sick leave days."),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="What leave am I entitled to?",
            session=session,
        )

        assert len(resp.sources) == 2
        chunk_ids = {s.chunk_id for s in resp.sources}
        assert "leave_policy_annual_0001" in chunk_ids
        assert "leave_policy_sick_0002" in chunk_ids

    def test_invalid_tenant_returns_error(self):
        session = _make_session(tenant=None)  # no tenant found

        service = RAGService(
            retriever=MagicMock(spec=RAGRetriever),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=MagicMock(spec=AnswerGenerator),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="Leave policy?",
            session=session,
        )

        assert resp.error is not None
        assert "not found" in resp.error.lower() or resp.confidence == "NO_MATCH"

    def test_empty_question_returns_error(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=MagicMock(spec=RAGRetriever),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=MagicMock(spec=AnswerGenerator),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="",
            session=session,
        )

        assert resp.error is not None

    def test_retrieval_error_returns_safe_error(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.side_effect = Exception("DB connection failed")

        service = RAGService(
            retriever=retriever,
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=MagicMock(spec=AnswerGenerator),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="Leave policy?",
            session=session,
        )

        assert resp.error is not None
        assert "service error" in resp.error.lower() or resp.confidence == "NO_MATCH"

    def test_llm_failure_returns_safe_error(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        generator = MagicMock(spec=AnswerGenerator)
        generator.generate.side_effect = Exception("OpenAI API unavailable")

        service = RAGService(
            retriever=_stub_retriever([ANNUAL_LEAVE_CHUNK]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=generator,
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID,
            question="Leave?",
            session=session,
        )

        assert resp.error is not None
        assert "service error" in resp.error.lower() or resp.confidence == "NO_MATCH"


class TestCrossTenantIsolation:
    """RAGService must enforce tenant isolation through the retriever."""

    def test_tenant_a_only_gets_tenant_a_results(self):
        tenant_a = _make_tenant(TENANT_A_ID, "Org A")
        session = _make_session(tenant_a)

        captured_tenant_ids = []

        def track_retrieve(session, tenant_id, question):
            captured_tenant_ids.append(tenant_id)
            return [ANNUAL_LEAVE_CHUNK]

        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.side_effect = track_retrieve

        service = RAGService(
            retriever=retriever,
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("20 days."),
        )

        service.answer_question(
            tenant_id=TENANT_A_ID,
            question="Leave days?",
            session=session,
        )

        assert captured_tenant_ids == [TENANT_A_ID]

    def test_tenant_b_cannot_see_tenant_a_chunks(self):
        """Retriever is called with TENANT_B_ID — it must NOT return Tenant A data."""
        tenant_b = _make_tenant(TENANT_B_ID, "Org B")
        session = _make_session(tenant_b)

        def tenant_b_retrieve(session, tenant_id, question):
            # Simulate DB returning only Tenant B's data
            assert tenant_id == TENANT_B_ID, "Must query Tenant B, not Tenant A"
            return [TENANT_B_TRAVEL_CHUNK]

        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.side_effect = tenant_b_retrieve

        service = RAGService(
            retriever=retriever,
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("Economy class required."),
        )

        resp = service.answer_question(
            tenant_id=TENANT_B_ID,
            question="Travel policy?",
            session=session,
        )

        for src in resp.sources:
            assert src.chunk_id != ANNUAL_LEAVE_CHUNK.chunk_id, (
                "Tenant B must not receive Tenant A's Leave Policy chunks"
            )


class TestConfidenceLabels:
    """Verifies confidence label derivation from retrieval scores."""

    def test_high_confidence_above_threshold(self):
        sources = [SourceReference(
            document_name="Policy", source_file="p.pdf",
            page_start=1, page_end=2, section=None,
            chunk_id="c1", similarity_score=0.90,
        )]
        assert _derive_confidence(sources) == "HIGH"

    def test_medium_confidence(self):
        sources = [SourceReference(
            document_name="Policy", source_file="p.pdf",
            page_start=1, page_end=2, section=None,
            chunk_id="c1", similarity_score=0.75,
        )]
        assert _derive_confidence(sources) == "MEDIUM"

    def test_low_confidence(self):
        sources = [SourceReference(
            document_name="Policy", source_file="p.pdf",
            page_start=1, page_end=2, section=None,
            chunk_id="c1", similarity_score=0.55,
        )]
        assert _derive_confidence(sources) == "LOW"

    def test_no_match_empty_sources(self):
        assert _derive_confidence([]) == "NO_MATCH"

    def test_no_match_below_all_thresholds(self):
        sources = [SourceReference(
            document_name="Policy", source_file="p.pdf",
            page_start=1, page_end=2, section=None,
            chunk_id="c1", similarity_score=0.20,
        )]
        assert _derive_confidence(sources) == "NO_MATCH"


class TestResponseStructure:
    """Verifies RAGResponse serialises correctly and contains expected fields."""

    def test_to_dict_contains_all_required_fields(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=_stub_retriever([ANNUAL_LEAVE_CHUNK]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("20 days."),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID, question="Leave?", session=session
        )

        d = resp.to_dict()
        required_keys = {
            "answer", "sources", "retrieved_count", "used_context_count",
            "confidence", "mode", "model", "latency_ms", "error",
        }
        assert required_keys.issubset(d.keys())

    def test_no_raw_embeddings_in_response(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        service = RAGService(
            retriever=_stub_retriever([ANNUAL_LEAVE_CHUNK]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer("20 days."),
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID, question="Leave?", session=session
        )

        d = resp.to_dict()
        # No raw embedding vectors should appear in the structured response
        assert "embedding" not in d
        for src in d["sources"]:
            assert "embedding" not in src

    def test_mode_production_when_allow_mock_false(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        cfg = RetrievalConfig(allow_mock=False)
        service = RAGService(
            retriever=_stub_retriever([]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer(),
            retrieval_config=cfg,
        )
        service._answer_generator.generate.return_value = GeneratedAnswer(
            answer=NO_CONTEXT_ANSWER, model="gpt-4o-mini",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID, question="Mars policy?", session=session
        )

        assert resp.mode == "production"

    def test_mode_mock_when_allow_mock_true(self):
        tenant = _make_tenant(TENANT_A_ID)
        session = _make_session(tenant)

        cfg = RetrievalConfig(allow_mock=True)
        service = RAGService(
            retriever=_stub_retriever([]),
            context_builder=ContextBuilder(max_chunks=5),
            answer_generator=_stub_answer(),
            retrieval_config=cfg,
        )
        service._answer_generator.generate.return_value = GeneratedAnswer(
            answer=NO_CONTEXT_ANSWER, model="gpt-4o-mini",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
        )

        resp = service.answer_question(
            tenant_id=TENANT_A_ID, question="Mars policy?", session=session
        )

        assert resp.mode == "mock_development"
