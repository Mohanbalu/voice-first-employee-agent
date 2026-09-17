"""Tests for RAG Retriever — Module 4.

All tests use mocked embedding providers and vector stores.
No real OpenAI API calls. No live database required.
"""

from __future__ import annotations

import math
import uuid
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from backend.app.rag.retriever import RAGRetriever, RetrievalConfig
from backend.app.rag.embeddings import EmbeddingService, MockEmbeddingProvider
from backend.app.rag.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_result(
    chunk_id: str = "chunk_001",
    document_name: str = "Leave Policy",
    text: str = "Employees are entitled to 20 days annual leave.",
    page_start: int = 5,
    page_end: int = 6,
    section: str = "Annual Leave",
    similarity_score: float = 0.92,
    tenant_id: Optional[uuid.UUID] = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="leave_policy",
        document_name=document_name,
        source_file="Leave_Policy.pdf",
        text=text,
        page_start=page_start,
        page_end=page_end,
        section=section,
        similarity_score=similarity_score,
        distance=round(1.0 - similarity_score, 4),
        metadata={},
    )


def _mock_embedding_service(dimension: int = 4) -> EmbeddingService:
    """Returns a deterministic mock embedding service for testing."""
    provider = MockEmbeddingProvider(dimension=dimension)
    return EmbeddingService(provider)


def _make_session(search_results: List[SearchResult], is_mock_map: dict = None):
    """
    Returns a mock SQLAlchemy session:
    - search_similar_chunks returns search_results
    - ChunkEmbedding.is_mock lookups return values from is_mock_map
    """
    session = MagicMock()

    # is_mock DB lookup — returns rows with chunk_id and is_mock fields
    if is_mock_map is not None:
        rows = []
        for chunk_id, is_mock in is_mock_map.items():
            row = MagicMock()
            row.chunk_id = chunk_id
            row.is_mock = is_mock
            rows.append(row)
        session.execute.return_value.fetchall.return_value = rows
    else:
        session.execute.return_value.fetchall.return_value = []

    return session


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestQueryEmbedding:
    """Verifies query embedding generation through the retriever."""

    def test_mock_provider_embeds_query(self):
        svc = _mock_embedding_service(dimension=4)
        vec = svc.embed_text("How many days leave?")
        assert isinstance(vec, list)
        assert len(vec) == 4
        assert all(isinstance(v, float) for v in vec)

    def test_different_questions_produce_different_vectors(self):
        svc = _mock_embedding_service(dimension=4)
        v1 = svc.embed_text("Leave policy?")
        v2 = svc.embed_text("Travel reimbursement?")
        assert v1 != v2

    def test_empty_question_raises(self):
        cfg = RetrievalConfig(allow_mock=True)
        retriever = RAGRetriever(
            embedding_service=_mock_embedding_service(dimension=4),
            config=cfg,
        )
        session = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            retriever.retrieve(session=session, tenant_id=uuid.uuid4(), question="")

    def test_whitespace_question_raises(self):
        cfg = RetrievalConfig(allow_mock=True)
        retriever = RAGRetriever(
            embedding_service=_mock_embedding_service(dimension=4),
            config=cfg,
        )
        session = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            retriever.retrieve(session=session, tenant_id=uuid.uuid4(), question="   ")


class TestVectorRetrieval:
    """Tests retrieval flow with mocked vector store."""

    def test_retrieval_returns_results_above_threshold(self):
        result = _make_mock_result(similarity_score=0.85)
        cfg = RetrievalConfig(min_similarity=0.70, allow_mock=True, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        session = _make_session(
            [result],
            is_mock_map={"chunk_001": True},
        )

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = [result]
                results = retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="How many leave days?",
                )

        assert len(results) == 1
        assert results[0].similarity_score == 0.85

    def test_results_below_threshold_are_dropped(self):
        low_result = _make_mock_result(similarity_score=0.40)
        cfg = RetrievalConfig(min_similarity=0.70, allow_mock=True, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        session = _make_session([], is_mock_map={})

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = [low_result]
                results = retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="Leave policy?",
                )

        assert results == []

    def test_top_k_limits_candidates(self):
        results_list = [_make_mock_result(chunk_id=f"c{i}", similarity_score=0.90 - i * 0.01) for i in range(10)]
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=True, top_k=3, max_context_chunks=3)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        session = _make_session([], is_mock_map={})

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = results_list[:3]
                results = retriever.retrieve(
                    session=session, tenant_id=uuid.uuid4(), question="Policy?"
                )

        assert len(results) <= 3


class TestMockVectorRejection:
    """Ensures mock vectors are rejected in production mode."""

    def test_mock_vectors_rejected_in_production_mode(self):
        result = _make_mock_result(chunk_id="mock_chunk", similarity_score=0.95)
        # allow_mock=False -> production mode
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=False, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        # Session returns is_mock=True for this chunk
        session = _make_session([result], is_mock_map={"mock_chunk": True})

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = [result]
                results = retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="Leave policy?",
                )

        assert all(r.chunk_id != "mock_chunk" for r in results)

    def test_production_vectors_accepted_in_production_mode(self):
        result = _make_mock_result(chunk_id="prod_chunk", similarity_score=0.95)
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=False, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        # Session returns is_mock=False for this chunk
        session = _make_session([result], is_mock_map={"prod_chunk": False})

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = [result]
                results = retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="Leave policy?",
                )

        assert len(results) == 1
        assert results[0].chunk_id == "prod_chunk"

    def test_mock_vectors_allowed_when_flag_set(self):
        result = _make_mock_result(chunk_id="mock_ok", similarity_score=0.95)
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=True, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)

        session = _make_session([result], is_mock_map={"mock_ok": True})

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.return_value = [result]
                results = retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="Leave policy?",
                )

        assert len(results) == 1


class TestCrossTenantIsolation:
    """Confirms retriever enforces tenant isolation through VectorStore."""

    def test_tenant_id_passed_to_vector_store(self):
        tenant_a = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=True, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=_mock_embedding_service(dimension=4), config=cfg)
        session = _make_session([], is_mock_map={})

        captured_tenant_ids = []

        def fake_search(session, tenant_id, query_embedding, top_k, min_similarity):
            captured_tenant_ids.append(tenant_id)
            return []

        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 4
            with patch("backend.app.rag.retriever.VectorStore") as MockVS:
                MockVS.return_value.search_similar_chunks.side_effect = fake_search
                retriever.retrieve(session=session, tenant_id=tenant_a, question="Policy?")

        assert len(captured_tenant_ids) == 1
        assert captured_tenant_ids[0] == tenant_a

    def test_wrong_dimension_raises(self):
        # If the embedding service returns a vector of wrong dimension, should raise
        svc = _mock_embedding_service(dimension=4)
        cfg = RetrievalConfig(min_similarity=0.0, allow_mock=True, top_k=5, max_context_chunks=5)
        retriever = RAGRetriever(embedding_service=svc, config=cfg)
        session = MagicMock()

        # Patch config.db.vector_dimension to 1536 so the 4-dim vector is "wrong"
        with patch("backend.app.rag.retriever.config") as mock_cfg:
            mock_cfg.db.vector_dimension = 1536
            with pytest.raises(ValueError, match="dimension"):
                retriever.retrieve(
                    session=session,
                    tenant_id=uuid.uuid4(),
                    question="Leave days?",
                )
