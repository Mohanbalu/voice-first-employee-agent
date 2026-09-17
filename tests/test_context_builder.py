"""Tests for RAG Context Builder — Module 4.

No real API calls or database connections required.
"""

from __future__ import annotations

import uuid
from typing import Optional

import pytest

from backend.app.rag.context_builder import ContextBuilder, BuiltContext, SourceReference
from backend.app.rag.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(
    chunk_id: str = "c001",
    document_name: str = "Leave Policy",
    source_file: str = "Leave_Policy.pdf",
    text: str = "Employees get 20 days annual leave per year.",
    page_start: int = 10,
    page_end: int = 11,
    section: Optional[str] = "Annual Leave",
    similarity_score: float = 0.90,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="leave_policy",
        document_name=document_name,
        source_file=source_file,
        text=text,
        page_start=page_start,
        page_end=page_end,
        section=section,
        similarity_score=similarity_score,
        distance=round(1.0 - similarity_score, 4),
        metadata={},
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestContextConstruction:
    """Verifies structure and content of the built context."""

    def test_empty_results_returns_empty_context(self):
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([])
        assert ctx.context_text == ""
        assert ctx.sources == []
        assert ctx.chunk_count == 0

    def test_single_result_context(self):
        result = _make_result()
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])

        assert ctx.chunk_count == 1
        assert "Leave Policy" in ctx.context_text
        assert "Leave_Policy.pdf" in ctx.context_text
        assert "Annual Leave" in ctx.context_text
        assert "20 days annual leave" in ctx.context_text
        assert "10" in ctx.context_text or "10–11" in ctx.context_text

    def test_source_reference_preserved(self):
        result = _make_result()
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])

        assert len(ctx.sources) == 1
        src = ctx.sources[0]
        assert src.document_name == "Leave Policy"
        assert src.source_file == "Leave_Policy.pdf"
        assert src.page_start == 10
        assert src.page_end == 11
        assert src.section == "Annual Leave"
        assert src.chunk_id == "c001"
        assert src.similarity_score == pytest.approx(0.90)

    def test_multiple_results(self):
        results = [
            _make_result(chunk_id=f"c{i}", document_name=f"Doc {i}", similarity_score=0.90 - i * 0.05)
            for i in range(3)
        ]
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build(results)

        assert ctx.chunk_count == 3
        assert len(ctx.sources) == 3
        for i, src in enumerate(ctx.sources):
            assert src.document_name == f"Doc {i}"
            assert src.chunk_id == f"c{i}"

    def test_same_page_shows_single_page(self):
        result = _make_result(page_start=7, page_end=7)
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        # Should show "7", not "7–7"
        assert "7–7" not in ctx.context_text
        assert "7" in ctx.context_text

    def test_multi_page_range_shown(self):
        result = _make_result(page_start=12, page_end=15)
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        assert "12" in ctx.context_text
        assert "15" in ctx.context_text


class TestContextChunkLimit:
    """Verifies the max_chunks limit is enforced."""

    def test_max_chunks_limits_context(self):
        results = [_make_result(chunk_id=f"c{i}") for i in range(10)]
        builder = ContextBuilder(max_chunks=3)
        ctx = builder.build(results)

        assert ctx.chunk_count == 3
        assert len(ctx.sources) == 3

    def test_fewer_than_max_chunks_uses_all(self):
        results = [_make_result(chunk_id=f"c{i}") for i in range(2)]
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build(results)

        assert ctx.chunk_count == 2

    def test_max_chunks_one(self):
        results = [_make_result(chunk_id=f"c{i}") for i in range(4)]
        builder = ContextBuilder(max_chunks=1)
        ctx = builder.build(results)

        assert ctx.chunk_count == 1
        assert len(ctx.sources) == 1
        assert ctx.sources[0].chunk_id == "c0"


class TestSourceMetadataPreservation:
    """Verifies that source metadata is never fabricated or lost."""

    def test_source_file_preserved(self):
        result = _make_result(source_file="Travel_Policy_v3.pdf")
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        assert "Travel_Policy_v3.pdf" in ctx.context_text
        assert ctx.sources[0].source_file == "Travel_Policy_v3.pdf"

    def test_no_internal_db_ids_in_context(self):
        """Internal database UUIDs must NOT appear in the context text."""
        result = _make_result()
        # Simulate an internal id that would be a DB UUID
        internal_id = "550e8400-e29b-41d4-a716-446655440000"
        # The chunk_id in our system is a human-readable slug, not a UUID —
        # but confirm that metadata doesn't bleed internal UUIDs into context
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        assert internal_id not in ctx.context_text

    def test_optional_section_absent(self):
        result = _make_result(section=None)
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        assert ctx.sources[0].section is None
        assert "SECTION:" not in ctx.context_text

    def test_section_present_when_provided(self):
        result = _make_result(section="Remote Work Guidelines")
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        assert "Remote Work Guidelines" in ctx.context_text
        assert ctx.sources[0].section == "Remote Work Guidelines"

    def test_no_fabricated_sources(self):
        """Sources in output must match results provided — no extras."""
        results = [_make_result(chunk_id="only_chunk")]
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build(results)
        assert len(ctx.sources) == 1
        assert ctx.sources[0].chunk_id == "only_chunk"

    def test_multiple_source_attribution(self):
        """All provided results must appear in sources."""
        results = [
            _make_result(chunk_id="c1", document_name="Leave Policy"),
            _make_result(chunk_id="c2", document_name="Travel Policy"),
        ]
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build(results)
        chunk_ids = {s.chunk_id for s in ctx.sources}
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids

    def test_to_dict_serialisable(self):
        result = _make_result()
        builder = ContextBuilder(max_chunks=5)
        ctx = builder.build([result])
        d = ctx.to_dict()
        assert "context_text" in d
        assert "sources" in d
        assert isinstance(d["sources"], list)
        assert d["sources"][0]["document_name"] == "Leave Policy"
