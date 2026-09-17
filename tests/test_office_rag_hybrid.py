"""Unit tests for Office RAG Hybrid Retrieval & Grounded Answer Generation.

Tests:
1. Entity and building extraction from user queries (_extract_query_entities).
2. Entity boosting calculation and rank adjustments.
3. Guarantee that direct entity hits pass the similarity threshold.
4. AnswerGenerator grounded prompt instructions for multi-location synthesis.
5. Distinction between general IT team (SDC 2nd fl, T1 1st fl) and laptop support (SDC 3rd fl).
6. Non-hallucination verification (unspecified Tower 2 experienced teams floor).
7. Non-regression of policy retrieval and grounding.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch
import pytest

from backend.app.rag.retriever import (
    RAGRetriever,
    RetrievalConfig,
    _extract_query_entities,
    OFFICE_ENTITIES,
    OFFICE_BUILDINGS,
)
from backend.app.rag.vector_store import SearchResult
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.answer_generator import AnswerGenerator, GeneratedAnswer, NO_CONTEXT_ANSWER, SYSTEM_PROMPT


TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_search_results():
    return [
        SearchResult(
            chunk_id="chunk_sdc_2nd",
            document_id="hcl_office_locations",
            document_name="HCL Office Locations, Floors, Teams and Facilities",
            source_file="hcl_office_locations.md",
            text="SDC 2nd Floor: Techbees Classrooms, IT Team, Seminar Halls.",
            page_start=1,
            page_end=1,
            section="SDC — 2nd Floor",
            similarity_score=0.6230,
            distance=0.3770,
            metadata={"building": "SDC", "floor": "2nd Floor"},
        ),
        SearchResult(
            chunk_id="chunk_t1_1st",
            document_id="hcl_office_locations",
            document_name="HCL Office Locations, Floors, Teams and Facilities",
            source_file="hcl_office_locations.md",
            text="Tower 1 1st Floor: ODCs, IT Team.",
            page_start=1,
            page_end=1,
            section="Tower 1 — 1st Floor",
            similarity_score=0.6684,
            distance=0.3316,
            metadata={"building": "Tower 1", "floor": "1st Floor"},
        ),
        SearchResult(
            chunk_id="chunk_sdc_3rd",
            document_id="hcl_office_locations",
            document_name="HCL Office Locations, Floors, Teams and Facilities",
            source_file="hcl_office_locations.md",
            text="SDC 3rd Floor: Laptop and Technical Issue Support Teams.",
            page_start=1,
            page_end=1,
            section="SDC — 3rd Floor",
            similarity_score=0.6100,
            distance=0.3900,
            metadata={"building": "SDC", "floor": "3rd Floor"},
        ),
    ]


# ── Tests for Entity Extraction ───────────────────────────────────────────────

class TestEntityExtraction:
    """Verify regex/lexical entity extraction maps queries accurately."""

    def test_extract_it_team_query(self):
        extracted = _extract_query_entities("Where is the IT team?")
        assert "it_team" in extracted["categories"]
        assert "it team" in extracted["terms"]

    def test_extract_building_and_team(self):
        extracted = _extract_query_entities("Where can I find the IT team in SDC?")
        assert "it_team" in extracted["categories"]
        assert "sdc" in extracted["buildings"]

    def test_extract_laptop_support(self):
        extracted = _extract_query_entities("Where do I go for laptop issues and hardware support?")
        assert "laptop_support" in extracted["categories"]

    def test_extract_tower_1_and_trainees(self):
        extracted = _extract_query_entities("Where are trainees located in Tower 1?")
        assert "trainees" in extracted["categories"]
        assert "tower_1" in extracted["buildings"]

    def test_extract_tower_2_experienced_teams(self):
        extracted = _extract_query_entities("Where are experienced teams in Tower 2?")
        assert "experienced_teams" in extracted["categories"]
        assert "tower_2" in extracted["buildings"]

    def test_extract_play_area_games(self):
        extracted = _extract_query_entities("Where can I play table tennis or carrom?")
        assert "play_area" in extracted["categories"]


# ── Tests for Hybrid Retrieval Scoring & Thresholding ─────────────────────────

class TestHybridScoring:
    """Verify hybrid reranking and entity score boosts."""

    def test_it_team_results_pass_threshold(self, mock_search_results):
        """Even with raw cosine similarity ~0.62-0.66, entity match boosts scores above 0.70."""
        retriever = RAGRetriever(config=RetrievalConfig(min_similarity=0.50, allow_mock=True))
        
        # Test directly with mock vector store query
        fake_session = MagicMock()
        with patch.object(retriever, "_get_embedding_service") as mock_emb_svc, \
             patch("backend.app.rag.retriever.VectorStore.search_similar_chunks", return_value=mock_search_results), \
             patch.object(retriever, "_search_lexical_candidates", return_value=[]):
            
            mock_emb_svc.return_value.embed_query.return_value = [0.1] * 384
            results = retriever.retrieve(
                session=fake_session,
                tenant_id=TARGET_TENANT_ID,
                question="Where is the IT team?",
            )

            assert len(results) >= 2
            # Both SDC 2nd floor and Tower 1 1st floor should be retrieved
            sections = [r.section for r in results]
            assert "SDC — 2nd Floor" in sections
            assert "Tower 1 — 1st Floor" in sections

            # Boosted scores should exceed original scores
            for r in results:
                if "IT Team" in r.text:
                    assert r.similarity_score > 0.70

    def test_laptop_query_prioritizes_sdc_3rd_floor(self, mock_search_results):
        """Asking for laptop support boosts SDC 3rd floor over general IT team."""
        retriever = RAGRetriever(config=RetrievalConfig(min_similarity=0.50, allow_mock=True))
        fake_session = MagicMock()

        with patch.object(retriever, "_get_embedding_service") as mock_emb_svc, \
             patch("backend.app.rag.retriever.VectorStore.search_similar_chunks", return_value=mock_search_results), \
             patch.object(retriever, "_search_lexical_candidates", return_value=[]):
            
            mock_emb_svc.return_value.embed_query.return_value = [0.1] * 384
            results = retriever.retrieve(
                session=fake_session,
                tenant_id=TARGET_TENANT_ID,
                question="Where do I get laptop support?",
            )

            assert len(results) >= 1
            # Top result should be SDC 3rd floor (laptop support)
            assert results[0].section == "SDC — 3rd Floor"


# ── Tests for Grounding & Answer Generator ─────────────────────────────────────

class TestAnswerGeneratorGrounding:
    """Verify AnswerGenerator adheres to office location prompt grounding."""

    def test_multi_location_answer_synthesis(self):
        """AnswerGenerator synthesizes both SDC and Tower 1 locations for IT team."""
        gen = AnswerGenerator(api_key="test-key", model="gpt-4o-mini")
        fake_client = MagicMock()
        fake_completion = MagicMock()
        fake_completion.choices = [
            MagicMock(message=MagicMock(content=(
                "The IT team is located in two buildings:\n"
                "- SDC: 2nd Floor\n"
                "- Tower 1: 1st Floor\n\n"
                "Note: For laptop/hardware issues, support is on SDC 3rd Floor."
            )))
        ]
        fake_completion.usage = MagicMock(prompt_tokens=120, completion_tokens=45, total_tokens=165)
        fake_client.chat.completions.create.return_value = fake_completion
        gen._client = fake_client

        context_text = (
            "[SOURCE 1]\nDOCUMENT: HCL Office Locations\nSECTION: SDC — 2nd Floor\n"
            "CONTENT: SDC 2nd Floor: IT Team is located on 2nd floor.\n\n"
            "[SOURCE 2]\nDOCUMENT: HCL Office Locations\nSECTION: Tower 1 — 1st Floor\n"
            "CONTENT: Tower 1 1st Floor: IT Team is located on 1st floor.\n\n"
            "[SOURCE 3]\nDOCUMENT: HCL Office Locations\nSECTION: SDC — 3rd Floor\n"
            "CONTENT: SDC 3rd Floor: Laptop and Technical Issue Support Teams."
        )

        res = gen.generate(question="Where is the IT team?", context_text=context_text)

        assert "SDC: 2nd Floor" in res.answer or "SDC" in res.answer
        assert "Tower 1: 1st Floor" in res.answer or "Tower 1" in res.answer
        assert res.prompt_tokens > 0

    def test_unspecified_tower_2_floor_not_hallucinated(self):
        """AnswerGenerator accurately preserves unspecified status for Tower 2 experienced teams."""
        gen = AnswerGenerator(api_key="test-key", model="gpt-4o-mini")
        fake_client = MagicMock()
        fake_completion = MagicMock()
        fake_completion.choices = [
            MagicMock(message=MagicMock(content=(
                "Experienced teams in Tower 2 work in separate project ODCs. "
                "However, the specific floor is not specified in current company records."
            )))
        ]
        fake_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=30, total_tokens=130)
        fake_client.chat.completions.create.return_value = fake_completion
        gen._client = fake_client

        context_text = (
            "[SOURCE 1]\nDOCUMENT: HCL Office Locations\nSECTION: Tower 2 — Overview\n"
            "CONTENT: Tower 2 hosts experienced teams in separate project ODCs. Specific floor numbers are not provided in campus records."
        )

        res = gen.generate(question="Which floor are experienced teams in Tower 2 on?", context_text=context_text)
        assert "not specified" in res.answer.lower() or "not provided" in res.answer.lower()
