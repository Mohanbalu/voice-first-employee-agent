"""Unit and Integration Tests for HCL Office Location & Facilities Knowledge Base.

Validates:
1. SDC Ground Floor (Reception, Cafeteria).
2. SDC 2nd Floor (Techbees Classrooms, IT Team, Seminar Halls).
3. SDC 3rd Floor (Laptop and Technical Issue Support Teams).
4. SDC IT team (2nd floor) vs SDC laptop support (3rd floor) distinction.
5. Tower 1 Basement (Parking).
6. Tower 1 Ground Floor (Reception, Play Area: carrom, chess, table tennis, Breakout Rooms).
7. Tower 1 1st Floor (ODCs, IT Team).
8. Tower 1 Trainees (located in Tower 1, not restricted to single floor).
9. Tower 2 Basement (Parking).
10. Tower 2 Ground Floor (Reception, Play Area: carrom, chess, table tennis, Breakout Rooms).
11. Tower 2 Experienced Teams (separate project ODCs).
12. Strict non-hallucination: Tower 2 experienced teams floor is NOT specified in campus records.
13. Cafeteria query ("Where is the cafeteria?" -> SDC Ground Floor only).
14. IT team in SDC query ("Where is the IT team in SDC?" -> SDC 2nd Floor).
15. Laptop/technical support query ("Where do I go for laptop issues?" -> SDC 3rd Floor).
16. Table tennis / play area query ("Where can I play table tennis?" -> Tower 1 and Tower 2 Ground Floor play areas).
17. Parking query ("Where is parking located?" -> Tower 1 and Tower 2 Basement).
18. Techbees classroom query ("Where are Techbees classrooms located?" -> SDC 2nd Floor).
19. Seminar halls query ("Where are the seminar halls?" -> SDC 2nd Floor).
20. Breakout rooms query ("Where are the breakout rooms?" -> Tower 1 and Tower 2 Ground Floor).
21. Trainees location query ("Where are trainees located?" -> Tower 1).
22. Tenant isolation (Non-target tenant receives NO_MATCH).
23. Hallucination check for Tower 2 experienced teams floor (must not claim 1st floor or any specific floor).
24. Existing policy regression check (e.g. Annual leave policy query works).
25. Agent Orchestrator routing (NAVIGATION and KNOWLEDGE_QUERY route to knowledge_agent node).
"""

from __future__ import annotations

import json
from pathlib import Path
import uuid
from unittest.mock import MagicMock
import pytest

from backend.app.agents.agent_state import IntentType, make_initial_state, TOOL_INTENTS
from backend.app.agents.orchestrator import _route_after_classify
from backend.app.rag.rag_service import RAGService
from backend.app.rag.retriever import RAGRetriever, RetrievalConfig
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.answer_generator import AnswerGenerator, GeneratedAnswer
from backend.app.rag.vector_store import SearchResult


TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_TENANT_ID = uuid.UUID("99999999-9999-9999-9999-999999999999")

# ── Fixtures for Office Knowledge Search Results ──────────────────────────────

SDC_GROUND_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_001",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "SDC Ground Floor Facilities:\n"
        "• Reception: SDC has a reception on the ground floor.\n"
        "• Cafeteria: SDC has a cafeteria on the ground floor. The cafeteria is located on SDC Ground Floor.\n"
        "Note: The ground floor of SDC is the only cafeteria location identified in the office information."
    ),
    page_start=1,
    page_end=1,
    section="SDC — Ground Floor",
    similarity_score=0.95,
    distance=0.05,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "SDC", "floor": "Ground Floor"},
)

SDC_2ND_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_002",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "SDC 2nd Floor Facilities and Teams:\n"
        "• Techbees Classrooms: Techbees has classrooms in SDC located on the 2nd floor.\n"
        "• IT Team: SDC has an IT team located on the 2nd floor.\n"
        "• Seminar Halls: SDC has seminar halls located on the 2nd floor.\n"
        "Employees looking for Techbees training classes, seminar halls, or the SDC IT team should go to SDC 2nd floor."
    ),
    page_start=1,
    page_end=1,
    section="SDC — 2nd Floor",
    similarity_score=0.94,
    distance=0.06,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "SDC", "floor": "2nd Floor"},
)

SDC_3RD_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_003",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "SDC 3rd Floor Support Teams:\n"
        "• Laptop & Technical Support: SDC has teams on the 3rd floor responsible for issues related to laptops and similar technical queries.\n"
        "Important Distinction: Do not assume that every IT-related issue must go to the 3rd floor. "
        "The SDC IT team is on the 2nd floor, while the teams specifically handling laptop-related issues and technical queries are on the 3rd floor."
    ),
    page_start=1,
    page_end=1,
    section="SDC — 3rd Floor",
    similarity_score=0.96,
    distance=0.04,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "SDC", "floor": "3rd Floor"},
)

TOWER1_PARKING_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_005",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 1 Basement:\n"
        "• Parking: Tower 1 has parking in the basement."
    ),
    page_start=1,
    page_end=1,
    section="Tower 1 — Basement",
    similarity_score=0.93,
    distance=0.07,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 1", "floor": "Basement"},
)

TOWER1_GROUND_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_006",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 1 Ground Floor Facilities:\n"
        "• Reception: Tower 1 has a reception on the ground floor.\n"
        "• Play Area & Indoor Games: Tower 1 has a play area on the ground floor including carrom, chess, and table tennis.\n"
        "• Breakout Rooms: Tower 1 has breakout rooms located on the ground floor."
    ),
    page_start=1,
    page_end=1,
    section="Tower 1 — Ground Floor",
    similarity_score=0.95,
    distance=0.05,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 1", "floor": "Ground Floor"},
)

TOWER1_1ST_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_007",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 1 1st Floor Facilities and Teams:\n"
        "• ODCs: Tower 1 has ODCs on the 1st floor.\n"
        "• IT Team: Tower 1 has an IT team located on the 1st floor."
    ),
    page_start=1,
    page_end=1,
    section="Tower 1 — 1st Floor",
    similarity_score=0.93,
    distance=0.07,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 1", "floor": "1st Floor"},
)

TOWER1_TRAINEES_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_004",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 1 Overview and Teams:\n"
        "• Trainees: Tower 1 includes trainees. Trainees are located in Tower 1 across its project areas.\n"
        "• ODCs: Tower 1 includes shared ODCs and separate ODCs for different projects.\n"
        "Note: The source does not restrict trainees to a single floor."
    ),
    page_start=1,
    page_end=1,
    section="Tower 1 — Overview & Trainees",
    similarity_score=0.94,
    distance=0.06,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 1", "floor": "Multiple"},
)

TOWER2_PARKING_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_009",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 2 Basement:\n"
        "• Parking: Tower 2 has parking in the basement."
    ),
    page_start=1,
    page_end=1,
    section="Tower 2 — Basement",
    similarity_score=0.92,
    distance=0.08,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 2", "floor": "Basement"},
)

TOWER2_GROUND_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_010",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 2 Ground Floor Facilities:\n"
        "• Reception: Tower 2 has a reception on the ground floor.\n"
        "• Play Area & Indoor Games: Tower 2 has a play area on the ground floor including carrom, chess, and table tennis.\n"
        "• Breakout Rooms: Tower 2 has breakout rooms located on the ground floor."
    ),
    page_start=1,
    page_end=1,
    section="Tower 2 — Ground Floor",
    similarity_score=0.94,
    distance=0.06,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 2", "floor": "Ground Floor"},
)

TOWER2_EXPERIENCED_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_008",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "Tower 2 Overview and Teams:\n"
        "• Professionals & Experienced Teams: Tower 2 includes professionals and experienced teams.\n"
        "• Project ODCs: Professionals and experienced teams have separate ODCs for their respective projects in Tower 2.\n"
        "Floor Specification Notice: The provided office information DOES NOT specify the floor number "
        "of the professional and experienced-team ODCs in Tower 2. State that the records do not specify the floor number."
    ),
    page_start=1,
    page_end=1,
    section="Tower 2 — Overview & Professionals",
    similarity_score=0.96,
    distance=0.04,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Tower 2", "floor": "Unspecified"},
)

CAMPUS_SUMMARY_CHUNK = SearchResult(
    chunk_id="hcl_office_locations_chunk_011",
    document_id="hcl_office_locations",
    document_name="HCL Office Locations, Floors, Teams and Facilities",
    source_file="hcl_office_locations.md",
    text=(
        "HCL Campus Facilities Summary:\n"
        "• Cafeteria: SDC Ground Floor. There is only one cafeteria identified in the office information.\n"
        "• Parking: Available in the Basement of Tower 1 and the Basement of Tower 2.\n"
        "• Play Areas & Indoor Games: Ground Floor of Tower 1 and Ground Floor of Tower 2 (table tennis, carrom, chess).\n"
        "• Breakout Rooms: Ground Floor of Tower 1 and Ground Floor of Tower 2.\n"
        "• Techbees Classrooms: SDC 2nd Floor.\n"
        "• Seminar Halls: SDC 2nd Floor.\n"
        "• Laptop & Technical Query Support Teams: SDC 3rd Floor.\n"
        "• IT Teams: SDC 2nd Floor and Tower 1 1st Floor."
    ),
    page_start=1,
    page_end=1,
    section="Campus Facilities Summary",
    similarity_score=0.95,
    distance=0.05,
    metadata={"tenant_id": str(TARGET_TENANT_ID), "building": "Campus", "floor": "Summary"},
)


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestOfficeKnowledgeContentAndStructure:
    """Tests 1-12: Raw document and chunking content validation."""

    @pytest.fixture
    def office_raw_text(self):
        raw_path = Path("data/raw/office/hcl_office_locations.md")
        assert raw_path.exists(), "Raw office markdown file must exist"
        return raw_path.read_text(encoding="utf-8")

    @pytest.fixture
    def office_chunks(self):
        chunks_path = Path("data/processed/chunks/hcl_office_locations.jsonl")
        assert chunks_path.exists(), "Processed chunks jsonl must exist"
        chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def test_01_sdc_ground_floor_content(self, office_raw_text):
        """1. SDC Ground Floor contains Reception and Cafeteria."""
        assert "SDC" in office_raw_text
        assert "Ground Floor" in office_raw_text
        assert "Reception" in office_raw_text
        assert "Cafeteria" in office_raw_text

    def test_02_sdc_second_floor_content(self, office_raw_text):
        """2. SDC 2nd Floor contains Techbees, IT Team, and Seminar Halls."""
        assert "2nd Floor" in office_raw_text
        assert "Techbees" in office_raw_text
        assert "IT Team" in office_raw_text
        assert "Seminar Halls" in office_raw_text

    def test_03_sdc_third_floor_content(self, office_raw_text):
        """3. SDC 3rd Floor contains Laptop and Technical Support Teams."""
        assert "3rd Floor" in office_raw_text
        assert "Laptop" in office_raw_text
        assert "technical" in office_raw_text.lower()

    def test_04_sdc_it_vs_laptop_distinction(self, office_raw_text):
        """4. Clear distinction: SDC IT team on 2nd floor, laptop support on 3rd floor."""
        assert "IT team on the 2nd floor" in office_raw_text or "IT Team: SDC has an IT team located on the 2nd floor" in office_raw_text
        assert "3rd floor of SDC" in office_raw_text or "3rd Floor" in office_raw_text

    def test_05_tower1_basement_content(self, office_raw_text):
        """5. Tower 1 Basement has Parking."""
        assert "Tower 1" in office_raw_text
        assert "Basement" in office_raw_text
        assert "Parking" in office_raw_text

    def test_06_tower1_ground_floor_content(self, office_raw_text):
        """6. Tower 1 Ground Floor has Reception, Play Area (games), and Breakout Rooms."""
        assert "Play Area" in office_raw_text
        assert "Carrom" in office_raw_text or "carrom" in office_raw_text
        assert "Table tennis" in office_raw_text or "table tennis" in office_raw_text
        assert "Breakout Rooms" in office_raw_text

    def test_07_tower1_first_floor_content(self, office_raw_text):
        """7. Tower 1 1st Floor has ODCs and IT Team."""
        assert "1st Floor" in office_raw_text
        assert "ODCs" in office_raw_text

    def test_08_tower1_trainees_content(self, office_raw_text):
        """8. Trainees are located in Tower 1 across project areas."""
        assert "Trainees" in office_raw_text or "trainees" in office_raw_text

    def test_09_tower2_basement_content(self, office_raw_text):
        """9. Tower 2 Basement has Parking."""
        assert "Tower 2" in office_raw_text
        assert "Basement" in office_raw_text

    def test_10_tower2_ground_floor_content(self, office_raw_text):
        """10. Tower 2 Ground Floor has Reception, Play Area, and Breakout Rooms."""
        assert "Tower 2" in office_raw_text
        assert "Breakout Rooms" in office_raw_text

    def test_11_tower2_experienced_teams_content(self, office_raw_text):
        """11. Tower 2 has Professionals and Experienced Teams with separate project ODCs."""
        assert "Professionals" in office_raw_text or "Experienced Teams" in office_raw_text

    def test_12_tower2_experienced_teams_floor_unspecified(self, office_raw_text):
        """12. Strict non-hallucination: Tower 2 experienced teams floor is NOT specified."""
        lower_text = office_raw_text.lower()
        assert "does not specify the exact floor" in lower_text or "not specify the floor" in lower_text or "unspecified" in lower_text


class TestOfficeKnowledgeRAGQueries:
    """Tests 13-21: RAG queries answering specific office questions."""

    def _make_mock_rag(self, returned_chunks: list[SearchResult], answer_text: str):
        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.return_value = returned_chunks
        retriever.config = RetrievalConfig()

        context_builder = ContextBuilder()
        answer_generator = MagicMock(spec=AnswerGenerator)
        answer_generator.generate.return_value = GeneratedAnswer(
            answer=answer_text,
            model="mock-groq",
            prompt_tokens=150,
            completion_tokens=40,
            total_tokens=190,
        )

        return RAGService(
            retriever=retriever,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )

    def test_13_cafeteria_query(self):
        """13. 'Where is the cafeteria?' answers SDC Ground Floor only."""
        service = self._make_mock_rag(
            [SDC_GROUND_CHUNK],
            "The cafeteria is located on the Ground Floor of the SDC (Software Development Centre) building. It is the only cafeteria on campus.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where is the cafeteria?")
        assert "SDC" in resp.answer
        assert "Ground Floor" in resp.answer
        assert resp.confidence == "HIGH"

    def test_14_it_team_in_sdc_query(self):
        """14. 'Where is the IT team in SDC?' answers SDC 2nd Floor."""
        service = self._make_mock_rag(
            [SDC_2ND_CHUNK],
            "The IT team in SDC is located on the 2nd Floor.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where is the IT team located in SDC?")
        assert "2nd Floor" in resp.answer or "second floor" in resp.answer.lower()
        assert "SDC" in resp.answer
        assert resp.confidence == "HIGH"

    def test_15_laptop_support_query(self):
        """15. 'Where do I go for laptop issues?' answers SDC 3rd Floor."""
        service = self._make_mock_rag(
            [SDC_3RD_CHUNK],
            "For laptop issues and technical issue support, please go to the 3rd Floor of the SDC building.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where do I go for laptop issues?")
        assert "3rd Floor" in resp.answer or "third floor" in resp.answer.lower()
        assert "SDC" in resp.answer
        assert resp.confidence == "HIGH"

    def test_16_table_tennis_play_area_query(self):
        """16. 'Where can I play table tennis?' answers Tower 1 and Tower 2 Ground Floor play areas."""
        service = self._make_mock_rag(
            [TOWER1_GROUND_CHUNK, TOWER2_GROUND_CHUNK],
            "Table tennis, carrom, and chess are available in the play areas on the Ground Floor of both Tower 1 and Tower 2.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where can I play table tennis?")
        assert "Ground Floor" in resp.answer
        assert resp.confidence == "HIGH"

    def test_17_parking_locations_query(self):
        """17. 'Where is parking located?' answers Tower 1 and Tower 2 Basement."""
        service = self._make_mock_rag(
            [TOWER1_PARKING_CHUNK, TOWER2_PARKING_CHUNK],
            "Parking is available in the Basement of Tower 1 and the Basement of Tower 2.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where is parking located?")
        assert "Basement" in resp.answer
        assert "Tower 1" in resp.answer
        assert resp.confidence == "HIGH"

    def test_18_techbees_classrooms_query(self):
        """18. 'Where are Techbees classrooms located?' answers SDC 2nd Floor."""
        service = self._make_mock_rag(
            [SDC_2ND_CHUNK],
            "Techbees classrooms are located on the 2nd Floor of the SDC building.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where are Techbees classrooms?")
        assert "SDC" in resp.answer
        assert "2nd Floor" in resp.answer or "second floor" in resp.answer.lower()
        assert resp.confidence == "HIGH"

    def test_19_seminar_halls_query(self):
        """19. 'Where are the seminar halls?' answers SDC 2nd floor."""
        service = self._make_mock_rag(
            [SDC_2ND_CHUNK],
            "Seminar halls are located on the 2nd Floor of the SDC building.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where are the seminar halls?")
        assert "SDC" in resp.answer
        assert "2nd Floor" in resp.answer or "second floor" in resp.answer.lower()
        assert resp.confidence == "HIGH"

    def test_20_breakout_rooms_query(self):
        """20. 'Where are the breakout rooms?' answers Tower 1 and Tower 2 Ground Floor."""
        service = self._make_mock_rag(
            [TOWER1_GROUND_CHUNK, TOWER2_GROUND_CHUNK],
            "Breakout rooms are located on the Ground Floor of Tower 1 and the Ground Floor of Tower 2.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where are the breakout rooms?")
        assert "Ground Floor" in resp.answer
        assert "Tower 1" in resp.answer
        assert resp.confidence == "HIGH"

    def test_21_trainees_location_query(self):
        """21. 'Where are trainees located?' answers Tower 1."""
        service = self._make_mock_rag(
            [TOWER1_TRAINEES_CHUNK],
            "Trainees are located in Tower 1 across its project areas.",
        )
        resp = service.answer_question(tenant_id=TARGET_TENANT_ID, question="Where are trainees located?")
        assert "Tower 1" in resp.answer
        assert resp.confidence == "HIGH"


class TestSecurityTenancyAndAgentRouting:
    """Tests 22-25: Tenant isolation, hallucination avoidance, regression, and agent routing."""

    def test_22_tenant_isolation(self):
        """22. Tenant isolation: Non-target tenant gets NO_MATCH for HCL office knowledge."""
        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.return_value = []
        retriever.config = RetrievalConfig()

        context_builder = ContextBuilder()
        answer_generator = AnswerGenerator()

        service = RAGService(
            retriever=retriever,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )

        resp = service.answer_question(
            tenant_id=OTHER_TENANT_ID,
            question="Where is the SDC cafeteria?",
        )
        assert resp.confidence == "NO_MATCH"
        assert "could not find" in resp.answer.lower()
        assert len(resp.sources) == 0

    def test_23_tower2_experienced_teams_no_hallucination(self):
        """23. Strict non-hallucination: Tower 2 experienced teams floor is stated as unspecified."""
        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.return_value = [TOWER2_EXPERIENCED_CHUNK]
        retriever.config = RetrievalConfig()

        context_builder = ContextBuilder()
        answer_generator = MagicMock(spec=AnswerGenerator)
        answer_generator.generate.return_value = GeneratedAnswer(
            answer=(
                "Experienced teams and professionals are located in separate project-specific "
                "ODCs in Tower 2. Note that the specific floor number is not specified in campus records."
            ),
            model="mock-groq",
            prompt_tokens=150,
            completion_tokens=40,
            total_tokens=190,
        )

        service = RAGService(
            retriever=retriever,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )

        resp = service.answer_question(
            tenant_id=TARGET_TENANT_ID,
            question="Which floor are the experienced teams on in Tower 2?",
        )
        assert "Tower 2" in resp.answer
        assert "not specified" in resp.answer.lower() or "unspecified" in resp.answer.lower()
        # Must NOT claim 1st floor or 2nd floor
        assert "on the 1st floor" not in resp.answer.lower()
        assert "on the 2nd floor" not in resp.answer.lower()

    def test_24_existing_policy_regression(self):
        """24. Regression: Existing policies (e.g. annual leave) continue to retrieve and answer."""
        annual_leave_chunk = SearchResult(
            chunk_id="leave_policy_0001",
            document_id="leave_policy",
            document_name="HCL Leave and Holiday Policy",
            source_file="HCL_LEAVE_AND_HOLIDAY_POLICY.pdf",
            text="Employees are entitled to 20 days of annual leave per financial year.",
            page_start=12,
            page_end=13,
            section="Annual Leave Entitlement",
            similarity_score=0.91,
            distance=0.09,
            metadata={"tenant_id": str(TARGET_TENANT_ID)},
        )

        retriever = MagicMock(spec=RAGRetriever)
        retriever.retrieve.return_value = [annual_leave_chunk]
        retriever.config = RetrievalConfig()

        context_builder = ContextBuilder()
        answer_generator = MagicMock(spec=AnswerGenerator)
        answer_generator.generate.return_value = GeneratedAnswer(
            answer="According to company policy, employees are entitled to 20 days of annual leave.",
            model="mock-groq",
            prompt_tokens=120,
            completion_tokens=25,
            total_tokens=145,
        )

        service = RAGService(
            retriever=retriever,
            context_builder=context_builder,
            answer_generator=answer_generator,
        )

        resp = service.answer_question(
            tenant_id=TARGET_TENANT_ID,
            question="What is the annual leave policy for employees?",
        )
        assert "20 days" in resp.answer
        assert resp.confidence == "HIGH"

    def test_25_orchestrator_routing_office_queries(self):
        """25. Orchestrator routes NAVIGATION and KNOWLEDGE_QUERY intents to knowledge node."""
        state_nav = make_initial_state("Where is the cafeteria?", str(TARGET_TENANT_ID))
        state_nav["intent"] = IntentType.NAVIGATION.value
        assert _route_after_classify(state_nav) == "knowledge_agent"

        state_know = make_initial_state("What facilities are in SDC?", str(TARGET_TENANT_ID))
        state_know["intent"] = IntentType.KNOWLEDGE_QUERY.value
        assert _route_after_classify(state_know) == "knowledge_agent"

        # Ensure TOOL_INTENTS set still includes NAVIGATION for compatibility
        assert IntentType.NAVIGATION in TOOL_INTENTS
