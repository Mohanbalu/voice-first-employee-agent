"""Tests for Module 2: Knowledge Base Ingestion and Chunking.

Covers:
1. Document ID and title generation
2. Text cleaning (whitespace, line endings, numbers/dates/lists preservation)
3. Document chunking (section detection, paragraph-awareness, sentence-awareness, overlap)
4. Metadata generation (schema validation, future tenancy hooks)
5. Empty and corrupt text handling
6. Multiple-document processing and summary tracking
"""

from pathlib import Path
import tempfile
import json
import pytest

from backend.app.rag.chunking import (
    ChunkMetadata,
    DocumentChunk,
    detect_section_heading,
    split_into_sentences,
    chunk_document,
)
from backend.app.rag.ingestion import (
    generate_document_id,
    generate_document_title,
    clean_text,
    clean_pages,
    IngestionPaths,
    process_document,
    process_all_documents,
)


class TestDocumentIdentifier:
    """Tests for document ID and title slugification."""

    def test_generate_document_id_simple(self):
        assert generate_document_id("Travel Policy.pdf") == "travel_policy"
        assert generate_document_id("Travel-Policy.pdf") == "travel_policy"

    def test_generate_document_id_complex(self):
        filename = "Anti Bribery-and-Anti-Corruption-Policy-Global_28-04-2026.pdf"
        expected = "anti_bribery_and_anti_corruption_policy_global_28_04_2026"
        assert generate_document_id(filename) == expected

    def test_generate_document_id_with_path(self):
        path = Path("data/policies/HCL LEAVE AND HOLIDAY POLICY.pdf")
        assert generate_document_id(path) == "hcl_leave_and_holiday_policy"

    def test_generate_document_title(self):
        assert generate_document_title("Travel-Policy.pdf") == "Travel Policy"
        assert generate_document_title("hcl wfh policy.pdf") == "Hcl Wfh Policy"


class TestTextCleaning:
    """Tests for conservative policy text cleaning."""

    def test_normalize_whitespace_and_newlines(self):
        raw = "Line 1   with   extra    spaces.\r\n\r\n\r\n\r\nLine 2 after blanks."
        cleaned = clean_text(raw)
        assert "   " not in cleaned
        assert "\r" not in cleaned
        assert cleaned == "Line 1 with extra spaces.\n\nLine 2 after blanks."

    def test_preserves_policy_numbers_dates_and_amounts(self):
        raw = (
            "1.1 Eligibility: Employees with > 6 months service qualify for $5,000 allowance "
            "effective from 01/04/2026 up to 100% reimbursement."
        )
        cleaned = clean_text(raw)
        assert "1.1 Eligibility:" in cleaned
        assert "> 6 months" in cleaned
        assert "$5,000" in cleaned
        assert "01/04/2026" in cleaned
        assert "100%" in cleaned

    def test_preserves_bullet_points_and_lists(self):
        raw = """Key Obligations:
• Maintain confidentiality
• Report conflicts of interest
* Comply with safety protocols
- Avoid unapproved gifts"""
        cleaned = clean_text(raw)
        assert "• Maintain confidentiality" in cleaned
        assert "* Comply with safety protocols" in cleaned
        assert "- Avoid unapproved gifts" in cleaned

    def test_removes_non_printable_artifacts(self):
        raw = "Header\x00 text with\x0c page artifact\x07."
        cleaned = clean_text(raw)
        assert "\x00" not in cleaned
        assert "\x0c" not in cleaned
        assert "\x07" not in cleaned
        assert cleaned == "Header text with page artifact."

    def test_clean_empty_text(self):
        assert clean_text("") == ""
        assert clean_text("   \n\n\t  ") == ""


class TestSectionDetection:
    """Tests for section and heading detection in policy text."""

    def test_detect_numbered_sections(self):
        assert detect_section_heading("1. Objective") == "1. Objective"
        assert detect_section_heading("2.1 Scope and Applicability") == "2.1 Scope and Applicability"
        assert detect_section_heading("Section 4: Disciplinary Actions") == "Section 4: Disciplinary Actions"

    def test_detect_all_caps_headings(self):
        assert detect_section_heading("BUSINESS TRAVEL POLICY") == "Business Travel Policy"
        assert detect_section_heading("ROLES AND RESPONSIBILITIES") == "Roles And Responsibilities"

    def test_reject_regular_sentences(self):
        assert detect_section_heading("This policy applies to all global employees.") is None
        assert detect_section_heading("Employees must submit expense claims within 30 days.") is None


class TestChunking:
    """Tests for document chunking and metadata generation."""

    def test_split_into_sentences(self):
        text = "This is sentence one. This is sentence two! Is this sentence three? Yes, it is."
        sentences = split_into_sentences(text)
        assert len(sentences) == 4
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two!"
        assert sentences[2] == "Is this sentence three?"
        assert sentences[3] == "Yes, it is."

    def test_chunk_document_basic(self):
        pages = [
            (1, "1. Objective\nThis policy ensures compliant operations.\n\n2. Scope\nApplies to all full-time personnel."),
            (2, "3. Entitlements\nEmployees receive standard per diem rates."),
        ]
        chunks = chunk_document(
            pages=pages,
            document_id="test_doc",
            document_name="Test Document",
            source_file="test_doc.pdf",
            chunk_size=500,
            chunk_overlap=50,
        )

        assert len(chunks) >= 1
        for idx, chunk in enumerate(chunks):
            assert chunk.metadata.document_id == "test_doc"
            assert chunk.metadata.document_name == "Test Document"
            assert chunk.metadata.source_file == "test_doc.pdf"
            assert chunk.metadata.chunk_id == f"test_doc_{idx + 1:04d}"
            assert chunk.metadata.chunk_index == idx
            assert chunk.metadata.page_start in (1, 2)
            assert chunk.metadata.page_end in (1, 2)
            assert len(chunk.text) > 0

    def test_chunk_metadata_serialization(self):
        meta = ChunkMetadata(
            document_id="policy_1",
            document_name="Policy One",
            source_file="policy_1.pdf",
            chunk_id="policy_1_0001",
            chunk_index=0,
            page_start=1,
            page_end=1,
            section="1. Overview",
            tenant_id="tenant_abc",
        )
        chunk = DocumentChunk(text="Sample text content.", metadata=meta)
        serialized = chunk.to_dict()

        assert serialized["document_id"] == "policy_1"
        assert serialized["document_name"] == "Policy One"
        assert serialized["source_file"] == "policy_1.pdf"
        assert serialized["chunk_id"] == "policy_1_0001"
        assert serialized["chunk_index"] == 0
        assert serialized["page_start"] == 1
        assert serialized["page_end"] == 1
        assert serialized["section"] == "1. Overview"
        assert serialized["tenant_id"] == "tenant_abc"
        assert serialized["text"] == "Sample text content."

    def test_chunk_empty_pages(self):
        pages = [(1, ""), (2, "   \n\n  ")]
        chunks = chunk_document(
            pages=pages,
            document_id="empty_doc",
            document_name="Empty Document",
            source_file="empty.pdf",
        )
        assert chunks == []


    def test_chunk_spans_multiple_pages(self):
        pages = [
            (1, "Page 1 intro paragraph describing travel."),
            (2, "Page 2 detailed policy provisions continuing the discussion."),
        ]
        chunks = chunk_document(
            pages=pages,
            document_id="multi_page_doc",
            document_name="Multi Page Doc",
            source_file="multi.pdf",
            chunk_size=1000,
            chunk_overlap=50,
        )
        assert len(chunks) == 1
        assert chunks[0].metadata.page_start == 1
        assert chunks[0].metadata.page_end == 2


class TestPipelineExecution:
    """Tests end-to-end processing with temporary directories."""

    def test_process_all_documents_empty_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()
            output_dir = temp_path / "output"

            summary = process_all_documents(
                policies_dir=policies_dir,
                output_base=output_dir,
            )

            assert summary.documents_found == 0
            assert summary.documents_processed == 0
            assert summary.documents_failed == 0
            assert summary.total_chunks == 0

    def test_process_corrupt_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            corrupt_pdf = temp_path / "corrupt.pdf"
            corrupt_pdf.write_bytes(b"Not a valid PDF header content")

            paths = IngestionPaths(
                policies_dir=temp_path,
                extracted_dir=temp_path / "extracted",
                cleaned_dir=temp_path / "cleaned",
                chunks_dir=temp_path / "chunks",
            )
            paths.ensure_directories()

            result = process_document(corrupt_pdf, paths)
            assert result.success is False
            assert result.error_message is not None

    def test_process_missing_pdf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            missing_pdf = temp_path / "non_existent.pdf"
            paths = IngestionPaths(
                policies_dir=temp_path,
                extracted_dir=temp_path / "extracted",
                cleaned_dir=temp_path / "cleaned",
                chunks_dir=temp_path / "chunks",
            )
            paths.ensure_directories()

            result = process_document(missing_pdf, paths)
            assert result.success is False
            assert "does not exist" in result.error_message

