"""Unit tests for Knowledge Chunk Quality Validation (Module 3.1).

Covers:
1. Required metadata field validation
2. Text sizing and categorization (EMPTY, VERY_SHORT, SHORT, NORMAL, LARGE, VERY_LARGE)
3. Exact duplicate detection
4. Near-duplicate Jaccard similarity detection
5. Page range and sequence validity
6. Missing source file detection
7. Chunk ID global uniqueness
8. Suspicious section detection (navigation, menus, TOC)
9. End-to-end report generation (JSON and TXT serialization)
"""

import json
import tempfile
from pathlib import Path
import pytest

from backend.app.rag.validate_chunks import (
    REQUIRED_FIELDS,
    analyze_text_metrics,
    is_suspicious_section,
    compute_word_jaccard,
    validate_all_chunks,
    format_text_report,
    run_validation,
)


class TestTextMetricsAndSizing:
    """Tests for character, word, sentence, and size categorization."""

    def test_empty_chunk_metrics(self):
        metrics = analyze_text_metrics("")
        assert metrics.size_category == "EMPTY"
        assert metrics.char_count == 0
        assert metrics.word_count == 0

    def test_very_short_chunk(self):
        metrics = analyze_text_metrics("Short note.")
        assert metrics.size_category == "VERY_SHORT"
        assert metrics.char_count < 50
        assert metrics.word_count < 10

    def test_short_chunk(self):
        text = "This is a short policy statement covering basic employee guidelines for workplace attendance."
        metrics = analyze_text_metrics(text)
        assert metrics.size_category == "SHORT"
        assert 50 <= metrics.char_count < 200

    def test_normal_chunk(self):
        text = " ".join(["Policy paragraph discussing operational procedure."] * 15)
        metrics = analyze_text_metrics(text)
        assert metrics.size_category == "NORMAL"
        assert 200 <= metrics.char_count <= 1500

    def test_large_chunk(self):
        text = " ".join(["Policy paragraph discussing operational procedure."] * 35)
        metrics = analyze_text_metrics(text)
        assert metrics.size_category == "LARGE"
        assert 1500 < metrics.char_count <= 2500

    def test_very_large_chunk(self):
        text = " ".join(["Policy paragraph discussing operational procedure."] * 60)
        metrics = analyze_text_metrics(text)
        assert metrics.size_category == "VERY_LARGE"
        assert metrics.char_count > 2500

    def test_artifact_detection(self):
        text_with_replacement = "Damaged character \ufffd in text."
        metrics = analyze_text_metrics(text_with_replacement)
        assert metrics.has_artifacts is True
        assert any("\\ufffd" in d for d in metrics.artifact_details)

        text_with_dots = "Chapter 1 ............................ 5"
        metrics_dots = analyze_text_metrics(text_with_dots)
        assert metrics_dots.has_artifacts is True
        assert any("dot sequences" in d for d in metrics_dots.artifact_details)


class TestSuspiciousSectionDetection:
    """Tests for identifying web portal, menu, and TOC section headers."""

    def test_detect_portal_navigation(self):
        is_susp, reason = is_suspicious_section("Policy Process FAQ(s) Form(s) History")
        assert is_susp is True
        assert reason is not None

        is_susp, reason = is_suspicious_section("Home | Change Role | Quick Links | Logout")
        assert is_susp is True

    def test_detect_table_of_contents(self):
        is_susp, reason = is_suspicious_section("Table of Contents")
        assert is_susp is True

    def test_allow_valid_policy_sections(self):
        is_susp, reason = is_suspicious_section("1. Objective and Purpose")
        assert is_susp is False
        assert reason is None

        is_susp, reason = is_suspicious_section("Section 4: Disciplinary Sanctions")
        assert is_susp is False

        is_susp, reason = is_suspicious_section("Business Travel Policy")
        assert is_susp is False

    def test_detect_overly_long_section(self):
        long_sec = "A" * 105
        is_susp, reason = is_suspicious_section(long_sec)
        assert is_susp is True
        assert "exceeds 100 characters" in reason


class TestDuplicateDetection:
    """Tests for exact and near-duplicate similarity."""

    def test_jaccard_identical_text(self):
        t1 = "All full-time employees must submit travel receipts within 30 calendar days."
        t2 = "All full-time employees must submit travel receipts within 30 calendar days."
        assert compute_word_jaccard(t1, t2) == 1.0

    def test_jaccard_near_identical_text(self):
        t1 = "All full-time employees must submit travel receipts within thirty calendar days for review."
        t2 = "All full-time employees must submit travel receipts within 30 calendar days for review."
        similarity = compute_word_jaccard(t1, t2)
        assert similarity >= 0.80

    def test_jaccard_dissimilar_text(self):
        t1 = "Information security policy guidelines and password complexity."
        t2 = "Maternity leave entitlements and health insurance claims process."
        assert compute_word_jaccard(t1, t2) < 0.2


class TestChunkValidationEngine:
    """Tests end-to-end audit engine with synthetic and sample JSONL files."""

    def test_detects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_dir = temp_path / "chunks"
            chunks_dir.mkdir()
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()

            # Create mock source PDF
            (policies_dir / "sample.pdf").write_bytes(b"%PDF-1.4 Mock")

            # Missing 'text' and 'chunk_id'
            chunk_line = {
                "document_id": "sample",
                "document_name": "Sample",
                "source_file": "sample.pdf",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
            }
            (chunks_dir / "sample.jsonl").write_text(
                json.dumps(chunk_line) + "\n", encoding="utf-8"
            )

            report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)
            assert report.missing_metadata_fields > 0
            assert any("missing required fields" in issue for issue in report.critical_issues)
            assert report.overall_status == "NEEDS_REVIEW"

    def test_detects_invalid_page_range(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_dir = temp_path / "chunks"
            chunks_dir.mkdir()
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()
            (policies_dir / "sample.pdf").write_bytes(b"%PDF-1.4 Mock")

            # page_start (5) > page_end (2)
            chunk_line = {
                "document_id": "sample",
                "document_name": "Sample",
                "source_file": "sample.pdf",
                "chunk_id": "sample_0001",
                "chunk_index": 0,
                "page_start": 5,
                "page_end": 2,
                "text": "Valid body text describing company standards and expectations.",
            }
            (chunks_dir / "sample.jsonl").write_text(
                json.dumps(chunk_line) + "\n", encoding="utf-8"
            )

            report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)
            assert report.invalid_page_ranges == 1
            assert any("Invalid page range" in issue for issue in report.critical_issues)

    def test_detects_duplicate_chunk_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_dir = temp_path / "chunks"
            chunks_dir.mkdir()
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()
            (policies_dir / "sample.pdf").write_bytes(b"%PDF-1.4 Mock")

            lines = [
                {
                    "document_id": "sample",
                    "document_name": "Sample",
                    "source_file": "sample.pdf",
                    "chunk_id": "duplicate_id",
                    "chunk_index": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "First passage content here.",
                },
                {
                    "document_id": "sample",
                    "document_name": "Sample",
                    "source_file": "sample.pdf",
                    "chunk_id": "duplicate_id",
                    "chunk_index": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Second passage content with identical chunk_id.",
                },
            ]
            content = "\n".join(json.dumps(l) for l in lines)
            (chunks_dir / "sample.jsonl").write_text(content, encoding="utf-8")

            report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)
            assert any("Duplicate chunk_id" in issue for issue in report.critical_issues)

    def test_detects_missing_source_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_dir = temp_path / "chunks"
            chunks_dir.mkdir()
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()
            # Do NOT create nonexistent.pdf in policies_dir

            chunk_line = {
                "document_id": "sample",
                "document_name": "Sample",
                "source_file": "nonexistent.pdf",
                "chunk_id": "sample_0001",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "text": "Passage text.",
            }
            (chunks_dir / "sample.jsonl").write_text(
                json.dumps(chunk_line) + "\n", encoding="utf-8"
            )

            report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)
            assert report.missing_source_files == 1
            assert any("does not exist" in issue for issue in report.critical_issues)

    def test_report_formatting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_dir = temp_path / "chunks"
            chunks_dir.mkdir()
            policies_dir = temp_path / "policies"
            policies_dir.mkdir()
            (policies_dir / "sample.pdf").write_bytes(b"%PDF-1.4 Mock")

            chunk_line = {
                "document_id": "sample",
                "document_name": "Sample Policy",
                "source_file": "sample.pdf",
                "chunk_id": "sample_0001",
                "chunk_index": 0,
                "page_start": 1,
                "page_end": 1,
                "section": "1. Purpose",
                "text": "This is a clean, compliant policy chunk meeting all length guidelines.",
            }
            (chunks_dir / "sample.jsonl").write_text(
                json.dumps(chunk_line) + "\n", encoding="utf-8"
            )

            report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)
            formatted = format_text_report(report)
            assert "Knowledge Chunk Quality Report" in formatted
            assert "Documents analyzed:     1" in formatted
            assert "Total chunks:           1" in formatted
            assert "Valid chunks:           1" in formatted
            assert report.overall_status == "READY_FOR_EMBEDDINGS"
