"""Knowledge Chunk Quality Validation Module.

Audits processed JSONL chunks for metadata integrity, text size distribution,
exact and near duplicates, suspicious section headers (such as portal navigation),
page range validity, and potential extraction artifacts before RAG embedding.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("knowledge_base.validate_chunks")

REQUIRED_FIELDS = [
    "document_id",
    "document_name",
    "source_file",
    "chunk_id",
    "chunk_index",
    "page_start",
    "page_end",
    "text",
]

# Patterns characteristic of portal menus, web headers, or non-policy navigation
SUSPICIOUS_SECTION_PATTERNS = [
    re.compile(r"\bFAQ\(s\)\b", re.IGNORECASE),
    re.compile(r"\bForm\(s\)\b", re.IGNORECASE),
    re.compile(r"\bHistory\b", re.IGNORECASE),
    re.compile(r"\bQuick Links\b", re.IGNORECASE),
    re.compile(r"\bChange Role\b", re.IGNORECASE),
    re.compile(r"\bLogout\b", re.IGNORECASE),
    re.compile(r"\bHome\s*\|", re.IGNORECASE),
    re.compile(r"\bEnter Keyword\b", re.IGNORECASE),
    re.compile(r"\bTable of Contents\b", re.IGNORECASE),
    re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"\|.*\|", re.IGNORECASE),  # Pipe-delimited navigation bar
]


def get_project_root() -> Path:
    """Locates the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "backend").exists():
            return parent
        if (parent / "data" / "processed" / "chunks").exists():
            return parent
    return Path.cwd()


@dataclass
class ChunkMetrics:
    char_count: int
    word_count: int
    sentence_count: int
    line_count: int
    repeated_line_count: int
    whitespace_ratio: float
    size_category: str  # EMPTY, VERY_SHORT, SHORT, NORMAL, LARGE, VERY_LARGE
    has_artifacts: bool
    artifact_details: List[str] = field(default_factory=list)


def analyze_text_metrics(text: str) -> ChunkMetrics:
    """Computes sizing, linguistic, and artifact metrics for a single chunk."""
    if not text or not text.strip():
        return ChunkMetrics(
            char_count=0,
            word_count=0,
            sentence_count=0,
            line_count=0,
            repeated_line_count=0,
            whitespace_ratio=0.0,
            size_category="EMPTY",
            has_artifacts=False,
        )

    char_count = len(text)
    words = text.split()
    word_count = len(words)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line_count = len(lines)

    # Estimate sentence count
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    sentence_count = len(sentences)

    # Count repeated lines within chunk
    line_freq: Dict[str, int] = defaultdict(int)
    for l in lines:
        line_freq[l] += 1
    repeated_line_count = sum(c - 1 for c in line_freq.values() if c > 1)

    # Whitespace ratio
    whitespace_chars = sum(1 for c in text if c.isspace())
    whitespace_ratio = round(whitespace_chars / max(char_count, 1), 3)

    # Categorize size
    if char_count < 50 or word_count < 10:
        size_category = "VERY_SHORT"
    elif char_count < 200:
        size_category = "SHORT"
    elif char_count <= 1500:
        size_category = "NORMAL"
    elif char_count <= 2500:
        size_category = "LARGE"
    else:
        size_category = "VERY_LARGE"

    # Artifact detection
    artifacts: List[str] = []
    if "\ufffd" in text:
        artifacts.append("Contains unicode replacement character (\\ufffd)")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        artifacts.append("Contains non-printable control characters")

    # Excessive single character lines (fragmented OCR/extraction)
    single_char_lines = sum(1 for l in lines if len(l) == 1)
    if line_count > 5 and (single_char_lines / line_count) > 0.3:
        artifacts.append(f"High single-character line ratio ({single_char_lines}/{line_count})")

    # Suspicious excessive dot runs (e.g., table-of-contents dot leaders)
    if re.search(r"\.{5,}", text):
        artifacts.append("Contains long dot sequences (possible table of contents)")

    return ChunkMetrics(
        char_count=char_count,
        word_count=word_count,
        sentence_count=sentence_count,
        line_count=line_count,
        repeated_line_count=repeated_line_count,
        whitespace_ratio=whitespace_ratio,
        size_category=size_category,
        has_artifacts=len(artifacts) > 0,
        artifact_details=artifacts,
    )


def is_suspicious_section(section: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Checks whether a section name appears to be navigation, footer, or TOC text."""
    if not section:
        return False, None

    clean_sec = section.strip()
    if len(clean_sec) > 100:
        return True, f"Section name exceeds 100 characters ({len(clean_sec)} chars)"

    for pattern in SUSPICIOUS_SECTION_PATTERNS:
        if pattern.search(clean_sec):
            return True, f"Matches navigation/menu pattern: '{pattern.pattern}'"

    # Check for pipe delimiters (e.g., "Home | Change Role | Quick Links")
    if clean_sec.count("|") >= 2:
        return True, "Contains multiple pipe separators (menu bar artifact)"

    return False, None


def compute_word_jaccard(text_a: str, text_b: str) -> float:
    """Computes Jaccard similarity of word tokens between two chunks."""
    tokens_a = set(w.lower() for w in re.findall(r"\b[a-z0-9]{3,}\b", text_a))
    tokens_b = set(w.lower() for w in re.findall(r"\b[a-z0-9]{3,}\b", text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return round(intersection / union, 4) if union > 0 else 0.0


@dataclass
class DocumentValidationSummary:
    document_id: str
    document_name: str
    source_file: str
    source_file_exists: bool
    total_chunks: int
    min_page: int
    max_page: int
    pages_represented: int
    empty_chunks: int = 0
    very_short_chunks: int = 0
    short_chunks: int = 0
    normal_chunks: int = 0
    large_chunks: int = 0
    very_large_chunks: int = 0
    missing_metadata_count: int = 0
    invalid_page_ranges: int = 0
    index_sequence_errors: int = 0
    suspicious_sections: List[Dict[str, Any]] = field(default_factory=list)
    artifacts_found: List[Dict[str, Any]] = field(default_factory=list)
    exact_duplicates_internal: int = 0
    quality_status: str = "READY"  # READY or REVIEW


@dataclass
class GlobalValidationReport:
    total_documents: int
    total_chunks: int
    valid_chunks: int
    empty_chunks: int
    very_short_chunks: int
    short_chunks: int
    normal_chunks: int
    large_chunks: int
    very_large_chunks: int
    exact_duplicate_chunks: int
    exact_duplicate_groups: int
    near_duplicate_pairs: int
    missing_metadata_fields: int
    invalid_page_ranges: int
    invalid_chunk_indexes: int
    missing_source_files: int
    suspicious_section_chunks: int
    suspected_artifact_chunks: int
    overall_status: str  # READY_FOR_EMBEDDINGS or NEEDS_REVIEW
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    informational_findings: List[str] = field(default_factory=list)
    document_summaries: Dict[str, Any] = field(default_factory=dict)
    duplicate_groups_detail: List[Dict[str, Any]] = field(default_factory=list)
    near_duplicates_detail: List[Dict[str, Any]] = field(default_factory=list)
    suspicious_sections_detail: List[Dict[str, Any]] = field(default_factory=list)


def validate_all_chunks(
    chunks_dir: Optional[Path] = None,
    policies_dir: Optional[Path] = None,
    near_dup_threshold: float = 0.88,
) -> GlobalValidationReport:
    """Performs full quality validation across all JSONL chunk files.

    Args:
        chunks_dir: Directory containing .jsonl chunk files.
        policies_dir: Directory containing original PDF source files.
        near_dup_threshold: Jaccard threshold for flagging near-duplicates.

    Returns:
        GlobalValidationReport with complete diagnostics.
    """
    root = get_project_root()
    c_dir = chunks_dir or (root / "data" / "processed" / "chunks")
    p_dir = policies_dir or (root / "data" / "policies")

    jsonl_files = sorted(c_dir.glob("*.jsonl"))
    logger.info("Found %d chunk file(s) in %s", len(jsonl_files), c_dir)

    all_chunk_ids: Set[str] = set()
    duplicate_chunk_ids: Set[str] = set()
    text_hashes: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # hash -> [(chunk_id, doc_id)]
    all_chunks_for_neardup: List[Dict[str, Any]] = []

    doc_summaries: Dict[str, DocumentValidationSummary] = {}

    total_chunks = 0
    empty_chunks = 0
    very_short_chunks = 0
    short_chunks = 0
    normal_chunks = 0
    large_chunks = 0
    very_large_chunks = 0
    missing_metadata_fields = 0
    invalid_page_ranges = 0
    invalid_chunk_indexes = 0
    missing_source_files = 0
    suspicious_section_chunks = 0
    suspected_artifact_chunks = 0

    critical_issues: List[str] = []
    warnings: List[str] = []
    informational: List[str] = []
    suspicious_sections_detail: List[Dict[str, Any]] = []

    for file_path in jsonl_files:
        doc_id = file_path.stem
        lines = file_path.read_text(encoding="utf-8").splitlines()

        doc_chunks: List[Dict[str, Any]] = []
        for line_num, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                doc_chunks.append(data)
            except json.JSONDecodeError as exc:
                critical_issues.append(
                    f"Corrupt JSON in {file_path.name} line {line_num + 1}: {exc}"
                )

        if not doc_chunks:
            continue

        first_chunk = doc_chunks[0]
        doc_name = first_chunk.get("document_name", doc_id)
        source_file = first_chunk.get("source_file", "")
        source_exists = (p_dir / source_file).exists() if source_file else False
        if not source_exists:
            missing_source_files += 1
            critical_issues.append(
                f"Document '{doc_id}' references source file '{source_file}' which does not exist in {p_dir}"
            )

        doc_summary = DocumentValidationSummary(
            document_id=doc_id,
            document_name=doc_name,
            source_file=source_file,
            source_file_exists=source_exists,
            total_chunks=len(doc_chunks),
            min_page=sys.maxsize,
            max_page=0,
            pages_represented=0,
        )

        distinct_pages: Set[int] = set()
        expected_index = 0

        for chunk_data in doc_chunks:
            total_chunks += 1
            cid = chunk_data.get("chunk_id", "")
            cidx = chunk_data.get("chunk_index", -1)
            p_start = chunk_data.get("page_start", 0)
            p_end = chunk_data.get("page_end", 0)
            text = chunk_data.get("text", "")
            section = chunk_data.get("section")

            # Check 1: Required metadata fields
            missing = [f for f in REQUIRED_FIELDS if f not in chunk_data]
            if missing:
                missing_metadata_fields += len(missing)
                doc_summary.missing_metadata_count += len(missing)
                critical_issues.append(
                    f"Chunk '{cid or 'unknown'}' in {doc_id} missing required fields: {missing}"
                )

            # Check 2: Chunk ID uniqueness
            if cid in all_chunk_ids:
                duplicate_chunk_ids.add(cid)
                critical_issues.append(f"Duplicate chunk_id detected: '{cid}'")
            else:
                all_chunk_ids.add(cid)

            # Check 3: Chunk index continuity
            if cidx != expected_index:
                invalid_chunk_indexes += 1
                doc_summary.index_sequence_errors += 1
                critical_issues.append(
                    f"Index sequence break in {doc_id}: expected {expected_index}, found {cidx} (chunk_id={cid})"
                )
            expected_index += 1

            # Check 4: Page ranges
            if p_start > p_end or p_start <= 0 or p_end <= 0:
                invalid_page_ranges += 1
                doc_summary.invalid_page_ranges += 1
                critical_issues.append(
                    f"Invalid page range in {cid}: page_start={p_start}, page_end={p_end}"
                )
            else:
                doc_summary.min_page = min(doc_summary.min_page, p_start)
                doc_summary.max_page = max(doc_summary.max_page, p_end)
                for p in range(p_start, p_end + 1):
                    distinct_pages.add(p)

            # Check 5: Text metrics & sizing
            metrics = analyze_text_metrics(text)
            if metrics.size_category == "EMPTY":
                empty_chunks += 1
                doc_summary.empty_chunks += 1
                critical_issues.append(f"Empty text in chunk '{cid}' ({doc_id})")
            elif metrics.size_category == "VERY_SHORT":
                very_short_chunks += 1
                doc_summary.very_short_chunks += 1
                warnings.append(
                    f"Very short chunk in {cid} ({doc_id}): {metrics.char_count} chars, {metrics.word_count} words"
                )
            elif metrics.size_category == "SHORT":
                short_chunks += 1
                doc_summary.short_chunks += 1
            elif metrics.size_category == "NORMAL":
                normal_chunks += 1
                doc_summary.normal_chunks += 1
            elif metrics.size_category == "LARGE":
                large_chunks += 1
                doc_summary.large_chunks += 1
            elif metrics.size_category == "VERY_LARGE":
                very_large_chunks += 1
                doc_summary.very_large_chunks += 1
                warnings.append(
                    f"Very large chunk in {cid} ({doc_id}): {metrics.char_count} chars"
                )

            # Check 6: Extraction artifacts
            if metrics.has_artifacts:
                suspected_artifact_chunks += 1
                doc_summary.artifacts_found.append({
                    "chunk_id": cid,
                    "details": metrics.artifact_details,
                })
                warnings.append(
                    f"Artifact detected in {cid}: {'; '.join(metrics.artifact_details)}"
                )

            # Check 7: Suspicious section names
            is_susp, reason = is_suspicious_section(section)
            if is_susp:
                suspicious_section_chunks += 1
                item = {
                    "document_id": doc_id,
                    "chunk_id": cid,
                    "section": section,
                    "reason": reason,
                }
                doc_summary.suspicious_sections.append(item)
                suspicious_sections_detail.append(item)
                warnings.append(
                    f"Suspicious section in {cid} ('{section}'): {reason}"
                )

            # Exact duplicate hashing (normalized text)
            norm_text = " ".join(text.lower().split())
            if norm_text:
                thash = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
                text_hashes[thash].append((cid, doc_id))

            # Store for near-duplicate analysis
            if len(text.split()) >= 12:
                all_chunks_for_neardup.append({
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "text": text,
                    "page_start": p_start,
                })

        doc_summary.pages_represented = len(distinct_pages)
        if doc_summary.min_page == sys.maxsize:
            doc_summary.min_page = 0

        # Assign document status
        if (
            doc_summary.empty_chunks > 0
            or doc_summary.missing_metadata_count > 0
            or doc_summary.invalid_page_ranges > 0
            or doc_summary.index_sequence_errors > 0
            or not doc_summary.source_file_exists
            or doc_summary.suspicious_sections
            or doc_summary.very_short_chunks > 0
        ):
            doc_summary.quality_status = "REVIEW"
        else:
            doc_summary.quality_status = "READY"

        doc_summaries[doc_id] = doc_summary

    # Exact duplicate analysis
    exact_duplicate_chunks = 0
    duplicate_groups_detail: List[Dict[str, Any]] = []
    for thash, occurrences in text_hashes.items():
        if len(occurrences) > 1:
            exact_duplicate_chunks += (len(occurrences) - 1)
            group_cids = [occ[0] for occ in occurrences]
            group_docs = list(set(occ[1] for occ in occurrences))
            duplicate_groups_detail.append({
                "duplicate_count": len(occurrences),
                "chunk_ids": group_cids,
                "documents": group_docs,
            })
            warnings.append(
                f"Exact duplicate text across {len(occurrences)} chunks: {group_cids} in docs {group_docs}"
            )

    # Near duplicate analysis (word Jaccard >= threshold)
    # Consecutive chunks naturally overlap by ~15%; check for > near_dup_threshold within doc or cross-doc
    near_duplicates_detail: List[Dict[str, Any]] = []
    num_eval = len(all_chunks_for_neardup)
    for i in range(num_eval):
        c1 = all_chunks_for_neardup[i]
        # Compare with subsequent chunks in same document
        for j in range(i + 1, min(i + 8, num_eval)):
            c2 = all_chunks_for_neardup[j]
            if c1["document_id"] != c2["document_id"]:
                continue
            # Skip adjacent chunk overlap check if overlap is normal
            if abs(int(c1["chunk_id"].split("_")[-1]) - int(c2["chunk_id"].split("_")[-1])) == 1:
                # Adjacent chunk naturally overlaps; only flag if > 0.90 (virtually identical)
                jacc = compute_word_jaccard(c1["text"], c2["text"])
                if jacc >= 0.92:
                    near_duplicates_detail.append({
                        "chunk_id_1": c1["chunk_id"],
                        "chunk_id_2": c2["chunk_id"],
                        "document_id": c1["document_id"],
                        "similarity": jacc,
                        "note": "Adjacent chunks with unusually high similarity (>92%)",
                    })
            else:
                jacc = compute_word_jaccard(c1["text"], c2["text"])
                if jacc >= near_dup_threshold:
                    near_duplicates_detail.append({
                        "chunk_id_1": c1["chunk_id"],
                        "chunk_id_2": c2["chunk_id"],
                        "document_id": c1["document_id"],
                        "similarity": jacc,
                        "note": f"Non-adjacent chunks with {int(jacc*100)}% word overlap",
                    })

    if near_duplicates_detail:
        informational.append(
            f"Detected {len(near_duplicates_detail)} near-duplicate chunk pairs (Jaccard similarity >= {near_dup_threshold})"
        )

    # Calculate valid chunks count
    valid_chunks = total_chunks - empty_chunks - invalid_page_ranges

    # Determine overall status
    if critical_issues:
        overall_status = "NEEDS_REVIEW"
    elif warnings:
        overall_status = "NEEDS_REVIEW"
    else:
        overall_status = "READY_FOR_EMBEDDINGS"

    return GlobalValidationReport(
        total_documents=len(doc_summaries),
        total_chunks=total_chunks,
        valid_chunks=valid_chunks,
        empty_chunks=empty_chunks,
        very_short_chunks=very_short_chunks,
        short_chunks=short_chunks,
        normal_chunks=normal_chunks,
        large_chunks=large_chunks,
        very_large_chunks=very_large_chunks,
        exact_duplicate_chunks=exact_duplicate_chunks,
        exact_duplicate_groups=len(duplicate_groups_detail),
        near_duplicate_pairs=len(near_duplicates_detail),
        missing_metadata_fields=missing_metadata_fields,
        invalid_page_ranges=invalid_page_ranges,
        invalid_chunk_indexes=invalid_chunk_indexes,
        missing_source_files=missing_source_files,
        suspicious_section_chunks=suspicious_section_chunks,
        suspected_artifact_chunks=suspected_artifact_chunks,
        overall_status=overall_status,
        critical_issues=critical_issues,
        warnings=warnings,
        informational_findings=informational,
        document_summaries={k: asdict(v) for k, v in doc_summaries.items()},
        duplicate_groups_detail=duplicate_groups_detail,
        near_duplicates_detail=near_duplicates_detail,
        suspicious_sections_detail=suspicious_sections_detail,
    )


def format_text_report(report: GlobalValidationReport) -> str:
    """Formats human-readable text report."""
    sep = "=" * 60
    subsep = "-" * 60
    lines: List[str] = []

    lines.append(sep)
    lines.append("Knowledge Chunk Quality Report")
    lines.append(sep)
    lines.append(f"Documents analyzed:     {report.total_documents}")
    lines.append(f"Total chunks:           {report.total_chunks}")
    lines.append(f"Valid chunks:           {report.valid_chunks}")
    lines.append(f"Overall Quality Status: {report.overall_status}")
    lines.append(sep)

    lines.append("\nCHUNK SIZE DISTRIBUTION:")
    lines.append(f"  Empty chunks (<1 char):      {report.empty_chunks}")
    lines.append(f"  Very short (<50 chars):      {report.very_short_chunks}")
    lines.append(f"  Short (50-200 chars):        {report.short_chunks}")
    lines.append(f"  Normal (200-1500 chars):     {report.normal_chunks}")
    lines.append(f"  Large (1500-2500 chars):     {report.large_chunks}")
    lines.append(f"  Very large (>2500 chars):    {report.very_large_chunks}")

    lines.append("\nINTEGRITY & DUPLICATION METRICS:")
    lines.append(f"  Exact duplicate chunks:      {report.exact_duplicate_chunks}")
    lines.append(f"  Exact duplicate groups:      {report.exact_duplicate_groups}")
    lines.append(f"  Near-duplicate pairs:        {report.near_duplicate_pairs}")
    lines.append(f"  Missing metadata fields:     {report.missing_metadata_fields}")
    lines.append(f"  Invalid page ranges:         {report.invalid_page_ranges}")
    lines.append(f"  Index sequence errors:       {report.invalid_chunk_indexes}")
    lines.append(f"  Missing source files:        {report.missing_source_files}")
    lines.append(f"  Suspicious section tags:     {report.suspicious_section_chunks}")
    lines.append(f"  Suspected text artifacts:    {report.suspected_artifact_chunks}")

    lines.append(f"\n{sep}")
    lines.append("DOCUMENT-LEVEL BREAKDOWN")
    lines.append(sep)
    for doc_id, s in report.document_summaries.items():
        lines.append(f"\nDocument: {doc_id}")
        lines.append(f"  Name:               {s['document_name']}")
        lines.append(f"  Source File:        {s['source_file']} (Exists: {s['source_file_exists']})")
        lines.append(f"  Page Range:         {s['min_page']} to {s['max_page']} ({s['pages_represented']} pages represented)")
        lines.append(f"  Chunks:             {s['total_chunks']}")
        lines.append(f"  Empty:              {s['empty_chunks']}")
        lines.append(f"  Very Short:         {s['very_short_chunks']}")
        lines.append(f"  Large / Very Large: {s['large_chunks']} / {s['very_large_chunks']}")
        lines.append(f"  Suspicious Sections:{len(s['suspicious_sections'])}")
        lines.append(f"  Quality Status:     {s['quality_status']}")

    lines.append(f"\n{sep}")
    lines.append("DIAGNOSTIC FINDINGS BREAKDOWN")
    lines.append(sep)

    lines.append(f"\n1. CRITICAL ISSUES ({len(report.critical_issues)}):")
    if report.critical_issues:
        for issue in report.critical_issues:
            lines.append(f"  [CRITICAL] {issue}")
    else:
        lines.append("  None. All required metadata, page bounds, and sequence indexes are intact.")

    lines.append(f"\n2. WARNINGS ({len(report.warnings)}):")
    if report.warnings:
        for warn in report.warnings:
            lines.append(f"  [WARNING] {warn}")
    else:
        lines.append("  None.")

    lines.append(f"\n3. INFORMATIONAL FINDINGS ({len(report.informational_findings)}):")
    if report.informational_findings:
        for info in report.informational_findings:
            lines.append(f"  [INFO] {info}")
    else:
        lines.append("  None.")

    if report.suspicious_sections_detail:
        lines.append(f"\n{subsep}")
        lines.append("SUSPICIOUS SECTION TAG DETAILS:")
        lines.append(subsep)
        for s in report.suspicious_sections_detail:
            lines.append(
                f"  Doc: {s['document_id']} | Chunk: {s['chunk_id']}\n"
                f"    Section: '{s['section']}'\n"
                f"    Reason:  {s['reason']}"
            )

    lines.append(f"\n{sep}")
    lines.append(f"FINAL CONCLUSION: {report.overall_status}")
    lines.append(sep)

    return "\n".join(lines)


def run_validation(
    output_json: Optional[Path] = None,
    output_txt: Optional[Path] = None,
) -> GlobalValidationReport:
    """Executes validation, prints formatted summary, and saves reports."""
    root = get_project_root()
    chunks_dir = root / "data" / "processed" / "chunks"
    policies_dir = root / "data" / "policies"

    report = validate_all_chunks(chunks_dir=chunks_dir, policies_dir=policies_dir)

    # Output paths
    json_path = output_json or (chunks_dir / "chunk_validation_report.json")
    txt_path = output_txt or (chunks_dir / "chunk_validation_report.txt")

    # Serialize JSON report
    report_dict = asdict(report)
    json_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    # Serialize TXT report
    text_report = format_text_report(report)
    txt_path.write_text(text_report, encoding="utf-8")

    logger.info("Validation report saved to: %s", json_path)
    logger.info("Human-readable report saved to: %s", txt_path)

    return report


def main() -> None:
    """CLI entrypoint."""
    report = run_validation()
    text_summary = format_text_report(report)
    print(text_summary)


if __name__ == "__main__":
    main()
