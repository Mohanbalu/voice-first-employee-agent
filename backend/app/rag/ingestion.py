"""Knowledge Base PDF Ingestion Pipeline.

Extracts, cleans, chunks, and persists company policy documents
into structured local formats (raw text, cleaned text, and JSONL chunks).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

# Dynamic imports supporting module and script execution modes
try:
    from backend.app.rag.chunking import DocumentChunk, chunk_document
except ImportError:
    try:
        from app.rag.chunking import DocumentChunk, chunk_document
    except ImportError:
        from chunking import DocumentChunk, chunk_document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("knowledge_base.ingestion")


@dataclass
class IngestionPaths:
    """Manages input and output directory paths for knowledge ingestion."""

    policies_dir: Path
    extracted_dir: Path
    cleaned_dir: Path
    chunks_dir: Path

    @classmethod
    def from_root(cls, root_path: Path) -> "IngestionPaths":
        """Instantiate standard paths relative to project root."""
        return cls(
            policies_dir=root_path / "data" / "policies",
            extracted_dir=root_path / "data" / "processed" / "extracted",
            cleaned_dir=root_path / "data" / "processed" / "cleaned",
            chunks_dir=root_path / "data" / "processed" / "chunks",
        )

    def ensure_directories(self) -> None:
        """Ensure all required output directories exist."""
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DocumentProcessingResult:
    """Record of an individual document processing attempt."""

    source_file: str
    document_id: str
    document_name: str
    success: bool
    page_count: int = 0
    chunk_count: int = 0
    error_message: Optional[str] = None


@dataclass
class IngestionSummary:
    """Consolidated summary of knowledge base ingestion execution."""

    documents_found: int
    documents_processed: int
    documents_failed: int
    total_pages: int
    total_chunks: int
    paths: IngestionPaths
    failures: Dict[str, str] = field(default_factory=dict)
    results: List[DocumentProcessingResult] = field(default_factory=list)


def get_project_root() -> Path:
    """Locate the project root directory containing 'data' or 'backend'."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "backend").exists():
            return parent
        if (parent / "data" / "policies").exists():
            return parent
    return Path.cwd()


def generate_document_id(filename_or_path: Path | str) -> str:
    """Generates a clean, normalized document slug identifier from filename.

    Example:
        'Travel-Policy.pdf' -> 'travel_policy'
        'HCL LEAVE AND HOLIDAY POLICY.pdf' -> 'hcl_leave_and_holiday_policy'
    """
    stem = Path(filename_or_path).stem
    # Replace non-alphanumeric characters with underscore
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem)
    slug = slug.strip("_").lower()
    return slug or "unnamed_document"


def generate_document_title(filename_or_path: Path | str) -> str:
    """Generates a human-friendly document title from filename."""
    stem = Path(filename_or_path).stem
    title = re.sub(r"[_\-]+", " ", stem)
    return " ".join(title.split()).title()


def extract_pdf_text(pdf_path: Path) -> Tuple[List[Tuple[int, str]], Dict[str, Any]]:
    """Extracts text from a PDF file page-by-page using pypdf.

    Args:
        pdf_path: Path to the target PDF.

    Returns:
        A tuple of:
          - List of (page_number, extracted_text) pairs (1-indexed).
          - Dictionary of document metadata and flags.

    Raises:
        ValueError: If file is encrypted, missing, or corrupt.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

    if pdf_path.stat().st_size == 0:
        raise ValueError(f"PDF file is empty (0 bytes): {pdf_path.name}")

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Failed to open PDF '{pdf_path.name}': {exc}") from exc

    if reader.is_encrypted:
        try:
            # Attempt empty password decryption
            decrypted = reader.decrypt("")
            if not decrypted:
                raise ValueError(f"PDF '{pdf_path.name}' is password protected.")
        except Exception as exc:
            raise ValueError(f"PDF '{pdf_path.name}' is password protected.") from exc

    pages_data: List[Tuple[int, str]] = []
    total_pages = len(reader.pages)

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            extracted = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Error extracting text on page %d of %s: %s", page_num, pdf_path.name, exc)
            extracted = ""
        pages_data.append((page_num, extracted))

    # Check if the document yielded any extractable text across all pages
    total_text = "".join(text for _, text in pages_data).strip()
    has_text = len(total_text) > 0

    metadata = {
        "total_pages": total_pages,
        "has_extractable_text": has_text,
    }

    return pages_data, metadata


def clean_text(text: str) -> str:
    """Conservatively cleans raw extracted policy text.

    - Normalizes line endings to LF (\n).
    - Removes non-printable extraction artifacts (\x00-\x08, \x0c, etc.).
    - Collapses multiple horizontal spaces/tabs into a single space per line.
    - Limits repeated blank lines to a maximum of 2 (preserving paragraphs).
    - Preserves headings, numbered sections, bullet points, numbers,
      dates, conditions, and policy wording without summarization.
    """
    if not text:
        return ""

    # Normalize line endings
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove non-printable control characters (keeping standard \n and \t)
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

    # Normalize horizontal whitespace per line
    lines: List[str] = []
    for line in cleaned.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line.strip())

    cleaned = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into exactly 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def clean_pages(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Applies clean_text to each page individually while preserving page indices."""
    return [(page_num, clean_text(text)) for page_num, text in pages]


def process_document(
    pdf_path: Path,
    paths: IngestionPaths,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    tenant_id: Optional[str] = None,
) -> DocumentProcessingResult:
    """Processes a single PDF document through extraction, cleaning, and chunking.

    Idempotent: Re-running cleanly overwrites existing extracted, cleaned, and chunk files.

    Args:
        pdf_path: Path to the PDF file.
        paths: IngestionPaths output directory specifications.
        chunk_size: Target maximum chunk size in characters.
        chunk_overlap: Target chunk overlap in characters.
        tenant_id: Optional tenant identifier for future SaaS tenancy.

    Returns:
        DocumentProcessingResult detailing status and counts.
    """
    doc_id = generate_document_id(pdf_path)
    doc_name = generate_document_title(pdf_path)
    source_file = pdf_path.name

    try:
        # Step 1: Extract PDF text
        raw_pages, meta = extract_pdf_text(pdf_path)
        page_count = meta["total_pages"]

        if not meta["has_extractable_text"]:
            logger.warning("PDF %s contains no extractable text.", source_file)

        # Step 2: Save raw extracted text with clear page separation
        extracted_file = paths.extracted_dir / f"{doc_id}.txt"
        extracted_content = []
        for p_num, p_text in raw_pages:
            extracted_content.append(f"--- PAGE {p_num} ---\n\n{p_text.strip()}\n")
        extracted_file.write_text("\n".join(extracted_content), encoding="utf-8")

        # Step 3: Clean text
        cleaned_pages = clean_pages(raw_pages)
        cleaned_file = paths.cleaned_dir / f"{doc_id}.txt"
        cleaned_content = []
        for p_num, p_text in cleaned_pages:
            cleaned_content.append(f"--- PAGE {p_num} ---\n\n{p_text}\n")
        cleaned_file.write_text("\n".join(cleaned_content), encoding="utf-8")

        # Step 4: Chunk document
        chunks = chunk_document(
            pages=cleaned_pages,
            document_id=doc_id,
            document_name=doc_name,
            source_file=source_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tenant_id=tenant_id,
        )

        # Step 5: Save chunks as JSONL
        chunk_file = paths.chunks_dir / f"{doc_id}.jsonl"
        with open(chunk_file, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            "Processed '%s': %d pages, %d chunks -> %s",
            source_file,
            page_count,
            len(chunks),
            chunk_file.name,
        )

        return DocumentProcessingResult(
            source_file=source_file,
            document_id=doc_id,
            document_name=doc_name,
            success=True,
            page_count=page_count,
            chunk_count=len(chunks),
        )

    except Exception as exc:
        logger.error("Failed to process '%s': %s", source_file, exc)
        return DocumentProcessingResult(
            source_file=source_file,
            document_id=doc_id,
            document_name=doc_name,
            success=False,
            error_message=str(exc),
        )


def process_all_documents(
    policies_dir: Optional[Path] = None,
    output_base: Optional[Path] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    tenant_id: Optional[str] = None,
) -> IngestionSummary:
    """Discovers and ingests all PDF policy documents.

    Args:
        policies_dir: Directory containing source PDFs (default: data/policies).
        output_base: Base directory for data/processed (default: project root).
        chunk_size: Chunk size in characters.
        chunk_overlap: Overlap in characters.
        tenant_id: Optional tenant identifier for future SaaS tenancy.

    Returns:
        IngestionSummary containing statistics and output paths.
    """
    root = get_project_root()
    paths = IngestionPaths(
        policies_dir=policies_dir or (root / "data" / "policies"),
        extracted_dir=(output_base or root) / "data" / "processed" / "extracted",
        cleaned_dir=(output_base or root) / "data" / "processed" / "cleaned",
        chunks_dir=(output_base or root) / "data" / "processed" / "chunks",
    )

    if not paths.policies_dir.exists():
        logger.error("Policies directory not found: %s", paths.policies_dir)
        paths.ensure_directories()
        return IngestionSummary(
            documents_found=0,
            documents_processed=0,
            documents_failed=0,
            total_pages=0,
            total_chunks=0,
            paths=paths,
            failures={"all": f"Policies directory does not exist: {paths.policies_dir}"},
        )

    paths.ensure_directories()

    # Discover all PDF files (case-insensitive glob)
    pdf_files = sorted(
        [p for p in paths.policies_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    )

    summary = IngestionSummary(
        documents_found=len(pdf_files),
        documents_processed=0,
        documents_failed=0,
        total_pages=0,
        total_chunks=0,
        paths=paths,
    )

    if not pdf_files:
        logger.warning("No PDF documents found in %s", paths.policies_dir)
        return summary

    logger.info("Found %d PDF document(s) in %s", len(pdf_files), paths.policies_dir)

    for pdf in pdf_files:
        result = process_document(
            pdf_path=pdf,
            paths=paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tenant_id=tenant_id,
        )
        summary.results.append(result)
        if result.success:
            summary.documents_processed += 1
            summary.total_pages += result.page_count
            summary.total_chunks += result.chunk_count
        else:
            summary.documents_failed += 1
            summary.failures[result.source_file] = result.error_message or "Unknown error"

    return summary


def print_summary(summary: IngestionSummary) -> None:
    """Prints a structured, user-friendly summary of the ingestion process."""
    separator = "=" * 50
    print(f"\n{separator}")
    print("Knowledge Base Ingestion Complete")
    print(separator)
    print(f"Documents found:        {summary.documents_found}")
    print(f"Successfully processed: {summary.documents_processed}")
    print(f"Failed:                 {summary.documents_failed}")
    print(f"Total pages extracted:  {summary.total_pages}")
    print(f"Total chunks created:   {summary.total_chunks}")
    print()
    print(f"Extracted text: {summary.paths.extracted_dir}")
    print(f"Cleaned text:   {summary.paths.cleaned_dir}")
    print(f"Chunks:         {summary.paths.chunks_dir}")
    print(separator)

    if summary.failures:
        print("\nFailed documents:")
        for filename, error in summary.failures.items():
            print(f"- {filename} -> {error}")
        print(separator)


def main() -> None:
    """Command-line entrypoint for knowledge base ingestion."""
    # Ensure project root is on sys.path for direct script execution
    root = get_project_root()
    backend_path = root / "backend"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))

    summary = process_all_documents()
    print_summary(summary)


if __name__ == "__main__":
    main()
