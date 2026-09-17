"""Chunking module for Knowledge Base documents.

Provides structured, section-aware, and paragraph-aware text chunking
with metadata generation designed for RAG ingestion and SaaS multi-tenancy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChunkMetadata:
    """Metadata container for a document chunk.

    Designed to be extensible for future multi-tenant SaaS features
    (e.g., tenant_id, vector_id, permissions) without schema breaking changes.
    """

    document_id: str
    document_name: str
    source_file: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    section: Optional[str] = None
    # Extensibility hooks for future modules:
    tenant_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    vector_id: Optional[str] = None
    doc_version: Optional[str] = None
    permissions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary, omitting None optional values."""
        data: Dict[str, Any] = {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "source_file": self.source_file,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
        }
        if self.section:
            data["section"] = self.section
        if self.tenant_id is not None:
            data["tenant_id"] = self.tenant_id
        if self.doc_version is not None:
            data["doc_version"] = self.doc_version
        if self.permissions is not None:
            data["permissions"] = self.permissions
        if self.vector_id is not None:
            data["vector_id"] = self.vector_id
        if self.embedding is not None:
            data["embedding"] = self.embedding
        return data


@dataclass
class DocumentChunk:
    """Represents an extracted chunk of text along with its metadata."""

    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to a JSON-compatible dictionary containing metadata and text."""
        result = self.metadata.to_dict()
        result["text"] = self.text
        return result


# Regular expressions for identifying policy headings and section titles
_SECTION_NUMBERED_RE = re.compile(
    r"^(?:(?:(?:\d+\.)+\d*|\d+\b|[A-Z]\.|\b(?:Section|Article|Clause|Policy|Part)\s+[A-Z0-9\.\-]+:?))\s+([A-Za-z0-9\s\-–—,':/()&]+)$",
    re.IGNORECASE,
)
_ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9\s\-–—,':/()&]{4,70}$")


def detect_section_heading(line: str) -> Optional[str]:
    """Detect if a given line is likely a policy section heading.

    Returns the cleaned heading text if detected, otherwise None.
    Avoids false positives on regular sentences or list items.
    """
    clean_line = line.strip()
    if not clean_line or len(clean_line) < 3 or len(clean_line) > 100:
        return None

    # Skip lines ending with sentence punctuation
    if clean_line.endswith((".", ";", ":", ",")):
        # Exception: "Section 1:" might end with colon
        if clean_line.endswith(":") and len(clean_line.split()) <= 6:
            return clean_line.rstrip(":").strip()
        return None

    # Numbered patterns (e.g., '1. Objective', '2.1 Scope', 'Section 4 Policy Guidelines')
    numbered_match = _SECTION_NUMBERED_RE.match(clean_line)
    if numbered_match:
        return clean_line

    # All-caps headings (e.g., 'BUSINESS TRAVEL POLICY', 'PURPOSE AND SCOPE')
    words = clean_line.split()
    if len(words) >= 1 and len(words) <= 10:
        if _ALL_CAPS_HEADING_RE.match(clean_line) and any(c.isalpha() for c in clean_line):
            return clean_line.title()

    return None


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving trailing punctuation and spacing."""
    if not text.strip():
        return []
    # Match punctuation followed by whitespace and capital letter or digit
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_by_words(text: str, max_size: int) -> List[str]:
    """Splits text into chunks of at most max_size by word boundaries or character slices."""
    words = text.split()
    if not words:
        if not text:
            return []
        return [text[j:j + max_size] for j in range(0, len(text), max_size)]

    fragments: List[str] = []
    current: List[str] = []
    curr_len = 0

    for w in words:
        if len(w) > max_size:
            if current:
                fragments.append(" ".join(current))
                current = []
                curr_len = 0
            for j in range(0, len(w), max_size):
                fragments.append(w[j:j + max_size])
            continue

        if curr_len + len(w) + 1 > max_size and current:
            fragments.append(" ".join(current))
            current = [w]
            curr_len = len(w)
        else:
            current.append(w)
            curr_len += len(w) + 1

    if current:
        fragments.append(" ".join(current))
    return fragments


def _split_long_text(text: str, max_size: int) -> List[str]:
    """Splits a long paragraph into sentence-bounded or word-bounded fragments."""
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return _split_by_words(text, max_size)

    fragments: List[str] = []
    current_sentences: List[str] = []
    curr_len = 0

    for sent in sentences:
        if len(sent) > max_size:
            if current_sentences:
                fragments.append(" ".join(current_sentences))
                current_sentences = []
                curr_len = 0
            fragments.extend(_split_by_words(sent, max_size))
            continue

        if curr_len + len(sent) + 1 > max_size and current_sentences:
            fragments.append(" ".join(current_sentences))
            current_sentences = [sent]
            curr_len = len(sent)
        else:
            current_sentences.append(sent)
            curr_len += len(sent) + 1

    if current_sentences:
        fragments.append(" ".join(current_sentences))

    return fragments


def chunk_document(
    pages: List[Tuple[int, str]],
    document_id: str,
    document_name: str,
    source_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    tenant_id: Optional[str] = None,
) -> List[DocumentChunk]:
    """Chunks a document into structured, paragraph- and sentence-aware segments.

    Args:
        pages: List of (page_number, text) tuples.
        document_id: Normalized document slug identifier.
        document_name: Human-friendly document title.
        source_file: Original filename (e.g., 'Travel-Policy.pdf').
        chunk_size: Maximum target character size for each chunk.
        chunk_overlap: Target character overlap between consecutive chunks.
        tenant_id: Optional future tenant identifier.

    Returns:
        List of DocumentChunk instances with rich metadata.
    """
    if not pages:
        return []

    # Validate parameters
    if chunk_size <= 0:
        chunk_size = 1000
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    # Structure document units: (page_num, text_block, optional_heading)
    units: List[Tuple[int, str, Optional[str]]] = []
    active_section: Optional[str] = None

    for page_num, page_text in pages:
        if not page_text or not page_text.strip():
            continue

        # Split page into paragraphs
        paragraphs = re.split(r"\n\s*\n", page_text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if this paragraph contains or starts with a heading
            lines = [line.strip() for line in para.split("\n") if line.strip()]
            if lines:
                detected = detect_section_heading(lines[0])
                if detected:
                    active_section = detected

            # Break overly long paragraphs to maintain target chunk_size
            if len(para) > chunk_size:
                sub_parts = _split_long_text(para, max_size=chunk_size)
                for part in sub_parts:
                    units.append((page_num, part, active_section))
            else:
                units.append((page_num, para, active_section))

    if not units:
        return []

    chunks: List[DocumentChunk] = []
    current_unit_indices: List[int] = []
    current_char_count = 0
    chunk_index = 0

    def build_chunk(indices: List[int]) -> DocumentChunk:
        nonlocal chunk_index
        text_parts = [units[i][1] for i in indices]
        combined_text = "\n\n".join(text_parts).strip()
        page_numbers = [units[i][0] for i in indices]
        page_start = min(page_numbers)
        page_end = max(page_numbers)

        # Determine prevailing section
        sections = [units[i][2] for i in indices if units[i][2]]
        section = sections[0] if sections else None

        chunk_id = f"{document_id}_{chunk_index + 1:04d}"
        metadata = ChunkMetadata(
            document_id=document_id,
            document_name=document_name,
            source_file=source_file,
            chunk_id=chunk_id,
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            section=section,
            tenant_id=tenant_id,
        )
        chunk_index += 1
        return DocumentChunk(text=combined_text, metadata=metadata)

    i = 0
    n = len(units)
    while i < n:
        unit_page, unit_text, unit_sec = units[i]
        unit_len = len(unit_text)

        # Check if adding this unit exceeds chunk_size
        if current_unit_indices and (current_char_count + unit_len + 2 > chunk_size):
            chunks.append(build_chunk(current_unit_indices))

            # Compute overlap: take trailing units until overlap target is met
            overlap_indices: List[int] = []
            overlap_chars = 0
            for idx in reversed(current_unit_indices):
                unit_chars = len(units[idx][1])
                if overlap_chars + unit_chars <= chunk_overlap or not overlap_indices:
                    overlap_indices.insert(0, idx)
                    overlap_chars += unit_chars + 2
                else:
                    break

            # If overlap included all units, drop the oldest to ensure progression
            if len(overlap_indices) == len(current_unit_indices):
                overlap_indices = overlap_indices[1:]

            current_unit_indices = overlap_indices
            current_char_count = sum(len(units[idx][1]) + 2 for idx in current_unit_indices)

        current_unit_indices.append(i)
        current_char_count += unit_len + 2
        i += 1

    if current_unit_indices:
        chunks.append(build_chunk(current_unit_indices))

    return chunks
