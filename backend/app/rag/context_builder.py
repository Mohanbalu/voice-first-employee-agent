"""RAG Context Builder — Module 4.

Assembles a structured LLM context string from retrieved SearchResults,
with source metadata preserved for later attribution.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    from backend.app.rag.vector_store import SearchResult
except ImportError:
    from app.rag.vector_store import SearchResult

logger = logging.getLogger("rag.context_builder")

RAG_MAX_CONTEXT_CHUNKS: int = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "6"))


@dataclass
class SourceReference:
    """Structured source citation attached to an answer."""

    document_name: str
    source_file: str
    page_start: int
    page_end: int
    section: Optional[str]
    chunk_id: str
    similarity_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BuiltContext:
    """Output of the context builder: formatted prompt text + source list."""

    context_text: str
    sources: List[SourceReference]
    chunk_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_text": self.context_text,
            "sources": [s.to_dict() for s in self.sources],
            "chunk_count": self.chunk_count,
        }


class ContextBuilder:
    """
    Converts a ranked list of SearchResults into:
    - A formatted context block for the LLM prompt.
    - A structured list of SourceReference objects for the API response.

    Design goals:
    - No internal database IDs in the LLM prompt.
    - Each chunk labelled with document name, pages, section.
    - Bounded number of chunks (configurable).
    """

    def __init__(self, max_chunks: int = RAG_MAX_CONTEXT_CHUNKS):
        self.max_chunks = max_chunks

    def build(self, results: List[SearchResult]) -> BuiltContext:
        """
        Assembles context from up to max_chunks results.
        Results should already be ranked by similarity (highest first).
        """
        selected = results[: self.max_chunks]
        if not selected:
            return BuiltContext(context_text="", sources=[], chunk_count=0)

        context_parts: List[str] = []
        sources: List[SourceReference] = []

        for i, result in enumerate(selected, start=1):
            section_line = f"\nSECTION: {result.section}" if result.section else ""
            page_range = (
                f"{result.page_start}–{result.page_end}"
                if result.page_start != result.page_end
                else str(result.page_start)
            )

            block = (
                f"[SOURCE {i}]\n"
                f"DOCUMENT: {result.document_name}\n"
                f"FILE: {result.source_file}\n"
                f"PAGES: {page_range}"
                f"{section_line}\n"
                f"CONTENT:\n{result.text.strip()}"
            )
            context_parts.append(block)

            sources.append(
                SourceReference(
                    document_name=result.document_name,
                    source_file=result.source_file,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    section=result.section,
                    chunk_id=result.chunk_id,
                    similarity_score=result.similarity_score,
                )
            )

        context_text = "\n\n---\n\n".join(context_parts)
        logger.debug("Built context from %d chunks.", len(selected))
        return BuiltContext(
            context_text=context_text,
            sources=sources,
            chunk_count=len(selected),
        )
