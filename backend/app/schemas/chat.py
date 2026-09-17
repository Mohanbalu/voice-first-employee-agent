"""Chat API Schemas — Module 4.

Pydantic v2 request/response models for the POST /api/chat endpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SourceRef(BaseModel):
    """A single source cited in a RAG answer."""

    document_name: str
    source_file: str
    page_start: int
    page_end: int
    section: Optional[str] = None
    chunk_id: str
    similarity_score: float


class ChatRequest(BaseModel):
    """
    Request body for POST /api/chat.

    NOTE (Development-only):
    The `tenant_id` field is accepted directly in the request body for
    development and testing. In Module 8 (Authentication), the tenant identity
    will be derived from the authenticated JWT token and this field will be removed.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The employee's question about company policy.",
        examples=["How many days of annual leave can I take?"],
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description=(
            "[DEVELOPMENT ONLY] Explicit tenant UUID. "
            "Will be replaced by authenticated identity in Module 8."
        ),
        examples=["00000000-0000-0000-0000-000000000001"],
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question must not be blank.")
        return v.strip()


class ChatResponse(BaseModel):
    """
    Structured response from the RAG pipeline.

    Fields:
    - answer: The grounded answer text.
    - sources: Sources actually used for the answer.
    - retrieved_count: Number of candidates retrieved from pgvector.
    - used_context_count: Number of chunks sent to the LLM.
    - confidence: Retrieval quality indicator (HIGH/MEDIUM/LOW/NO_MATCH).
      NOTE: This is a RETRIEVAL RELEVANCE INDICATOR, not a calibrated probability.
    - mode: "production" or "mock_development".
    - model: LLM model identifier.
    - latency_ms: End-to-end latency in milliseconds.
    - error: Optional error description if the pipeline encountered an issue.
    """

    answer: str
    sources: List[SourceRef] = Field(default_factory=list)
    retrieved_count: int
    used_context_count: int
    confidence: str = Field(description="HIGH | MEDIUM | LOW | NO_MATCH")
    mode: str = Field(description="production | mock_development")
    model: str
    latency_ms: float
    error: Optional[str] = None


class ChatErrorResponse(BaseModel):
    """Error response for non-2xx chat responses."""

    detail: str
    code: Optional[str] = None
