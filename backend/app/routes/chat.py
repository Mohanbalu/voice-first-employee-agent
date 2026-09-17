"""Chat Route — Module 4.

POST /api/chat — RAG-powered Q&A over company knowledge base.

⚠ DEVELOPMENT NOTE:
tenant_id is currently accepted from the request body for development and testing.
Module 8 (Authentication) will replace this with JWT-derived tenant identity.
Until then, the endpoint requires an explicit tenant_id in the request.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

try:
    from backend.app.config import config
    from backend.app.database import get_db
    from backend.app.rag.retriever import RetrievalConfig, RAG_ALLOW_MOCK
    from backend.app.rag.rag_service import RAGService
    from backend.app.schemas.chat import ChatRequest, ChatResponse, SourceRef
except ImportError:
    from app.config import config
    from app.database import get_db
    from app.rag.retriever import RetrievalConfig, RAG_ALLOW_MOCK
    from app.rag.rag_service import RAGService
    from app.schemas.chat import ChatRequest, ChatResponse, SourceRef

logger = logging.getLogger("routes.chat")
router = APIRouter(prefix="/api", tags=["chat"])

# Shared RAGService instance (stateless — safe to share across requests)
_rag_service: Optional[RAGService] = None


def _get_rag_service() -> RAGService:
    """Returns the shared RAGService, initialising it on first call."""
    global _rag_service
    if _rag_service is None:
        retrieval_cfg = RetrievalConfig()
        _rag_service = RAGService(retrieval_config=retrieval_cfg)
    return _rag_service


def _resolve_tenant_id(request: ChatRequest) -> uuid.UUID:
    """
    Resolves the tenant UUID from the request.

    Development mode:
    - If tenant_id is provided in the request body, use it.
    - Otherwise fall back to the configured DEV_TENANT_ID.

    Module 8 will replace this with authenticated tenant resolution.
    """
    raw = request.tenant_id or config.tenant.default_id
    try:
        return uuid.UUID(raw)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant_id: '{raw}'. Must be a valid UUID.",
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Company Policy Q&A",
    description=(
        "Ask a question about company policies. The answer is grounded in the "
        "retrieved company knowledge base using RAG. Sources are included in the response.\n\n"
        "**Development note**: `tenant_id` is accepted from the request body until "
        "Module 8 authentication is implemented."
    ),
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
) -> ChatResponse:
    """RAG-powered company policy Q&A endpoint."""
    tenant_id = _resolve_tenant_id(request)

    logger.info(
        "Chat request received. tenant=%s question_len=%d",
        tenant_id,
        len(request.question),
    )

    service = _get_rag_service()

    rag_resp = service.answer_question(
        tenant_id=tenant_id,
        question=request.question,
        session=db,
    )

    if rag_resp.error and not rag_resp.answer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=rag_resp.error,
        )

    # Convert SourceReference → SourceRef (Pydantic schema)
    sources = [
        SourceRef(
            document_name=s.document_name,
            source_file=s.source_file,
            page_start=s.page_start,
            page_end=s.page_end,
            section=s.section,
            chunk_id=s.chunk_id,
            similarity_score=s.similarity_score,
        )
        for s in rag_resp.sources
    ]

    return ChatResponse(
        answer=rag_resp.answer,
        sources=sources,
        retrieved_count=rag_resp.retrieved_count,
        used_context_count=rag_resp.used_context_count,
        confidence=rag_resp.confidence,
        mode=rag_resp.mode,
        model=rag_resp.model,
        latency_ms=rag_resp.latency_ms,
        error=rag_resp.error,
    )
