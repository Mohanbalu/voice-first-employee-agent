"""RAG Service — Module 4.

High-level orchestrator: question → embedding → retrieval → context → LLM → answer.
Keeps all RAG logic independent from FastAPI routes.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from backend.app.config import config
    from backend.app.database import get_session_factory, is_db_reachable
    from backend.app.models.tenant import Tenant
    from backend.app.rag.answer_generator import AnswerGenerator, GeneratedAnswer, NO_CONTEXT_ANSWER
    from backend.app.rag.context_builder import BuiltContext, ContextBuilder, SourceReference
    from backend.app.rag.retriever import RAGRetriever, RetrievalConfig
    from backend.app.rag.embeddings import EmbeddingService
except ImportError:
    from app.config import config
    from app.database import get_session_factory, is_db_reachable
    from app.models.tenant import Tenant
    from app.rag.answer_generator import AnswerGenerator, GeneratedAnswer, NO_CONTEXT_ANSWER
    from app.rag.context_builder import BuiltContext, ContextBuilder, SourceReference
    from app.rag.retriever import RAGRetriever, RetrievalConfig
    from app.rag.embeddings import EmbeddingService

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger("rag.service")

# ── Retrieval quality → confidence label ─────────────────────────────────────
# Configurable thresholds (not calibrated probabilities — retrieval indicators only)
CONFIDENCE_HIGH: float = float(os.getenv("RAG_CONFIDENCE_HIGH", "0.85"))
CONFIDENCE_MEDIUM: float = float(os.getenv("RAG_CONFIDENCE_MEDIUM", "0.70"))
CONFIDENCE_LOW: float = float(os.getenv("RAG_CONFIDENCE_LOW", "0.50"))


def _derive_confidence(sources: List[SourceReference]) -> str:
    """
    Derives a retrieval quality label from the top source similarity score.

    This is a RETRIEVAL RELEVANCE INDICATOR, not a calibrated probability.
    Thresholds are configurable and should be tuned for the production embedding model.
    """
    if not sources:
        return "NO_MATCH"
    top_score = sources[0].similarity_score
    if top_score >= CONFIDENCE_HIGH:
        return "HIGH"
    if top_score >= CONFIDENCE_MEDIUM:
        return "MEDIUM"
    if top_score >= CONFIDENCE_LOW:
        return "LOW"
    return "NO_MATCH"


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""

    answer: str
    sources: List[SourceReference]
    retrieved_count: int        # candidates from pgvector before filtering
    used_context_count: int     # chunks actually sent to LLM
    confidence: str             # retrieval quality label: HIGH/MEDIUM/LOW/NO_MATCH
    mode: str                   # "production" | "mock_development"
    model: str                  # LLM model used
    latency_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "retrieved_count": self.retrieved_count,
            "used_context_count": self.used_context_count,
            "confidence": self.confidence,
            "mode": self.mode,
            "model": self.model,
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


class RAGService:
    """
    Full RAG pipeline:
    1. Validate tenant + question.
    2. Generate query embedding.
    3. pgvector similarity search (tenant-isolated).
    4. Filter mock vectors in production mode.
    5. Apply similarity threshold.
    6. Build context.
    7. Call LLM with grounding prompt.
    8. Attach sources.
    9. Return RAGResponse.
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        retrieval_config: Optional[RetrievalConfig] = None,
    ):
        self._retrieval_config = retrieval_config or RetrievalConfig()
        self._retriever = retriever or RAGRetriever(config=self._retrieval_config)
        self._context_builder = context_builder or ContextBuilder(
            max_chunks=self._retrieval_config.max_context_chunks
        )
        self._answer_generator = answer_generator or AnswerGenerator()

    def answer_question(
        self,
        tenant_id: uuid.UUID,
        question: str,
        session: Optional[Session] = None,
    ) -> RAGResponse:
        """
        Main RAG entry point. Accepts an open SQLAlchemy session (for testing/injection)
        or opens its own from the session factory.
        """
        start_time = time.monotonic()
        mode = "mock_development" if self._retrieval_config.allow_mock else "production"

        # Validate question
        if not question or not question.strip():
            return self._error_response(
                "Question must not be empty.",
                mode=mode,
                latency_ms=_elapsed_ms(start_time),
            )

        # Validate tenant (we need a session for this)
        def _run(session: Optional[Session]) -> RAGResponse:
            # 1. Validate tenant exists
            tenant = None
            if session is not None and is_db_reachable():
                try:
                    tenant = session.execute(
                        select(Tenant).where(Tenant.id == tenant_id)
                    ).scalar_one_or_none()
                except Exception as db_err:
                    logger.warning("Tenant DB lookup failed (%s); using fallback tenant.", db_err)
                    tenant = Tenant(id=tenant_id, name="HCL Enterprise", slug="dev-org")
            else:
                tenant = Tenant(id=tenant_id, name="HCL Enterprise", slug="dev-org")

            if tenant is None:
                return self._error_response(
                    f"Tenant {tenant_id} not found.",
                    mode=mode,
                    latency_ms=_elapsed_ms(start_time),
                )

            # 2. Retrieve candidates
            try:
                results = self._retriever.retrieve(
                    session=session,
                    tenant_id=tenant_id,
                    question=question,
                )
            except ValueError as exc:
                return self._error_response(
                    str(exc),
                    mode=mode,
                    latency_ms=_elapsed_ms(start_time),
                )
            except Exception as exc:
                logger.error("Retrieval failed: %s", exc)
                return self._error_response(
                    "Retrieval service error. Please try again.",
                    mode=mode,
                    latency_ms=_elapsed_ms(start_time),
                )

            retrieved_count = len(results)

            # 3. Build context
            built: BuiltContext = self._context_builder.build(results)

            # 4. Generate answer
            try:
                generated: GeneratedAnswer = self._answer_generator.generate(
                    question=question.strip(),
                    context_text=built.context_text,
                )
            except Exception as exc:
                logger.error("LLM answer generation failed: %s", exc)
                return self._error_response(
                    "LLM service error. Please try again.",
                    mode=mode,
                    latency_ms=_elapsed_ms(start_time),
                )

            confidence = _derive_confidence(built.sources)

            return RAGResponse(
                answer=generated.answer,
                sources=built.sources,
                retrieved_count=retrieved_count,
                used_context_count=built.chunk_count,
                confidence=confidence,
                mode=mode,
                model=generated.model,
                latency_ms=_elapsed_ms(start_time),
            )

        if session is not None:
            try:
                return _run(session)
            except Exception as exc:
                logger.warning("Session query failed in answer_question (%s). Retrying with offline fallback.", exc)
                return _run(session=None)

        try:
            session_factory = get_session_factory()
            with session_factory() as db_session:
                return _run(db_session)
        except Exception as db_err:
            logger.warning("Database unreachable in answer_question (%s). Using offline fallback.", db_err)
            return _run(session=None)

    @staticmethod
    def _error_response(
        message: str,
        mode: str,
        latency_ms: float,
    ) -> RAGResponse:
        return RAGResponse(
            answer=NO_CONTEXT_ANSWER,
            sources=[],
            retrieved_count=0,
            used_context_count=0,
            confidence="NO_MATCH",
            mode=mode,
            model="",
            latency_ms=latency_ms,
            error=message,
        )


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
