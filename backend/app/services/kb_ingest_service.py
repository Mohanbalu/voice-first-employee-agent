"""Knowledge Base Ingestion Service for Ticket Resolutions.

When HR answers a support ticket, this service ingests the Q&A pair as
a new knowledge chunk into the pgvector database so that the AI assistant
can retrieve and use it for future similar questions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger("kb.ingest")


def ingest_ticket_answer_to_kb(
    db: Session,
    tenant_id: uuid.UUID,
    ticket_number: str,
    question: str,
    answer: str,
    category: str,
) -> Optional[str]:
    """Ingests an HR-provided Q&A pair from a ticket into the RAG knowledge base.

    The Q&A pair is stored as:
    - A Document record (source = 'HR Ticket Resolutions')
    - A Chunk record with the formatted Q&A text
    - A ChunkEmbedding record with the computed vector

    Returns:
        chunk_id of the ingested chunk, or None if ingestion failed.
    """
    try:
        from backend.app.models.chunk import Chunk
        from backend.app.models.document import Document
        from backend.app.models.embedding import ChunkEmbedding
        from backend.app.rag.retriever import _build_embedding_service, RAG_ALLOW_MOCK
    except ImportError:
        from app.models.chunk import Chunk
        from app.models.document import Document
        from app.models.embedding import ChunkEmbedding
        from app.rag.retriever import _build_embedding_service, RAG_ALLOW_MOCK

    # ── 1. Compose the canonical Q&A text for the knowledge chunk ─────────────
    # Use a format that is semantically rich and RAG-retrieval friendly
    qa_text = (
        f"Question: {question.strip()}\n\n"
        f"Answer: {answer.strip()}\n\n"
        f"Category: {category}\n"
        f"Source: HR Support Ticket Resolution — {ticket_number}"
    )

    # ── 2. Get or create the 'HR Ticket Resolutions' virtual document ─────────
    doc_slug = f"hr_ticket_resolutions_{str(tenant_id)[:8]}"
    doc_name = "HR Ticket Resolutions"

    doc_obj = db.execute(
        select(Document).where(
            Document.tenant_id == tenant_id,
            Document.document_id == doc_slug,
        )
    ).scalar_one_or_none()

    if doc_obj is None:
        doc_obj = Document(
            tenant_id=tenant_id,
            document_id=doc_slug,
            document_name=doc_name,
            source_file="hr_ticket_resolutions.kb",
        )
        db.add(doc_obj)
        db.flush()
        logger.info("Created KB document: %s for tenant %s", doc_name, tenant_id)

    # ── 3. Build unique chunk_id for idempotency ───────────────────────────────
    chunk_id = f"ticket_qa_{ticket_number.lower().replace('-', '_')}"

    # Skip if already ingested (idempotent re-run safety)
    existing = db.execute(
        select(Chunk).where(
            Chunk.tenant_id == tenant_id,
            Chunk.chunk_id == chunk_id,
        )
    ).scalar_one_or_none()

    if existing is not None:
        logger.info("Chunk %s already exists in KB — skipping re-ingestion", chunk_id)
        return chunk_id

    # ── 4. Count existing chunks to determine chunk_index ─────────────────────
    from sqlalchemy import func
    chunk_count = db.execute(
        select(func.count(Chunk.id)).where(
            Chunk.tenant_id == tenant_id,
            Chunk.document_id == doc_slug,
        )
    ).scalar() or 0

    # ── 5. Create the Chunk record ─────────────────────────────────────────────
    metadata = {
        "document_id": doc_slug,
        "document_name": doc_name,
        "source_file": "hr_ticket_resolutions.kb",
        "ticket_number": ticket_number,
        "category": category,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "chunk_type": "ticket_resolution",
        "page_start": 1,
        "page_end": 1,
        "section": f"HR Resolution — {category}",
    }

    chunk_obj = Chunk(
        tenant_id=tenant_id,
        document_ref_id=doc_obj.id,
        document_id=doc_slug,
        chunk_id=chunk_id,
        chunk_index=int(chunk_count),
        text=qa_text,
        page_start=1,
        page_end=1,
        section=f"HR Resolution — {category}",
        metadata_json=metadata,
    )
    db.add(chunk_obj)
    db.flush()
    logger.info("Inserted KB chunk: %s", chunk_id)

    # ── 6. Compute and store the embedding ────────────────────────────────────
    try:
        embedding_service = _build_embedding_service(allow_mock=RAG_ALLOW_MOCK)
        vector = embedding_service.embed_text(qa_text)

        emb_obj = ChunkEmbedding(
            tenant_id=tenant_id,
            chunk_id=chunk_id,
            embedding=vector,
        )
        db.add(emb_obj)
        db.flush()
        logger.info(
            "Embedded KB chunk %s (dim=%d) for tenant %s",
            chunk_id, len(vector), tenant_id
        )
    except Exception as emb_err:
        # Embedding failure is non-fatal — chunk text is stored; just not searchable yet
        logger.error(
            "Embedding generation failed for chunk %s: %s — chunk stored without vector",
            chunk_id, emb_err
        )

    db.commit()
    return chunk_id
