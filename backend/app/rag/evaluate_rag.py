"""RAG Evaluation Utility — Module 4.

Runs a set of manually defined questions through the RAG pipeline and
reports retrieval quality, context selection, and answers.

IMPORTANT:
- Does NOT make real OpenAI calls unless OPENAI_API_KEY is configured.
- Clearly reports MOCK mode when mock vectors/embeddings are used.
- Does NOT present mock-vector results as real retrieval quality.

Usage:
    py backend/app/rag/evaluate_rag.py

    # Force mock mode (no OpenAI credits spent):
    RAG_ALLOW_MOCK=true py backend/app/rag/evaluate_rag.py

    # With real OpenAI key and production embeddings:
    py backend/app/rag/evaluate_rag.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

# Ensure project root in sys.path
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

try:
    from dotenv import load_dotenv
    load_dotenv(Path("backend/.env"))
except ImportError:
    pass

from backend.app.config import config
from backend.app.database import get_session_factory
from backend.app.rag.retriever import RetrievalConfig, RAG_ALLOW_MOCK
from backend.app.rag.rag_service import RAGService
from backend.app.models.embedding import ChunkEmbedding
from sqlalchemy import select, func


EVAL_QUESTIONS = [
    "What is the annual leave policy?",
    "Can employees work from home?",
    "What is the travel reimbursement policy?",
    "What is the IT support process?",
    "What is the health insurance policy?",
    "What is the disciplinary procedure?",
    "What is the policy on business expenses?",
    "What is the environmental policy?",
]


def _check_mock_status(session_factory) -> tuple[int, int]:
    """Returns (mock_count, production_count) from the database."""
    with session_factory() as session:
        mock_count = session.execute(
            select(func.count(ChunkEmbedding.id)).where(ChunkEmbedding.is_mock == True)
        ).scalar() or 0
        prod_count = session.execute(
            select(func.count(ChunkEmbedding.id)).where(ChunkEmbedding.is_mock == False)
        ).scalar() or 0
    return mock_count, prod_count


def run_evaluation() -> None:
    allow_mock = RAG_ALLOW_MOCK
    tenant_id = uuid.UUID(config.tenant.default_id)

    session_factory = get_session_factory()
    mock_count, prod_count = _check_mock_status(session_factory)

    sep = "=" * 65
    print(f"\n{sep}")
    print("RAG EVALUATION — Voice-First Employee Assistant")
    print(sep)
    print(f"Tenant:                  {config.tenant.default_slug} ({tenant_id})")
    print(f"Mock embeddings in DB:   {mock_count}")
    print(f"Production embeddings:   {prod_count}")
    print(f"RAG_ALLOW_MOCK:          {allow_mock}")
    print(f"RAG_MIN_SIMILARITY:      {os.getenv('RAG_MIN_SIMILARITY', '0.70')}")
    print(f"LLM_MODEL:               {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")

    if prod_count == 0:
        print("\n*** EVALUATION MODE: MOCK / NOT SEMANTICALLY VALIDATED ***")
        print("*** Production embeddings are NOT present in the database. ***")
        print("*** Results below reflect mock-vector retrieval only.      ***")
        if not allow_mock:
            print("\n[SKIP] RAG_ALLOW_MOCK=false and no production embeddings present.")
            print("Set RAG_ALLOW_MOCK=true to evaluate with mock vectors,")
            print("or generate real embeddings first:")
            print("  py backend/app/rag/embeddings.py")
            print(sep)
            return

    print(sep)

    retrieval_cfg = RetrievalConfig(allow_mock=allow_mock)
    service = RAGService(retrieval_config=retrieval_cfg)

    api_key = os.getenv("OPENAI_API_KEY", "")
    llm_available = api_key and api_key != "your_openai_api_key_placeholder"

    if not llm_available:
        print("\n[INFO] OPENAI_API_KEY not configured — LLM answer generation skipped.")
        print("[INFO] Showing retrieval-only results.\n")

    for i, question in enumerate(EVAL_QUESTIONS, start=1):
        print(f"\n[Q{i}] {question}")
        print("-" * 65)

        try:
            with session_factory() as session:
                from backend.app.models.tenant import Tenant
                from sqlalchemy.orm import Session

                tenant = session.execute(
                    select(Tenant).where(Tenant.id == tenant_id)
                ).scalar_one_or_none()

                if tenant is None:
                    print(f"  [ERROR] Tenant {tenant_id} not found in database.")
                    continue

                # Get retrieval results
                from backend.app.rag.retriever import RAGRetriever
                retriever = RAGRetriever(config=retrieval_cfg)
                try:
                    results = retriever.retrieve(
                        session=session,
                        tenant_id=tenant_id,
                        question=question,
                    )
                except Exception as exc:
                    print(f"  [RETRIEVAL ERROR] {exc}")
                    continue

                if not results:
                    print("  Retrieved:  0 chunks above threshold")
                    print("  Answer:     [NO MATCH — no relevant chunks found]")
                    continue

                print(f"  Retrieved:  {len(results)} chunks above threshold")
                for j, r in enumerate(results, start=1):
                    pages = (
                        f"{r.page_start}–{r.page_end}"
                        if r.page_start != r.page_end
                        else str(r.page_start)
                    )
                    section_str = f" | {r.section}" if r.section else ""
                    print(
                        f"  [{j}] score={r.similarity_score:.4f} | {r.document_name} | "
                        f"pp.{pages}{section_str}"
                    )

                if llm_available:
                    resp = service.answer_question(
                        tenant_id=tenant_id,
                        question=question,
                        session=session,
                    )
                    print(f"\n  Confidence: {resp.confidence}")
                    print(f"  Answer:\n  {resp.answer[:500]}{'...' if len(resp.answer) > 500 else ''}")
                else:
                    print("\n  [LLM SKIPPED — OPENAI_API_KEY not set]")

        except Exception as exc:
            print(f"  [ERROR] {exc}")

    print(f"\n{sep}")
    print("Evaluation complete.")
    if prod_count == 0:
        print("REMINDER: Results above used MOCK vectors — NOT semantically valid.")
        print("Generate real embeddings with: py backend/app/rag/embeddings.py")
    print(sep)


if __name__ == "__main__":
    run_evaluation()
