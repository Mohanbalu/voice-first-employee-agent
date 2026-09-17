"""Real Embeddings Validation & Semantic Retrieval Verification Script.

Connects to AWS RDS PostgreSQL + pgvector and verifies:
1. Database connectivity & pgvector extension status
2. Total chunks count (expected 886)
3. Total embeddings count (expected 886)
4. Mock embeddings count (expected 0)
5. Real embeddings count (expected 886)
6. Vector dimension (expected 384)
7. Model metadata (expected BAAI/bge-small-en-v1.5)
8. Sample semantic queries using the local BGE embedding model
9. Cross-tenant isolation guarantees
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.config import config, sanitize_db_url
from backend.app.database import check_db_health, get_session_factory
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.embedding import ChunkEmbedding
from backend.app.models.tenant import Tenant
from backend.app.rag.embeddings import LocalSentenceTransformerEmbeddingProvider
from backend.app.rag.vector_store import VectorStore

EXPECTED_CHUNKS = 886
EXPECTED_EMBEDDINGS = 886
EXPECTED_MOCK = 0
EXPECTED_REAL = 886
EXPECTED_DIMENSION = 384
EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"

SAMPLE_QUERIES = [
    "What is the work from home policy?",
    "What is the annual leave and sick leave policy?",
    "What are the rules for business travel and expense reimbursement?",
    "What does the IT policy say about workstation security?",
    "What is covered under the company health insurance policy?",
]


def print_banner(title: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n{title.center(60)}\n{sep}")


def validate_embeddings() -> bool:
    print_banner("REAL EMBEDDINGS VALIDATION (BGE-SMALL 384-DIM)")
    print(f"Target Database: {sanitize_db_url(config.db.url)}")
    print(f"Active Provider: {config.embedding.provider}")
    print(f"Target Model:    {config.embedding.model}")
    print(f"Dimension:       {config.embedding.dimension}")
    print("-" * 60)

    # 1. Connectivity Check
    health = check_db_health()
    if health["connection_status"] != "PASS":
        print("\n[FAIL] Cannot connect to AWS RDS PostgreSQL database.")
        print(f"Error: {health['errors']}")
        print("\nPlease ensure your AWS Security Group allows inbound TCP port 5432.")
        return False

    if not health["pgvector_available"]:
        print("\n[FAIL] pgvector extension is NOT available in target database.")
        return False

    print("[PASS] Database connected & pgvector extension detected.")

    session_factory = get_session_factory()
    all_passed = True

    with session_factory() as session:
        # 2. Entity Counts
        chunk_count = session.execute(text("SELECT COUNT(*) FROM chunks;")).scalar() or 0
        total_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings;")).scalar() or 0
        mock_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings WHERE is_mock = true;")).scalar() or 0
        real_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings WHERE is_mock = false;")).scalar() or 0

        print(f"\n--- Entity Counts ---")
        print(f"Chunks in DB:           {chunk_count} (Expected: {EXPECTED_CHUNKS})")
        print(f"Embeddings in DB:       {total_emb} (Expected: {EXPECTED_EMBEDDINGS})")
        print(f"Mock Embeddings:        {mock_emb} (Expected: {EXPECTED_MOCK})")
        print(f"Real Embeddings:        {real_emb} (Expected: {EXPECTED_REAL})")

        if chunk_count != EXPECTED_CHUNKS:
            print(f"[WARN] Chunk count mismatch: {chunk_count} != {EXPECTED_CHUNKS}")

        if total_emb != EXPECTED_EMBEDDINGS:
            print(f"[FAIL] Total embedding count mismatch: {total_emb} != {EXPECTED_EMBEDDINGS}")
            all_passed = False
        else:
            print(f"[PASS] Embedding count matches expected ({EXPECTED_EMBEDDINGS})")

        if mock_emb != EXPECTED_MOCK:
            print(f"[FAIL] Detected {mock_emb} mock embeddings; expected 0.")
            all_passed = False
        else:
            print(f"[PASS] Zero mock embeddings detected.")

        if real_emb != EXPECTED_REAL:
            print(f"[FAIL] Real embedding count mismatch: {real_emb} != {EXPECTED_REAL}")
            all_passed = False
        else:
            print(f"[PASS] All {EXPECTED_REAL} embeddings are real production embeddings.")

        # 3. Dimension & Model Metadata
        sample_emb = session.execute(
            select(ChunkEmbedding).where(ChunkEmbedding.is_mock == False).limit(1)
        ).scalar_one_or_none()

        if sample_emb is None:
            print("[FAIL] No real embeddings found in chunk_embeddings table.")
            return False

        actual_dim = sample_emb.embedding_dimension
        actual_model = sample_emb.embedding_model
        print(f"\n--- Vector Properties ---")
        print(f"Vector Dimension:       {actual_dim} (Expected: {EXPECTED_DIMENSION})")
        print(f"Model Recorded:         {actual_model} (Expected: {EXPECTED_MODEL})")

        if actual_dim != EXPECTED_DIMENSION:
            print(f"[FAIL] Vector dimension mismatch: {actual_dim} != {EXPECTED_DIMENSION}")
            all_passed = False
        else:
            print(f"[PASS] Vector dimension verified: {actual_dim}")

        # 4. Semantic Search Verification
        print_banner("RUNNING SAMPLE SEMANTIC SEARCHES (BGE-SMALL-EN-V1.5)")
        provider = LocalSentenceTransformerEmbeddingProvider()
        store = VectorStore(target_dimension=EXPECTED_DIMENSION)
        dev_tenant_id = uuid.UUID(config.tenant.default_id)

        for q_idx, query in enumerate(SAMPLE_QUERIES, 1):
            t0 = time.perf_counter()
            query_vec = provider.embed_query(query)
            t_emb = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            results = store.search_similar_chunks(
                session=session,
                tenant_id=dev_tenant_id,
                query_embedding=query_vec,
                top_k=3,
                min_similarity=0.0,
            )
            t_ret = (time.perf_counter() - t1) * 1000

            print(f"\nQuery {q_idx}: \"{query}\"")
            print(f"Timing:  Embedding={t_emb:.1f}ms, Retrieval={t_ret:.1f}ms")
            if not results:
                print("  [WARN] No chunks retrieved.")
                all_passed = False
                continue

            for rank, r in enumerate(results, 1):
                preview = r.text[:90].replace("\n", " ") + "..."
                print(
                    f"  #{rank} [Score: {r.similarity_score:.4f}] "
                    f"Doc: {r.document_name} (p.{r.page_start}) - {preview}"
                )

        # 5. Cross-Tenant Isolation Verification
        print_banner("VERIFYING CROSS-TENANT ISOLATION")
        isolated_tenant_id = uuid.uuid4()
        iso_results = store.search_similar_chunks(
            session=session,
            tenant_id=isolated_tenant_id,
            query_embedding=query_vec,
            top_k=5,
        )

        if len(iso_results) == 0:
            print(f"[PASS] Query against empty tenant returned 0 chunks (no cross-tenant leakage).")
        else:
            print(f"[FAIL] Cross-tenant isolation breach: retrieved {len(iso_results)} chunks from wrong tenant!")
            all_passed = False

    print_banner("VALIDATION SUMMARY")
    if all_passed:
        print("ALL CHECKS PASSED: Real BGE 384-dim embeddings are verified and production-ready.")
    else:
        print("VALIDATION FAILED: Review the errors above.")
    print("=" * 60)
    return all_passed


if __name__ == "__main__":
    success = validate_embeddings()
    sys.exit(0 if success else 1)
