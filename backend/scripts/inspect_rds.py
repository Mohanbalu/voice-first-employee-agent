"""Read-only diagnostic script to inspect AWS RDS PostgreSQL database.

Reuses the exact database configuration and session factory from backend/app/database.py
as used by validate_real_embeddings.py.

Performs read-only SELECT queries to verify:
1. Current database name
2. Current PostgreSQL user
3. PostgreSQL version
4. All public schema tables
5. Row counts: tenants, documents, chunks, chunk_embeddings
6. chunk_embeddings: total count, mock count, real count, vector dimension
7. 5 sample rows from documents
8. 5 sample rows from chunks (id, document_id, page_start/page_end, first 200 chars of text)
9. HNSW index verification on chunk_embeddings
10. Strict safety: no passwords, API keys, DATABASE_URL, or embedding values printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root in sys.path
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from sqlalchemy import text
from backend.app.database import get_session_factory


def run_diagnostic() -> None:
    session_factory = get_session_factory()

    with session_factory() as session:
        print("=" * 60)
        print("       AWS RDS POSTGRESQL READ-ONLY DIAGNOSTIC REPORT       ")
        print("=" * 60)

        # 1. Current database name
        db_name = session.execute(text("SELECT current_database();")).scalar()
        print(f"1. Current Database:    {db_name}")

        # 2. Current PostgreSQL user
        user = session.execute(text("SELECT current_user;")).scalar()
        print(f"2. Current User:        {user}")

        # 3. PostgreSQL version
        version = session.execute(text("SELECT version();")).scalar()
        print(f"3. PostgreSQL Version:  {version}")

        # 4. All public schema tables
        tables_query = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name;"
        )
        tables = [row[0] for row in session.execute(tables_query).fetchall()]
        print(f"4. Public Schema Tables: {', '.join(tables)}")

        # 5. Row counts
        print("\n5. Row Counts:")
        for tbl in ["tenants", "documents", "chunks", "chunk_embeddings"]:
            count = session.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
            print(f"   - {tbl:18} : {count}")

        # 6. chunk_embeddings details
        total_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings;")).scalar() or 0
        mock_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings WHERE is_mock = true;")).scalar() or 0
        real_emb = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings WHERE is_mock = false;")).scalar() or 0
        dim = session.execute(text("SELECT vector_dims(embedding) FROM chunk_embeddings LIMIT 1;")).scalar() or "N/A"

        print("\n6. chunk_embeddings Details:")
        print(f"   - Total count       : {total_emb}")
        print(f"   - Mock count        : {mock_emb}")
        print(f"   - Real count        : {real_emb}")
        print(f"   - Vector dimension  : {dim}")

        # 7. Show 5 sample rows from documents
        print("\n7. Sample Rows from documents (5 rows):")
        docs_query = text(
            "SELECT id, document_id, document_name, source_file "
            "FROM documents ORDER BY created_at LIMIT 5;"
        )
        docs = session.execute(docs_query).fetchall()
        for idx, d in enumerate(docs, 1):
            print(f"   [{idx}] ID: {d[0]}")
            print(f"       doc_id:      {d[1]}")
            print(f"       doc_name:    {d[2]}")
            print(f"       source_file: {d[3]}")

        # 8. Show 5 sample rows from chunks
        print("\n8. Sample Rows from chunks (5 rows):")
        chunks_query = text(
            "SELECT id, document_id, page_start, page_end, LEFT(text, 200) "
            "FROM chunks ORDER BY created_at, chunk_index LIMIT 5;"
        )
        chunks = session.execute(chunks_query).fetchall()
        for idx, c in enumerate(chunks, 1):
            clean_snippet = c[4].replace("\n", " ").strip()
            print(f"   [{idx}] ID: {c[0]}")
            print(f"       document_id: {c[1]}")
            print(f"       page_number: {c[2]} (page_end: {c[3]})")
            print(f"       content[:200]: {clean_snippet}...")

        # 9. Verify HNSW index exists on chunk_embeddings
        print("\n9. HNSW Index Verification on chunk_embeddings:")
        index_query = text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'chunk_embeddings' AND indexdef ILIKE '%hnsw%';"
        )
        indexes = session.execute(index_query).fetchall()
        if indexes:
            for idx in indexes:
                print(f"   [EXISTS] Index Name: {idx[0]}")
                print(f"            Definition: {idx[1]}")
        else:
            print("   [FAIL] No HNSW index found on chunk_embeddings table.")

        print("=" * 60)


if __name__ == "__main__":
    run_diagnostic()
