"""Transition chunk embeddings from 1536 to 384 dimensions for local BGE embeddings.

Revision ID: 0002_real_embeddings_384
Revises: 0001_initial_pgvector_schema
Create Date: 2026-09-17 12:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "0002_real_embeddings_384"
down_revision: Union[str, None] = "0001_initial_pgvector_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OLD_VECTOR_DIMENSION = 1536
NEW_VECTOR_DIMENSION = 384


def upgrade() -> None:
    # 1. SAFETY CHECK: Verify no production (non-mock) embeddings exist
    conn = op.get_bind()
    non_mock_count = conn.execute(
        text("SELECT COUNT(*) FROM chunk_embeddings WHERE is_mock = false;")
    ).scalar()

    if non_mock_count and non_mock_count > 0:
        raise RuntimeError(
            f"ABORTING MIGRATION: Detected {non_mock_count} production (non-mock) embeddings. "
            "Cannot safely drop or alter embedding vectors without explicit operator backup."
        )

    # 2. Clear old mock embeddings
    op.execute("DELETE FROM chunk_embeddings WHERE is_mock = true;")

    # 3. Drop old 1536-dim HNSW index
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw;")

    # 4. Alter column embedding type from Vector(1536) to Vector(384)
    op.execute(
        f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({NEW_VECTOR_DIMENSION});"
    )

    # 5. Recreate HNSW vector index for 384 dimensions
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw 
        ON chunk_embeddings 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )


def downgrade() -> None:
    # 1. Drop 384-dim HNSW index
    op.execute("DROP INDEX IF EXISTS ix_chunk_embeddings_hnsw;")

    # 2. Revert column embedding type to Vector(1536)
    op.execute("DELETE FROM chunk_embeddings;")
    op.execute(
        f"ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector({OLD_VECTOR_DIMENSION});"
    )

    # 3. Recreate HNSW vector index for 1536 dimensions
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_chunk_embeddings_hnsw 
        ON chunk_embeddings 
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """
    )
