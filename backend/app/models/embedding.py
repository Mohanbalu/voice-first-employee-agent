"""Chunk Embedding Model with pgvector support."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.config import config
from backend.app.database import Base

if TYPE_CHECKING:
    from backend.app.models.tenant import Tenant
    from backend.app.models.chunk import Chunk


class ChunkEmbedding(Base):
    """Represents a vector embedding for a chunk, stored using pgvector and scoped to a tenant."""

    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("tenant_id", "chunk_id", name="uq_tenant_embedding_chunk_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_ref_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chunks.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    chunk_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    embedding: Mapped[List[float]] = mapped_column(
        Vector(config.db.vector_dimension), nullable=False
    )
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    is_mock: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    tenant: Mapped["Tenant"] = relationship("Tenant", back_populates="embeddings")
    chunk: Mapped["Chunk"] = relationship("Chunk", back_populates="embedding")

    def __repr__(self) -> str:
        return (
            f"<ChunkEmbedding(id={self.id}, tenant_id={self.tenant_id}, "
            f"chunk_id='{self.chunk_id}', dim={self.embedding_dimension}, is_mock={self.is_mock})>"
        )
