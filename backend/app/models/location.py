"""Office Location Model for HCL Campus Geo-Location Data."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base

if TYPE_CHECKING:
    from backend.app.models.tenant import Tenant


class Location(Base):
    """Represents a physical office or campus location with precise geographic coordinates."""

    __tablename__ = "locations"
    __table_args__ = (
        UniqueConstraint("tenant_id", "name", name="uq_tenant_location_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )
    location_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="e.g. CAMPUS_ENTRANCE, BUILDING",
    )
    latitude: Mapped[float] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        comment="Geographic latitude in decimal degrees (-90.0 to 90.0)",
    )
    longitude: Mapped[float] = mapped_column(
        Numeric(9, 6),
        nullable=False,
        comment="Geographic longitude in decimal degrees (-180.0 to 180.0)",
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        index=True,
    )
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
    tenant: Mapped[Optional[Tenant]] = relationship("Tenant", foreign_keys=[tenant_id])

    def __repr__(self) -> str:
        return (
            f"<Location(id={self.id}, name='{self.name}', type='{self.location_type}', "
            f"lat={self.latitude}, lon={self.longitude})>"
        )
