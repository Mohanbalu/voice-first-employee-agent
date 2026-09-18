"""Timesheet Database Model.

Stores employee daily working hours, task descriptions, and approval states.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.app.database import Base


class TimesheetStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class Timesheet(Base):
    """Represents an employee timesheet / work hours entry."""

    __tablename__ = "timesheets"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
    )
    tenant_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    employee_id = Column(
        UUID(as_uuid=True),
        ForeignKey("employees.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    work_date = Column(Date, nullable=False, default=date.today, index=True)
    hours_worked = Column(Float, nullable=False, default=8.0)
    task_description = Column(Text, nullable=False)
    project_code = Column(String(50), default="GENERAL", nullable=False)

    status = Column(
        String(20),
        default="SUBMITTED",
        nullable=False,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    tenant = relationship("Tenant")
    user = relationship("User")
    employee = relationship("Employee")

    __table_args__ = (
        Index("ix_timesheets_tenant_user_date", "tenant_id", "user_id", "work_date"),
    )

    def __repr__(self) -> str:
        return f"<Timesheet id={self.id} date={self.work_date} hours={self.hours_worked} status={self.status}>"
