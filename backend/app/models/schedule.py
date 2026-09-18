"""Schedule / Reminder Database Model.

Stores user and employee schedules, reminders, and recurring tasks
with full tenant isolation and timezone awareness.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from backend.app.database import Base


class RecurrenceType(str, enum.Enum):
    NONE = "NONE"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    WEEKDAYS = "WEEKDAYS"


class ScheduleStatus(str, enum.Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    MISSED = "MISSED"


class ReminderType(str, enum.Enum):
    NOTIFICATION = "NOTIFICATION"
    TASK = "TASK"
    MEETING = "MEETING"
    TIMESHEET = "TIMESHEET"


class Schedule(Base):
    """Represents a scheduled reminder, calendar task, or recurring prompt."""

    __tablename__ = "schedules"

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

    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    scheduled_at = Column(DateTime(timezone=True), nullable=False, index=True)
    timezone = Column(String(50), default="Asia/Kolkata", nullable=False)

    recurrence_type = Column(
        String(20),
        default="NONE",
        nullable=False,
    )
    recurrence_rule = Column(String(100), nullable=True)  # e.g., "FREQ=WEEKLY;BYDAY=MO"

    status = Column(
        String(20),
        default="PENDING",
        nullable=False,
        index=True,
    )
    reminder_type = Column(
        String(30),
        default="NOTIFICATION",
        nullable=False,
    )
    duration_minutes = Column(Integer, default=15, nullable=False)

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
    completed_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    tenant = relationship("Tenant")
    user = relationship("User")
    employee = relationship("Employee")

    __table_args__ = (
        Index("ix_schedules_tenant_user_status", "tenant_id", "user_id", "status"),
        Index("ix_schedules_tenant_scheduled_at", "tenant_id", "scheduled_at"),
    )

    def __repr__(self) -> str:
        return f"<Schedule id={self.id} title={self.title!r} scheduled_at={self.scheduled_at} status={self.status}>"
