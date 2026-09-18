"""Schedule and Reminder Pydantic Schemas.

Type-safe request and response validation for scheduling endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ScheduleCreate(BaseModel):
    """Payload to create a new scheduled reminder or task."""
    title: str = Field(..., min_length=1, max_length=255, description="Summary or title of the reminder")
    description: Optional[str] = Field(None, description="Optional extra details or notes")
    scheduled_at: datetime = Field(..., description="Target execution timestamp (ISO-8601)")
    timezone: str = Field(default="Asia/Kolkata", description="Timezone name e.g. Asia/Kolkata")
    recurrence_type: str = Field(default="NONE", description="NONE | DAILY | WEEKLY | MONTHLY | WEEKDAYS")
    recurrence_rule: Optional[str] = Field(None, description="iCalendar RRULE string if recurring")
    reminder_type: str = Field(default="NOTIFICATION", description="NOTIFICATION | TASK | MEETING | TIMESHEET")
    duration_minutes: int = Field(default=15, ge=1, le=1440, description="Duration in minutes")


class ScheduleUpdate(BaseModel):
    """Payload to update an existing schedule."""
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    timezone: Optional[str] = None
    recurrence_type: Optional[str] = None
    recurrence_rule: Optional[str] = None
    status: Optional[str] = Field(None, description="PENDING | COMPLETED | CANCELLED | MISSED")
    reminder_type: Optional[str] = None
    duration_minutes: Optional[int] = Field(None, ge=1, le=1440)


class ScheduleResponse(BaseModel):
    """Public representation of a scheduled item."""
    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: uuid.UUID
    employee_id: Optional[uuid.UUID] = None
    title: str
    description: Optional[str] = None
    scheduled_at: datetime
    timezone: str
    recurrence_type: str
    recurrence_rule: Optional[str] = None
    status: str
    reminder_type: str
    duration_minutes: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ScheduleListResponse(BaseModel):
    """Paginated or filtered list of schedules."""
    total: int
    items: List[ScheduleResponse]

    @property
    def schedules(self) -> List[ScheduleResponse]:
        return self.items

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        d["schedules"] = d.get("items", [])
        return d


class ScheduleActionResponse(BaseModel):
    """Status acknowledgement after mutating a schedule."""
    success: bool
    message: str
    schedule: ScheduleResponse
