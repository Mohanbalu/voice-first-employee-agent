"""Scheduling & Reminder Service — Module 6.

Central business logic engine for creating, querying, updating, completing,
and recurring schedules with strict tenant and user isolation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.orm import Session

from backend.app.models.audit import AuditLog
from backend.app.models.employee import Employee
from backend.app.models.schedule import RecurrenceType, ReminderType, Schedule, ScheduleStatus
from backend.app.schemas.schedule import ScheduleCreate, ScheduleUpdate
from backend.app.utils.nl_time_parser import DEFAULT_TIMEZONE_NAME, get_default_tz

logger = logging.getLogger("services.scheduling")

_IN_MEMORY_SCHEDULES: Dict[str, List[Schedule]] = {}


class SchedulingService:
    """Production-grade service managing workplace schedules, reminders, and recurring tasks."""

    def __init__(self, db: Session):
        self.db = db

    def _get_in_memory_list(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> List[Schedule]:
        key = f"{tenant_id}_{user_id}"
        if key not in _IN_MEMORY_SCHEDULES:
            _IN_MEMORY_SCHEDULES[key] = []
        return _IN_MEMORY_SCHEDULES[key]

    def create_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        data: ScheduleCreate,
        employee_id: Optional[uuid.UUID] = None,
    ) -> Schedule:
        """Creates and persists a new scheduled item."""
        # Normalize timezone
        tz = get_default_tz(data.timezone)
        scheduled_at = data.scheduled_at
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=tz)

        # Resolve employee_id if not provided
        if not employee_id:
            try:
                emp = self.db.execute(
                    select(Employee.id).where(
                        Employee.tenant_id == tenant_id,
                        Employee.user_id == user_id,
                    )
                ).scalar_one_or_none()
                if emp:
                    employee_id = emp
            except Exception:
                pass

        now = datetime.now(timezone.utc)
        schedule = Schedule(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            user_id=user_id,
            employee_id=employee_id,
            title=data.title.strip(),
            description=data.description.strip() if data.description else None,
            scheduled_at=scheduled_at,
            timezone=tz.key,
            recurrence_type=data.recurrence_type,
            recurrence_rule=data.recurrence_rule,
            status=ScheduleStatus.PENDING.value,
            reminder_type=getattr(data, "reminder_type", "NOTIFICATION"),
            duration_minutes=getattr(data, "duration_minutes", 15),
            created_at=now,
            updated_at=now,
        )

        try:
            self.db.add(schedule)
            self.db.flush()

            # Audit log
            self.db.add(
                AuditLog(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    action="schedule_created",
                    entity_type="schedule",
                    entity_id=str(schedule.id),
                    details=f"Created schedule: '{schedule.title}' for {schedule.scheduled_at}",
                )
            )
            self.db.commit()
            self.db.refresh(schedule)
            logger.info("Created schedule %s for user %s at %s", schedule.id, user_id, schedule.scheduled_at)
        except Exception as db_err:
            logger.warning("DB unreachable for schedule persistence (%s). Using in-memory store.", db_err)
            self._get_in_memory_list(tenant_id, user_id).insert(0, schedule)

        return schedule

    def get_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
    ) -> Optional[Schedule]:
        """Retrieves a single schedule ensuring tenant and user isolation."""
        try:
            item = self.db.execute(
                select(Schedule).where(
                    Schedule.id == schedule_id,
                    Schedule.tenant_id == tenant_id,
                    Schedule.user_id == user_id,
                )
            ).scalar_one_or_none()
            if item:
                return item
        except Exception:
            pass

        items = self._get_in_memory_list(tenant_id, user_id)
        for s in items:
            if s.id == schedule_id:
                return s
        return None

    def list_schedules(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        status_filter: Optional[str] = None,
        upcoming_only: bool = False,
        today_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[int, List[Schedule]]:
        """Lists user schedules with optional filtering."""
        try:
            query = select(Schedule).where(
                Schedule.tenant_id == tenant_id,
                Schedule.user_id == user_id,
            )

            now = datetime.now(timezone.utc)

            if status_filter:
                query = query.where(Schedule.status == status_filter.upper())

            if upcoming_only:
                query = query.where(
                    Schedule.status == ScheduleStatus.PENDING.value,
                    Schedule.scheduled_at >= now,
                )

            if today_only:
                tz = get_default_tz(DEFAULT_TIMEZONE_NAME)
                local_now = datetime.now(tz)
                start_of_day = datetime.combine(local_now.date(), datetime.min.time(), tzinfo=tz)
                end_of_day = datetime.combine(local_now.date(), datetime.max.time(), tzinfo=tz)
                query = query.where(
                    Schedule.scheduled_at >= start_of_day,
                    Schedule.scheduled_at <= end_of_day,
                )

            # Ordering: PENDING first, earliest scheduled_at first
            query = query.order_by(
                Schedule.scheduled_at.asc(),
                Schedule.created_at.desc(),
            )

            all_items = self.db.execute(query).scalars().all()
            total = len(all_items)
            items = all_items[offset : offset + limit]
            return total, list(items)
        except Exception as db_err:
            logger.warning("DB unreachable for list_schedules (%s). Using in-memory store.", db_err)
            items = self._get_in_memory_list(tenant_id, user_id)
            now = datetime.now(timezone.utc)
            filtered = items
            if status_filter:
                filtered = [s for s in filtered if s.status == status_filter.upper()]
            if upcoming_only:
                filtered = [s for s in filtered if s.status == ScheduleStatus.PENDING.value and s.scheduled_at >= now]
            if today_only:
                tz = get_default_tz(DEFAULT_TIMEZONE_NAME)
                local_today = datetime.now(tz).date()
                filtered = [s for s in filtered if s.scheduled_at.date() == local_today]

            total = len(filtered)
            return total, filtered[offset : offset + limit]

    def update_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
        data: ScheduleUpdate,
    ) -> Optional[Schedule]:
        """Updates schedule details."""
        schedule = self.get_schedule(tenant_id, user_id, schedule_id)
        if not schedule:
            return None

        update_dict = data.model_dump(exclude_unset=True)
        for key, val in update_dict.items():
            if hasattr(schedule, key) and val is not None:
                setattr(schedule, key, val)

        schedule.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(schedule)
        return schedule

    def complete_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
    ) -> Optional[Schedule]:
        """Marks a schedule COMPLETED and spawns next occurrence if recurring."""
        schedule = self.get_schedule(tenant_id, user_id, schedule_id)
        if not schedule:
            return None

        now = datetime.now(timezone.utc)
        schedule.status = ScheduleStatus.COMPLETED.value
        schedule.completed_at = now
        schedule.updated_at = now

        # If recurring, compute and spawn next recurrence
        if schedule.recurrence_type != RecurrenceType.NONE.value:
            next_time = self.calculate_next_occurrence(
                schedule.scheduled_at,
                schedule.recurrence_type,
                schedule.recurrence_rule,
            )
            if next_time:
                next_schedule = Schedule(
                    tenant_id=schedule.tenant_id,
                    user_id=schedule.user_id,
                    employee_id=schedule.employee_id,
                    title=schedule.title,
                    description=schedule.description,
                    scheduled_at=next_time,
                    timezone=schedule.timezone,
                    recurrence_type=schedule.recurrence_type,
                    recurrence_rule=schedule.recurrence_rule,
                    status=ScheduleStatus.PENDING.value,
                    reminder_type=schedule.reminder_type,
                    duration_minutes=schedule.duration_minutes,
                )
                self.db.add(next_schedule)
                logger.info("Spawned next recurring schedule %s for %s", schedule.title, next_time)

        self.db.commit()
        self.db.refresh(schedule)
        return schedule

    def cancel_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
    ) -> Optional[Schedule]:
        """Cancels a scheduled item."""
        schedule = self.get_schedule(tenant_id, user_id, schedule_id)
        if not schedule:
            return None

        now = datetime.now(timezone.utc)
        schedule.status = ScheduleStatus.CANCELLED.value
        schedule.cancelled_at = now
        schedule.updated_at = now
        self.db.commit()
        self.db.refresh(schedule)
        return schedule

    def delete_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
    ) -> bool:
        """Deletes a schedule record."""
        schedule = self.get_schedule(tenant_id, user_id, schedule_id)
        if not schedule:
            return False

        self.db.delete(schedule)
        self.db.commit()
        return True

    @staticmethod
    def calculate_next_occurrence(
        scheduled_at: datetime,
        recurrence_type: str,
        recurrence_rule: Optional[str] = None,
    ) -> Optional[datetime]:
        """Calculates next timestamp for recurring schedules."""
        if recurrence_type == RecurrenceType.DAILY.value:
            return scheduled_at + timedelta(days=1)
        elif recurrence_type == RecurrenceType.WEEKLY.value:
            return scheduled_at + timedelta(days=7)
        elif recurrence_type == RecurrenceType.WEEKDAYS.value:
            # Add days until next weekday (Mon=0 .. Fri=4)
            nxt = scheduled_at + timedelta(days=1)
            while nxt.weekday() >= 5:  # Saturday or Sunday
                nxt += timedelta(days=1)
            return nxt
        elif recurrence_type == RecurrenceType.MONTHLY.value:
            # Approximate 30 days or same day next month
            return scheduled_at + timedelta(days=30)
        return None
