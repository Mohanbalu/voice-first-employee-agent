"""Scheduling & Reminder Service — Module 6.

Central business logic engine for creating, querying, updating, completing,
and recurring schedules with strict tenant and user isolation.
Supports online PostgreSQL with automatic local JSON/memory fallback when DB is offline.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from sqlalchemy import and_, desc, or_, select
from sqlalchemy.orm import Session

from backend.app.models.audit import AuditLog
from backend.app.models.employee import Employee
from backend.app.models.schedule import RecurrenceType, ReminderType, Schedule, ScheduleStatus
from backend.app.schemas.schedule import ScheduleCreate, ScheduleUpdate
from backend.app.utils.nl_time_parser import DEFAULT_TIMEZONE_NAME, get_default_tz

try:
    from backend.app.database import is_db_reachable
except ImportError:
    try:
        from app.database import is_db_reachable
    except ImportError:
        def is_db_reachable() -> bool:
            return False

logger = logging.getLogger("services.scheduling")

_LOCAL_SCHEDULES_FILE = Path(__file__).resolve().parents[3] / "data" / "schedules_local.json"
_IN_MEMORY_SCHEDULES: Dict[str, List[Schedule]] = {}


def _serialize_schedule(s: Schedule) -> Dict[str, Any]:
    return {
        "id": str(s.id),
        "tenant_id": str(s.tenant_id),
        "user_id": str(s.user_id),
        "employee_id": str(s.employee_id) if s.employee_id else None,
        "title": s.title,
        "description": s.description,
        "scheduled_at": s.scheduled_at.isoformat() if s.scheduled_at else None,
        "timezone": s.timezone,
        "recurrence_type": s.recurrence_type,
        "recurrence_rule": s.recurrence_rule,
        "status": s.status,
        "reminder_type": s.reminder_type,
        "duration_minutes": s.duration_minutes,
        "created_at": s.created_at.isoformat() if s.created_at else None,
        "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        "completed_at": s.completed_at.isoformat() if s.completed_at else None,
        "cancelled_at": s.cancelled_at.isoformat() if s.cancelled_at else None,
    }


def _deserialize_schedule(d: Dict[str, Any]) -> Schedule:
    def _parse_dt(v: Optional[str]) -> Optional[datetime]:
        if not v:
            return None
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return None

    return Schedule(
        id=uuid.UUID(d["id"]),
        tenant_id=uuid.UUID(d["tenant_id"]),
        user_id=uuid.UUID(d["user_id"]),
        employee_id=uuid.UUID(d["employee_id"]) if d.get("employee_id") else None,
        title=d.get("title") or "Reminder",
        description=d.get("description"),
        scheduled_at=_parse_dt(d.get("scheduled_at")) or datetime.now(timezone.utc),
        timezone=d.get("timezone") or DEFAULT_TIMEZONE_NAME,
        recurrence_type=d.get("recurrence_type") or "NONE",
        recurrence_rule=d.get("recurrence_rule"),
        status=d.get("status") or ScheduleStatus.PENDING.value,
        reminder_type=d.get("reminder_type") or "NOTIFICATION",
        duration_minutes=int(d.get("duration_minutes") or 15),
        created_at=_parse_dt(d.get("created_at")) or datetime.now(timezone.utc),
        updated_at=_parse_dt(d.get("updated_at")) or datetime.now(timezone.utc),
        completed_at=_parse_dt(d.get("completed_at")),
        cancelled_at=_parse_dt(d.get("cancelled_at")),
    )


def _load_local_schedules():
    global _IN_MEMORY_SCHEDULES
    if not _LOCAL_SCHEDULES_FILE.exists():
        return
    try:
        raw_text = _LOCAL_SCHEDULES_FILE.read_text(encoding="utf-8")
        raw_dict = json.loads(raw_text)
        loaded: Dict[str, List[Schedule]] = {}
        for key, items in raw_dict.items():
            loaded[key] = [_deserialize_schedule(item) for item in items]
        _IN_MEMORY_SCHEDULES = loaded
        logger.info("Loaded %d schedule groups from local JSON cache.", len(_IN_MEMORY_SCHEDULES))
    except Exception as exc:
        logger.warning("Failed to load local schedules file: %s", exc)


def _save_local_schedules():
    try:
        _LOCAL_SCHEDULES_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump_dict: Dict[str, List[Dict[str, Any]]] = {}
        for key, items in _IN_MEMORY_SCHEDULES.items():
            dump_dict[key] = [_serialize_schedule(item) for item in items]
        _LOCAL_SCHEDULES_FILE.write_text(json.dumps(dump_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save local schedules file: %s", exc)


# Initialize from local storage on module import
_load_local_schedules()


class SchedulingService:
    """Production-grade service managing workplace schedules, reminders, and recurring tasks."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def _get_in_memory_list(self, tenant_id: uuid.UUID, user_id: uuid.UUID) -> List[Schedule]:
        _load_local_schedules()
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
        tz = get_default_tz(data.timezone)
        scheduled_at = data.scheduled_at
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=tz)

        # Resolve employee_id if not provided
        if not employee_id and self.db is not None and is_db_reachable():
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

        # Always save to local store immediately
        self._get_in_memory_list(tenant_id, user_id).insert(0, schedule)
        _save_local_schedules()

        # If DB is online, persist to DB as well
        if self.db is not None and is_db_reachable():
            try:
                self.db.add(schedule)
                self.db.flush()
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
                logger.info("Created schedule %s in DB for user %s at %s", schedule.id, user_id, schedule.scheduled_at)
            except Exception as db_err:
                logger.warning("DB write failed (%s). Retained in local store.", db_err)

        return schedule

    def get_schedule(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        schedule_id: uuid.UUID,
    ) -> Optional[Schedule]:
        """Retrieves a single schedule ensuring tenant and user isolation."""
        if self.db is not None and is_db_reachable():
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
        if self.db is not None and is_db_reachable():
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
                query = query.order_by(
                    Schedule.scheduled_at.asc(),
                    Schedule.created_at.desc(),
                )
                all_items = self.db.execute(query).scalars().all()
                total = len(all_items)
                items = all_items[offset : offset + limit]
                return total, list(items)
            except Exception as db_err:
                logger.warning("DB unreachable for list_schedules (%s). Using local store.", db_err)

        items = self._get_in_memory_list(tenant_id, user_id)
        now = datetime.now(timezone.utc)
        filtered = list(items)

        if status_filter:
            filtered = [s for s in filtered if s.status == status_filter.upper()]

        if upcoming_only:
            filtered = [
                s for s in filtered
                if s.status == ScheduleStatus.PENDING.value
                and (s.scheduled_at.replace(tzinfo=timezone.utc) if s.scheduled_at.tzinfo is None else s.scheduled_at) >= now - timedelta(hours=1)
            ]

        if today_only:
            tz = get_default_tz(DEFAULT_TIMEZONE_NAME)
            local_today = datetime.now(tz).date()
            filtered = [
                s for s in filtered
                if (s.scheduled_at.astimezone(tz).date() if s.scheduled_at.tzinfo else s.scheduled_at.date()) == local_today
            ]

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
        _save_local_schedules()

        if self.db is not None and is_db_reachable():
            try:
                self.db.commit()
                self.db.refresh(schedule)
            except Exception as exc:
                logger.warning("DB commit failed on update_schedule: %s", exc)

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
        if schedule.recurrence_type and schedule.recurrence_type != RecurrenceType.NONE.value:
            next_time = self.calculate_next_occurrence(
                schedule.scheduled_at,
                schedule.recurrence_type,
                schedule.recurrence_rule,
            )
            if next_time:
                next_schedule = Schedule(
                    id=uuid.uuid4(),
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
                    created_at=now,
                    updated_at=now,
                )
                self._get_in_memory_list(tenant_id, user_id).insert(0, next_schedule)
                if self.db is not None and is_db_reachable():
                    try:
                        self.db.add(next_schedule)
                    except Exception:
                        pass
                logger.info("Spawned next recurring schedule %s for %s", schedule.title, next_time)

        _save_local_schedules()

        if self.db is not None and is_db_reachable():
            try:
                self.db.commit()
                self.db.refresh(schedule)
            except Exception as exc:
                logger.warning("DB commit failed on complete_schedule: %s", exc)

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
        _save_local_schedules()

        if self.db is not None and is_db_reachable():
            try:
                self.db.commit()
                self.db.refresh(schedule)
            except Exception as exc:
                logger.warning("DB commit failed on cancel_schedule: %s", exc)

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

        items = self._get_in_memory_list(tenant_id, user_id)
        if schedule in items:
            items.remove(schedule)
        else:
            _IN_MEMORY_SCHEDULES[f"{tenant_id}_{user_id}"] = [s for s in items if s.id != schedule_id]
        _save_local_schedules()

        if self.db is not None and is_db_reachable():
            try:
                self.db.delete(schedule)
                self.db.commit()
            except Exception as exc:
                logger.warning("DB delete failed: %s", exc)

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
            nxt = scheduled_at + timedelta(days=1)
            while nxt.weekday() >= 5:
                nxt += timedelta(days=1)
            return nxt
        elif recurrence_type == RecurrenceType.MONTHLY.value:
            return scheduled_at + timedelta(days=30)
        return None
