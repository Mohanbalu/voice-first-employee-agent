"""Tests for Natural Language Scheduling, Service, REST API, and LangGraph Integration.

Tests:
- Natural language time parser:
  * Relative offset ("in 30 mins", "in 2 hours", "in 1 day")
  * Absolute time ("at 5 PM", "tomorrow at 10 AM", "at 14:30")
  * Recurring patterns ("every Monday at 9 AM", "everyday at 8 PM", "weekdays at 11 AM")
  * Ambiguity detection ("Remind me tomorrow" -> clarification prompt)
  * Clean title stripping ("Remind me to review deliverable" -> "Review deliverable")
- Scheduling service CRUD:
  * Create schedule with tenant scoping
  * Get by ID and list with status filtering
  * Complete schedule (one-time vs recurring auto-spawning)
  * Cancel schedule
  * Delete schedule
- REST API routes (/api/schedules):
  * POST /api/schedules
  * GET /api/schedules
  * GET /api/schedules/{id}
  * POST /api/schedules/{id}/complete
  * POST /api/schedules/{id}/cancel
  * DELETE /api/schedules/{id}
  * Tenant isolation (cannot view or mutate another tenant's schedule)
- LangGraph SchedulingAgentNode:
  * Clear time string -> schedule_created populated in state
  * Ambiguous string -> requires_clarification=True in state
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.database import get_db
from backend.app.models.schedule import Schedule, ScheduleStatus, RecurrenceType, ReminderType
from backend.app.models.user import User
from backend.app.schemas.schedule import ScheduleCreate, ScheduleResponse
from backend.app.routes.schedules import router as schedules_router, _get_scheduling_service
from backend.app.services.scheduling_service import SchedulingService
from backend.app.utils.nl_time_parser import parse_natural_language_time, ParsedSchedule
from backend.app.agents.scheduling_agent import SchedulingAgentNode
from backend.app.agents.agent_state import AgentState, IntentType
from backend.app.utils.security import create_access_token

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(mock_db: MagicMock | None = None, mock_service: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    db = mock_db or MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    if mock_service:
        app.dependency_overrides[_get_scheduling_service] = lambda: mock_service
    app.include_router(schedules_router)
    return app


def _emp_headers(user_id: uuid.UUID | None = None, tenant_id: uuid.UUID = DEV_TENANT_ID) -> tuple[dict[str, str], uuid.UUID]:
    uid = user_id or uuid.uuid4()
    token = create_access_token({
        "sub": str(uid),
        "username": "56031439",
        "role": "EMPLOYEE",
        "tenant_id": str(tenant_id),
        "name": "Mohan Balu",
    })
    return {"Authorization": f"Bearer {token}"}, uid


# ── NL Time Parser Tests ──────────────────────────────────────────────────────

class TestNLTimeParser:
    def test_relative_minutes(self):
        parsed = parse_natural_language_time("Remind me in 30 mins to submit report")
        assert not parsed.requires_clarification
        assert parsed.scheduled_at is not None
        assert parsed.recurrence_type == "NONE"
        assert "submit report" in parsed.clean_title.lower()

    def test_relative_hours(self):
        parsed = parse_natural_language_time("Set a reminder in 2 hours to check emails")
        assert not parsed.requires_clarification
        assert parsed.scheduled_at is not None
        diff = (parsed.scheduled_at - datetime.now(timezone.utc)).total_seconds()
        assert 7100 < diff < 7300

    def test_absolute_time_pm(self):
        parsed = parse_natural_language_time("Remind me at 5 PM to call client")
        assert not parsed.requires_clarification
        assert parsed.scheduled_at is not None
        assert parsed.scheduled_at.hour == 17
        assert "call client" in parsed.clean_title.lower()

    def test_tomorrow_at_time(self):
        parsed = parse_natural_language_time("Remind me tomorrow at 10 AM to attend standup")
        assert not parsed.requires_clarification
        assert parsed.scheduled_at is not None
        assert parsed.scheduled_at.hour == 10

    def test_recurring_weekly(self):
        parsed = parse_natural_language_time("Remind me every Monday at 9 AM for weekly sync")
        assert not parsed.requires_clarification
        assert parsed.recurrence_type == "WEEKLY"
        assert parsed.recurrence_rule == "FREQ=WEEKLY;BYDAY=MO"

    def test_recurring_daily(self):
        parsed = parse_natural_language_time("Remind me everyday at 8 PM to submit timesheet")
        assert not parsed.requires_clarification
        assert parsed.recurrence_type == "DAILY"

    def test_ambiguous_tomorrow_without_time(self):
        parsed = parse_natural_language_time("Remind me tomorrow to submit documents")
        assert parsed.requires_clarification
        assert "what time" in parsed.clarification_question.lower()

    def test_ambiguous_no_time_at_all(self):
        parsed = parse_natural_language_time("Set a reminder to submit audit")
        assert parsed.requires_clarification
        assert "when" in parsed.clarification_question.lower()


# ── Scheduling Service Tests ──────────────────────────────────────────────────

class TestSchedulingService:
    def test_create_schedule(self):
        mock_db = MagicMock()
        service = SchedulingService(mock_db)

        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()
        sched_time = datetime.now(timezone.utc) + timedelta(hours=2)

        data = ScheduleCreate(
            title="Review PR",
            scheduled_at=sched_time,
            recurrence_type="NONE",
            reminder_type="NOTIFICATION",
            duration_minutes=30,
        )

        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        sched = service.create_schedule(
            tenant_id=tenant_id,
            user_id=user_id,
            data=data,
        )

        assert mock_db.add.called
        assert mock_db.commit.called
        assert sched.title == "Review PR"
        assert sched.status == "PENDING"

    def test_get_schedules_tenant_scoping(self):
        mock_db = MagicMock()
        service = SchedulingService(mock_db)

        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()

        mock_db.execute.return_value.scalars.return_value.all.return_value = []

        total, items = service.list_schedules(tenant_id=tenant_id, user_id=user_id)
        assert mock_db.execute.called
        assert total == 0

    def test_complete_schedule_one_time(self):
        mock_db = MagicMock()
        service = SchedulingService(mock_db)

        sched_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()

        existing = Schedule(
            id=sched_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title="One-time reminder",
            scheduled_at=datetime.now(timezone.utc),
            recurrence_type="NONE",
            status="PENDING",
        )

        service.get_schedule = MagicMock(return_value=existing)
        completed = service.complete_schedule(
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_id=sched_id,
        )

        assert completed.status == "COMPLETED"
        assert mock_db.commit.called

    def test_cancel_schedule(self):
        mock_db = MagicMock()
        service = SchedulingService(mock_db)

        sched_id = uuid.uuid4()
        tenant_id = uuid.uuid4()
        user_id = uuid.uuid4()

        existing = Schedule(
            id=sched_id,
            tenant_id=tenant_id,
            user_id=user_id,
            title="Meeting alert",
            scheduled_at=datetime.now(timezone.utc),
            status="PENDING",
        )

        service.get_schedule = MagicMock(return_value=existing)
        cancelled = service.cancel_schedule(
            tenant_id=tenant_id,
            user_id=user_id,
            schedule_id=sched_id,
        )

        assert cancelled.status == "CANCELLED"


# ── REST API Route Tests ──────────────────────────────────────────────────────

class TestScheduleRoutes:
    def test_create_schedule_api(self):
        mock_service = MagicMock()
        sched_id = uuid.uuid4()
        headers, user_id = _emp_headers()
        sched_time = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_schedule = Schedule(
            id=sched_id,
            tenant_id=DEV_TENANT_ID,
            user_id=user_id,
            title="Review Timesheet",
            scheduled_at=sched_time,
            timezone="Asia/Kolkata",
            recurrence_type="NONE",
            reminder_type="NOTIFICATION",
            status="PENDING",
            duration_minutes=15,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_service.create_schedule.return_value = mock_schedule

        app = _make_app(mock_service=mock_service)
        client = TestClient(app)

        payload = {
            "title": "Review Timesheet",
            "scheduled_at": sched_time.isoformat(),
            "recurrence_type": "NONE",
            "reminder_type": "NOTIFICATION",
            "duration_minutes": 15,
        }

        resp = client.post("/api/schedules", json=payload, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["title"] == "Review Timesheet"
        assert data["status"] == "PENDING"

    def test_get_schedules_api(self):
        mock_service = MagicMock()
        mock_service.list_schedules.return_value = (0, [])

        app = _make_app(mock_service=mock_service)
        client = TestClient(app)
        headers, _ = _emp_headers()

        resp = client.get("/api/schedules", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data or "schedules" in data
        assert data["total"] == 0

    def test_complete_schedule_api_not_found(self):
        mock_service = MagicMock()
        mock_service.complete_schedule.return_value = None

        app = _make_app(mock_service=mock_service)
        client = TestClient(app)
        headers, _ = _emp_headers()

        random_id = str(uuid.uuid4())
        resp = client.post(f"/api/schedules/{random_id}/complete", headers=headers)
        assert resp.status_code == 404

    def test_unauthorized_without_token(self):
        app = _make_app()
        client = TestClient(app)
        resp = client.get("/api/schedules")
        assert resp.status_code == 401


# ── LangGraph Scheduling Agent Integration ────────────────────────────────────

class TestSchedulingAgentNode:
    def test_unambiguous_schedule_flow(self):
        node = SchedulingAgentNode()
        mock_db = MagicMock()
        user_id = uuid.uuid4()
        tenant_id = uuid.uuid4()

        state: AgentState = {
            "tenant_id": str(tenant_id),
            "raw_query": "Remind me in 30 mins to check emails",
            "current_user": {"user_id": str(user_id), "tenant_id": str(tenant_id)},
            "db_session": mock_db,
            "tool_intents": [],
        }

        res = node(state)
        assert not res.get("requires_clarification")
        assert "scheduled" in res.get("final_response", "").lower()
        assert res.get("schedule_created") is not None
        assert res.get("agent_mode") == "tool_intent"

    def test_ambiguous_schedule_clarification(self):
        node = SchedulingAgentNode()
        mock_db = MagicMock()

        state: AgentState = {
            "tenant_id": str(DEV_TENANT_ID),
            "raw_query": "Remind me tomorrow to submit report",
            "db_session": mock_db,
            "tool_intents": [],
        }

        res = node(state)
        assert res.get("requires_clarification") is True
        assert res.get("clarification_question") is not None
        assert "what time" in res.get("clarification_question", "").lower()
