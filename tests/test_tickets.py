"""Tests for Ticket Lifecycle, RBAC, and Sequential Numbering.

Tests:
- Ticket creation by authenticated employee.
- Ticket number format HCL-{CATEGORY}-{SEQUENCE:06d}.
- Employee viewing own tickets (/api/tickets/me).
- HR viewing all organizational tickets (/api/tickets).
- Non-HR employee forbidden from viewing all tickets or updating status.
- HR updating ticket status (OPEN -> IN_PROGRESS -> RESOLVED).
- Tenant isolation (cannot view tickets belonging to another tenant).
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.database import get_db
from backend.app.models.ticket import Ticket
from backend.app.models.user import User
from backend.app.routes.tickets import router as tickets_router
from backend.app.utils.security import create_access_token

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(mock_db: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    db = mock_db or MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    app.include_router(tickets_router)
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


def _hr_headers(user_id: uuid.UUID | None = None, tenant_id: uuid.UUID = DEV_TENANT_ID) -> tuple[dict[str, str], uuid.UUID]:
    uid = user_id or uuid.uuid4()
    token = create_access_token({
        "sub": str(uid),
        "username": "hr@hclpass",
        "role": "HR",
        "tenant_id": str(tenant_id),
        "name": "HR Administrator",
    })
    return {"Authorization": f"Bearer {token}"}, uid


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTicketRoutes:
    """Ticket Management API test cases."""

    def test_employee_create_ticket_success(self):
        mock_db = MagicMock()
        headers, emp_uid = _emp_headers()
        emp_user = User(
            id=emp_uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )

        mock_exec = MagicMock()
        # 1. get_current_user -> emp_user
        # 2. _generate_ticket_number max sequence lookup -> 0
        # 3. _generate_ticket_number collision check -> None
        # 4. to_response user lookup -> emp_user
        # 5. to_response employee lookup -> None
        mock_exec.scalar_one_or_none.side_effect = [emp_user, 0, None, emp_user, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        payload = {
            "category": "IT",
            "subject": "External monitor not working in Tower 1 Floor 3",
            "description": "Monitor power LED blinks amber, no display via HDMI.",
            "priority": "HIGH",
        }
        resp = client.post("/api/tickets", json=payload, headers=headers)
        assert resp.status_code == 201
        data = resp.json()
        assert re.match(r"^HCL-IT-\d{6}$", data["ticket_number"])
        assert data["subject"] == payload["subject"]
        assert data["category"] == "IT"
        assert data["priority"] == "HIGH"
        assert data["status"] == "OPEN"

    def test_employee_list_own_tickets(self):
        mock_db = MagicMock()
        headers, emp_uid = _emp_headers()
        emp_user = User(
            id=emp_uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )
        ticket = Ticket(
            id=uuid.uuid4(),
            ticket_number="HCL-IT-000001",
            tenant_id=DEV_TENANT_ID,
            created_by=emp_uid,
            category="IT",
            subject="Monitor broken",
            description="Details",
            priority="HIGH",
            status="OPEN",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = emp_user
        mock_exec.scalars.return_value.all.return_value = [ticket]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get("/api/tickets/me", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["tickets"][0]["ticket_number"] == "HCL-IT-000001"

    def test_hr_list_all_tickets_success(self):
        mock_db = MagicMock()
        headers, hr_uid = _hr_headers()
        hr_user = User(
            id=hr_uid,
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        ticket1 = Ticket(
            id=uuid.uuid4(),
            ticket_number="HCL-IT-000001",
            tenant_id=DEV_TENANT_ID,
            created_by=uuid.uuid4(),
            category="IT",
            subject="Keyboard issue",
            description="Details",
            priority="LOW",
            status="OPEN",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        ticket2 = Ticket(
            id=uuid.uuid4(),
            ticket_number="HCL-HR-000001",
            tenant_id=DEV_TENANT_ID,
            created_by=uuid.uuid4(),
            category="HR",
            subject="Leave query",
            description="Details",
            priority="MEDIUM",
            status="IN_PROGRESS",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = hr_user
        mock_exec.scalars.return_value.all.return_value = [ticket1, ticket2]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get("/api/tickets", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_employee_list_all_tickets_forbidden(self):
        mock_db = MagicMock()
        headers, emp_uid = _emp_headers()
        emp_user = User(
            id=emp_uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = emp_user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get("/api/tickets", headers=headers)
        assert resp.status_code == 403

    def test_hr_update_ticket_status(self):
        mock_db = MagicMock()
        headers, hr_uid = _hr_headers()
        hr_user = User(
            id=hr_uid,
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        ticket_id = uuid.uuid4()
        ticket = Ticket(
            id=ticket_id,
            ticket_number="HCL-IT-000001",
            tenant_id=DEV_TENANT_ID,
            created_by=uuid.uuid4(),
            category="IT",
            subject="Issue",
            description="Desc",
            priority="HIGH",
            status="OPEN",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_exec = MagicMock()
        # 1. require_hr -> hr_user
        # 2. get_ticket -> ticket
        # 3. to_response user lookup -> None
        # 4. to_response employee lookup -> None
        mock_exec.scalar_one_or_none.side_effect = [hr_user, ticket, None, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.patch(
            f"/api/tickets/{ticket_id}/status",
            json={"status": "RESOLVED", "resolution_notes": "Replaced HDMI cable"},
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "RESOLVED"
        assert data["resolution_notes"] == "Replaced HDMI cable"

    def test_employee_update_ticket_status_forbidden(self):
        mock_db = MagicMock()
        headers, emp_uid = _emp_headers()
        emp_user = User(
            id=emp_uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = emp_user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.patch(
            f"/api/tickets/{uuid.uuid4()}/status",
            json={"status": "CLOSED"},
            headers=headers,
        )
        assert resp.status_code == 403

    def test_invalid_ticket_status_rejected(self):
        mock_db = MagicMock()
        headers, hr_uid = _hr_headers()
        hr_user = User(
            id=hr_uid,
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = hr_user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.patch(
            f"/api/tickets/{uuid.uuid4()}/status",
            json={"status": "SUPER_RESOLVED"},
            headers=headers,
        )
        assert resp.status_code == 422

    def test_ticket_tenant_isolation(self):
        mock_db = MagicMock()
        headers, hr_uid = _hr_headers()
        hr_user = User(
            id=hr_uid,
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )

        mock_exec = MagicMock()
        # 1. get_current_user -> hr_user
        # 2. get_ticket -> None (filtered by DEV_TENANT_ID)
        mock_exec.scalar_one_or_none.side_effect = [hr_user, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get(f"/api/tickets/{uuid.uuid4()}", headers=headers)
        assert resp.status_code == 404

    def test_ticket_sequential_numbering_collision_safe(self):
        """Tests that _generate_ticket_number advances sequence when candidate collides."""
        from backend.app.services.ticket_service import TicketService

        mock_db = MagicMock()
        # Count returns 2 -> candidate is HCL-IT-000003
        # First exists query returns collision (existing ID)
        # Second exists query returns None (free candidate)
        mock_db.execute.return_value.scalar.return_value = 2
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [uuid.uuid4(), None]

        ticket_number = TicketService._generate_ticket_number(
            mock_db,
            DEV_TENANT_ID,
            "IT",
        )
        assert ticket_number == "HCL-IT-000004"

    def test_hr_answer_ticket_success(self):
        """HR provides an official answer to an open ticket, resolving it."""
        mock_db = MagicMock()
        headers, hr_uid = _hr_headers()
        hr_user = User(
            id=hr_uid,
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )

        ticket_id = uuid.uuid4()
        existing_ticket = Ticket(
            id=ticket_id,
            tenant_id=DEV_TENANT_ID,
            created_by=uuid.uuid4(),
            ticket_number="HCL-HR-000010",
            category="HR",
            priority="MEDIUM",
            status="OPEN",
            subject="How to request parental leave?",
            description="Where do I find the application form?",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        mock_exec = MagicMock()
        # 1. get_current_user -> hr_user
        # 2. provide_answer find ticket -> existing_ticket
        mock_exec.scalar_one_or_none.side_effect = [hr_user, existing_ticket]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            f"/api/tickets/{ticket_id}/answer",
            json={
                "answer": "Submit form HR-402 on the Employee Portal under Leave & Benefits.",
                "ingest_to_kb": False,
            },
            headers=headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "RESOLVED"
        assert data["hr_answer"] == "Submit form HR-402 on the Employee Portal under Leave & Benefits."

    def test_employee_cannot_answer_ticket(self):
        """Non-HR employee cannot answer tickets (returns 403 Forbidden)."""
        mock_db = MagicMock()
        headers, emp_uid = _emp_headers()
        emp_user = User(
            id=emp_uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = emp_user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            f"/api/tickets/{uuid.uuid4()}/answer",
            json={"answer": "My own answer", "ingest_to_kb": False},
            headers=headers,
        )
        assert resp.status_code == 403

