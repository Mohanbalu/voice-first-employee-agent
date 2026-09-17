"""Tests for HR Employee Provisioning and Lifecycle Management Routes.

Validates:
- RBAC: Only users with role='HR' can access /api/hr routes; role='EMPLOYEE' gets 403.
- SAP ID validation: exactly 8 digits starting with 560 (^560\\d{5}$).
- Duplicate SAP ID rejection with 409 Conflict.
- Employee deactivation and activation.
- Multi-tenant isolation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.database import get_db
from backend.app.models.employee import Employee
from backend.app.models.user import User
from backend.app.routes.hr import router as hr_router
from backend.app.utils.security import create_access_token

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(mock_db: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    db = mock_db or MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    app.include_router(hr_router)
    return app


def _hr_headers(tenant_id: uuid.UUID = DEV_TENANT_ID) -> dict[str, str]:
    user_id = uuid.uuid4()
    token = create_access_token({
        "sub": str(user_id),
        "username": "hr@hclpass",
        "role": "HR",
        "tenant_id": str(tenant_id),
        "name": "HR Administrator",
    })
    return {"Authorization": f"Bearer {token}"}


def _emp_headers(tenant_id: uuid.UUID = DEV_TENANT_ID) -> dict[str, str]:
    user_id = uuid.uuid4()
    token = create_access_token({
        "sub": str(user_id),
        "username": "56031439",
        "role": "EMPLOYEE",
        "tenant_id": str(tenant_id),
        "name": "Mohan Balu",
    })
    return {"Authorization": f"Bearer {token}"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHREmployeeRoutes:
    """HR Employee Management API test cases."""

    def test_hr_list_employees_success(self):
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        emp1 = Employee(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            sap_id="56031439",
            full_name="Mohan Balu",
            email="mohan.balu@hcl.com",
            department="Engineering",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = hr_user
        mock_exec.scalars.return_value.all.return_value = [emp1]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get("/api/hr/employees", headers=_hr_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["sap_id"] == "56031439"
        assert data[0]["full_name"] == "Mohan Balu"

    def test_employee_list_employees_forbidden(self):
        mock_db = MagicMock()
        emp_user = User(
            id=uuid.uuid4(),
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

        resp = client.get("/api/hr/employees", headers=_emp_headers())
        assert resp.status_code == 403
        assert "HR administrator privileges required" in resp.json()["detail"]

    def test_hr_create_employee_success(self):
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.side_effect = [hr_user, None, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        payload = {
            "sap_id": "56031999",
            "full_name": "Kavya Ramesh",
            "email": "kavya.ramesh@hcl.com",
            "department": "IT Operations",
            "password": "SecurePassword123!",
        }
        resp = client.post("/api/hr/employees", json=payload, headers=_hr_headers())
        assert resp.status_code == 201
        data = resp.json()
        assert data["sap_id"] == "56031999"
        assert data["full_name"] == "Kavya Ramesh"
        assert data["email"] == "kavya.ramesh@hcl.com"
        assert data["department"] == "IT Operations"
        assert data["is_active"] is True

    def test_create_employee_invalid_sap_id_format(self):
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
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

        # Invalid 1: Less than 8 digits
        resp = client.post(
            "/api/hr/employees",
            json={
                "sap_id": "56031",
                "full_name": "Test User",
                "email": "test@hcl.com",
                "department": "Engineering",
            },
            headers=_hr_headers(),
        )
        assert resp.status_code == 422

        # Invalid 2: Doesn't start with 560
        resp = client.post(
            "/api/hr/employees",
            json={
                "sap_id": "12345678",
                "full_name": "Test User",
                "email": "test@hcl.com",
                "department": "Engineering",
            },
            headers=_hr_headers(),
        )
        assert resp.status_code == 422

        # Invalid 3: Non-numeric
        resp = client.post(
            "/api/hr/employees",
            json={
                "sap_id": "560ABC99",
                "full_name": "Test User",
                "email": "test@hcl.com",
                "department": "Engineering",
            },
            headers=_hr_headers(),
        )
        assert resp.status_code == 422

    def test_create_employee_duplicate_sap_id_conflict(self):
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        existing_emp = Employee(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            sap_id="56031439",
            full_name="Existing User",
            email="existing@hcl.com",
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.side_effect = [hr_user, existing_emp]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        payload = {
            "sap_id": "56031439",
            "full_name": "Duplicate User",
            "email": "dup@hcl.com",
        }
        resp = client.post("/api/hr/employees", json=payload, headers=_hr_headers())
        assert resp.status_code == 409
        assert "already exists" in resp.json()["detail"]

    def test_employee_cannot_create_employee_forbidden(self):
        mock_db = MagicMock()
        emp_user = User(
            id=uuid.uuid4(),
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

        payload = {
            "sap_id": "56031999",
            "full_name": "Attacker",
            "email": "attacker@hcl.com",
        }
        resp = client.post("/api/hr/employees", json=payload, headers=_emp_headers())
        assert resp.status_code == 403

    def test_hr_update_employee_status_deactivate(self):
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        emp_id = uuid.uuid4()
        emp_user_id = uuid.uuid4()
        target_emp = Employee(
            id=emp_id,
            tenant_id=DEV_TENANT_ID,
            user_id=emp_user_id,
            sap_id="56031439",
            full_name="Mohan Balu",
            email="mohan.balu@hcl.com",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )
        target_user = User(
            id=emp_user_id,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.side_effect = [hr_user, target_emp, target_user]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.patch(
            f"/api/hr/employees/{emp_id}/status",
            json={"is_active": False},
            headers=_hr_headers(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_active"] is False

    def test_employee_update_status_forbidden(self):
        mock_db = MagicMock()
        emp_user = User(
            id=uuid.uuid4(),
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

        some_id = uuid.uuid4()
        resp = client.patch(
            f"/api/hr/employees/{some_id}/status",
            json={"is_active": False},
            headers=_emp_headers(),
        )
        assert resp.status_code == 403

    def test_hr_employee_tenant_isolation(self):
        """Validates that HR user cannot access employees of another tenant."""
        mock_db = MagicMock()
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash="hash",
            role="HR",
            is_active=True,
        )
        other_tenant_emp_id = uuid.uuid4()

        mock_exec = MagicMock()
        # get_current_user -> hr_user, find employee in tenant -> None (filtered by tenant_id)
        mock_exec.scalar_one_or_none.side_effect = [hr_user, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get(
            f"/api/hr/employees/{other_tenant_emp_id}",
            headers=_hr_headers(DEV_TENANT_ID),
        )
        assert resp.status_code == 404

