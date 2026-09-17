"""Tests for Authentication & Security Module.

Tests password hashing (Argon2id), JWT generation/validation,
and /api/auth/login + /api/auth/me routes with mocked database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.models.employee import Employee
from backend.app.routes.auth import router as auth_router
from backend.app.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
)

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


# ── Fixtures & Helpers ─────────────────────────────────────────────────────────

def _make_app(mock_db: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    db = mock_db or MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    app.include_router(auth_router)
    return app


# ── Password & JWT Security Unit Tests ─────────────────────────────────────────

class TestPasswordAndTokenSecurity:
    """Argon2id and PyJWT verification."""

    def test_hash_password_produces_argon2(self):
        pw = "TestSecretPassword123!"
        hashed = hash_password(pw)
        assert hashed.startswith("$argon2id$")
        assert hashed != pw

    def test_verify_password_correct(self):
        pw = "CorrectHCLPassword@2026"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_verify_password_incorrect(self):
        pw = "CorrectPassword"
        hashed = hash_password(pw)
        assert verify_password("WrongPassword", hashed) is False

    def test_jwt_create_and_decode(self):
        user_id = uuid.uuid4()
        token = create_access_token(
            data={
                "sub": str(user_id),
                "tenant_id": str(DEV_TENANT_ID),
                "username": "56031439",
                "role": "EMPLOYEE",
            }
        )
        claims = decode_access_token(token)
        assert claims["sub"] == str(user_id)
        assert claims["tenant_id"] == str(DEV_TENANT_ID)
        assert claims["username"] == "56031439"
        assert claims["role"] == "EMPLOYEE"

    def test_jwt_invalid_token_rejected(self):
        with pytest.raises(Exception):
            decode_access_token("invalid.jwt.token.string")


# ── Auth API Route Tests ───────────────────────────────────────────────────────

class TestAuthRoutes:
    """POST /api/auth/login and GET /api/auth/me tests."""

    def test_hr_login_success(self):
        mock_db = MagicMock()
        pw_hash = hash_password("hclpass123")
        hr_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="hr@hclpass",
            password_hash=pw_hash,
            role="HR",
            is_active=True,
            must_change_password=False,
        )

        mock_exec_result = MagicMock()
        # First call is user lookup -> hr_user, second call is employee profile -> None
        mock_exec_result.scalar_one_or_none.side_effect = [hr_user, None]
        mock_db.execute.return_value = mock_exec_result

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            "/api/auth/login",
            json={"username": "hr@hclpass", "password": "hclpass123"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["user"]["username"] == "hr@hclpass"
        assert data["user"]["role"] == "HR"

    def test_employee_login_success(self):
        mock_db = MagicMock()
        pw_hash = hash_password("HclEmp@56031439")
        emp_user_id = uuid.uuid4()
        emp_user = User(
            id=emp_user_id,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash=pw_hash,
            role="EMPLOYEE",
            is_active=True,
            must_change_password=False,
        )
        emp_profile = Employee(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            user_id=emp_user_id,
            sap_id="56031439",
            full_name="Mohan Balu",
            email="mohan.balu@hcl.com",
            department="Engineering",
            is_active=True,
        )

        mock_exec_result = MagicMock()
        # In authenticate_user: User lookup returns emp_user
        # In issue_token_response: Employee lookup returns emp_profile
        mock_exec_result.scalar_one_or_none.side_effect = [emp_user, emp_profile]
        mock_db.execute.return_value = mock_exec_result

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            "/api/auth/login",
            json={"username": "56031439", "password": "HclEmp@56031439"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user"]["username"] == "56031439"
        assert data["user"]["role"] == "EMPLOYEE"
        assert data["user"]["sap_id"] == "56031439"
        assert data["user"]["name"] == "Mohan Balu"

    def test_login_wrong_password_fails(self):
        mock_db = MagicMock()
        pw_hash = hash_password("correct_password")
        user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash=pw_hash,
            role="EMPLOYEE",
            is_active=True,
        )
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            "/api/auth/login",
            json={"username": "56031439", "password": "wrong_password"},
        )
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    def test_login_user_not_found(self):
        mock_db = MagicMock()
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            "/api/auth/login",
            json={"username": "nonexistent", "password": "any_password"},
        )
        assert resp.status_code == 401
        assert "Invalid credentials" in resp.json()["detail"]

    def test_login_inactive_user_rejected(self):
        mock_db = MagicMock()
        pw_hash = hash_password("valid_password")
        inactive_user = User(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash=pw_hash,
            role="EMPLOYEE",
            is_active=False,
        )
        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.return_value = inactive_user
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.post(
            "/api/auth/login",
            json={"username": "56031439", "password": "valid_password"},
        )
        assert resp.status_code == 401
        assert "Account has been deactivated" in resp.json()["detail"]

    def test_get_me_authenticated(self):
        mock_db = MagicMock()
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
            must_change_password=False,
        )
        emp_profile = Employee(
            id=uuid.uuid4(),
            tenant_id=DEV_TENANT_ID,
            user_id=user_id,
            sap_id="56031439",
            full_name="Mohan Balu",
            email="mohan.balu@hcl.com",
            department="Engineering",
            is_active=True,
        )

        mock_exec = MagicMock()
        # get_current_user loads user, then route loads employee profile
        mock_exec.scalar_one_or_none.side_effect = [user, emp_profile]
        mock_db.execute.return_value = mock_exec

        token = create_access_token(
            data={
                "sub": str(user_id),
                "tenant_id": str(DEV_TENANT_ID),
                "username": "56031439",
                "role": "EMPLOYEE",
            }
        )

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "56031439"
        assert data["role"] == "EMPLOYEE"
        assert data["name"] == "Mohan Balu"

    def test_get_me_missing_token_unauthorized(self):
        app = _make_app()
        client = TestClient(app)

        resp = client.get("/api/auth/me")
        assert resp.status_code == 401
