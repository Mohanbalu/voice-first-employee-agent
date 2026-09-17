"""Authentication and Employee Provisioning Pydantic Schemas."""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Login & Token Schemas ─────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Payload for user login via username/email or SAP ID."""

    username: str = Field(..., min_length=1, max_length=100, description="Email for HR, SAP ID for Employees")
    password: str = Field(..., min_length=1, max_length=128, description="User password")


class UserResponse(BaseModel):
    """Safe user information returned upon login or profile retrieval."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    role: str
    tenant_id: uuid.UUID
    name: Optional[str] = None
    sap_id: Optional[str] = None
    is_active: bool
    must_change_password: bool


class TokenResponse(BaseModel):
    """JWT bearer token response."""

    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ── Employee Provisioning Schemas ─────────────────────────────────────────────

SAP_ID_REGEX = re.compile(r"^560\d{5}$")


class EmployeeCreateRequest(BaseModel):
    """Payload for HR provisioning a new employee."""

    sap_id: str = Field(..., description="Exactly 8 digits, starting with 560")
    full_name: str = Field(..., min_length=2, max_length=255, description="Full legal name of the employee")
    email: Optional[str] = Field(None, max_length=255, description="Corporate email address")
    department: Optional[str] = Field(None, max_length=100, description="Assigned department")

    @field_validator("sap_id")
    @classmethod
    def validate_sap_id(cls, v: str) -> str:
        clean = v.strip()
        if not SAP_ID_REGEX.match(clean):
            raise ValueError(
                "SAP ID must be exactly 8 numeric digits starting with 560 (e.g., 56031439)"
            )
        return clean

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) < 2:
            raise ValueError("Full name must contain at least 2 characters")
        return clean


class EmployeeResponse(BaseModel):
    """Safe employee details returned to HR or profile view."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tenant_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    sap_id: str
    full_name: str
    email: Optional[str] = None
    department: Optional[str] = None
    role: str = "EMPLOYEE"
    is_active: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None


class EmployeeStatusUpdateRequest(BaseModel):
    """Payload for deactivating or activating an employee."""

    is_active: Optional[bool] = None
    status: Optional[str] = None  # e.g. "active", "inactive"

    def get_target_is_active(self) -> bool:
        if self.is_active is not None:
            return self.is_active
        if self.status is not None:
            return self.status.strip().lower() in ("active", "true", "enabled")
        raise ValueError("Either is_active or status must be provided")
