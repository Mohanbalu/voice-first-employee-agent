"""HR Employee Management Endpoints.

Protected routes for HR administrators to manage employee lifecycle,
provision accounts, and view workforce directories.
Supports online PostgreSQL with automatic local fallback when DB is unreachable.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import get_current_user, require_hr
from backend.app.database import get_db
from backend.app.models.employee import Employee
from backend.app.models.user import User
from backend.app.schemas.auth import (
    EmployeeCreateRequest,
    EmployeeResponse,
    EmployeeStatusUpdateRequest,
)
from backend.app.services.auth_service import AuthService, LOCAL_DEV_ACCOUNTS

try:
    from backend.app.database import is_db_reachable
except ImportError:
    try:
        from app.database import is_db_reachable
    except ImportError:
        def is_db_reachable() -> bool:
            return False

logger = logging.getLogger("routes.hr")
router = APIRouter(prefix="/api/hr", tags=["hr-management"])

_LOCAL_EMPLOYEES_FILE = Path(__file__).resolve().parents[3] / "data" / "employees_local.json"


def _get_fallback_employees(tenant_id: uuid.UUID) -> List[EmployeeResponse]:
    """Generates employee list from local dev accounts and local JSON storage."""
    now = datetime.now(timezone.utc)
    res: List[EmployeeResponse] = []

    # 1. Base accounts from LOCAL_DEV_ACCOUNTS
    for sap_id, acc in LOCAL_DEV_ACCOUNTS.items():
        if acc.get("role") == "HR":
            continue
        res.append(
            EmployeeResponse(
                id=acc["id"],
                tenant_id=acc.get("tenant_id", tenant_id),
                user_id=acc["id"],
                sap_id=acc["sap_id"],
                full_name=acc["name"],
                email=f"{acc['name'].lower().replace(' ', '.')}@hcl.com",
                department="Engineering",
                role=acc["role"],
                is_active=acc.get("is_active", True),
                created_at=now,
                last_login=now,
            )
        )

    # 2. Add extra provisioned employees from local JSON file
    if _LOCAL_EMPLOYEES_FILE.exists():
        try:
            extra = json.loads(_LOCAL_EMPLOYEES_FILE.read_text(encoding="utf-8"))
            for item in extra:
                # Convert strings to UUID / datetime
                item_id = uuid.UUID(item["id"])
                # Avoid duplicates
                if not any(e.id == item_id or e.sap_id == item["sap_id"] for e in res):
                    res.append(
                        EmployeeResponse(
                            id=item_id,
                            tenant_id=uuid.UUID(item["tenant_id"]),
                            user_id=uuid.UUID(item["user_id"]),
                            sap_id=item["sap_id"],
                            full_name=item["full_name"],
                            email=item["email"],
                            department=item.get("department", "Engineering"),
                            role=item.get("role", "EMPLOYEE"),
                            is_active=item.get("is_active", True),
                            created_at=datetime.fromisoformat(item["created_at"]) if item.get("created_at") else now,
                            last_login=None,
                        )
                    )
        except Exception as exc:
            logger.warning("Error reading local employees file: %s", exc)

    return res


def _save_local_employee(emp: EmployeeResponse):
    """Saves provisioned employee to local JSON file."""
    try:
        _LOCAL_EMPLOYEES_FILE.parent.mkdir(parents=True, exist_ok=True)
        items = []
        if _LOCAL_EMPLOYEES_FILE.exists():
            try:
                items = json.loads(_LOCAL_EMPLOYEES_FILE.read_text(encoding="utf-8"))
            except Exception:
                items = []

        # Check existing
        items = [i for i in items if i.get("sap_id") != emp.sap_id]
        items.append({
            "id": str(emp.id),
            "tenant_id": str(emp.tenant_id),
            "user_id": str(emp.user_id),
            "sap_id": emp.sap_id,
            "full_name": emp.full_name,
            "email": emp.email,
            "department": emp.department,
            "role": emp.role,
            "is_active": emp.is_active,
            "created_at": emp.created_at.isoformat() if emp.created_at else None,
        })
        _LOCAL_EMPLOYEES_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Error saving local employee: %s", exc)


@router.get("/employees", response_model=List[EmployeeResponse])
def list_employees(
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> List[EmployeeResponse]:
    """Lists all employees within the HR administrator's tenant."""
    if not is_db_reachable():
        return _get_fallback_employees(current_hr.tenant_id)

    try:
        stmt = (
            select(Employee)
            .where(Employee.tenant_id == current_hr.tenant_id)
            .order_by(Employee.created_at.desc())
        )
        employees = db.execute(stmt).scalars().all()

        result = []
        for emp in employees:
            last_login = None
            role = "EMPLOYEE"
            if emp.user:
                last_login = emp.user.last_login
                role = emp.user.role

            result.append(
                EmployeeResponse(
                    id=emp.id,
                    tenant_id=emp.tenant_id,
                    user_id=emp.user_id,
                    sap_id=emp.sap_id,
                    full_name=emp.full_name,
                    email=emp.email,
                    department=emp.department,
                    role=role,
                    is_active=emp.is_active,
                    created_at=emp.created_at,
                    last_login=last_login,
                )
            )
        return result
    except Exception as exc:
        logger.warning("DB query failed in list_employees: %s. Using local fallback.", exc)
        return _get_fallback_employees(current_hr.tenant_id)


@router.post("/employees", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
def create_employee(
    payload: EmployeeCreateRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Provisions a new employee and associated login account."""
    if not is_db_reachable():
        # Check uniqueness against local store
        existing = _get_fallback_employees(current_hr.tenant_id)
        if any(e.sap_id == payload.sap_id for e in existing):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Employee with SAP ID {payload.sap_id} already exists in this organization",
            )

        new_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        emp_resp = EmployeeResponse(
            id=new_id,
            tenant_id=current_hr.tenant_id,
            user_id=new_id,
            sap_id=payload.sap_id,
            full_name=payload.full_name,
            email=payload.email,
            department=payload.department,
            role="EMPLOYEE",
            is_active=True,
            created_at=now,
            last_login=None,
        )
        _save_local_employee(emp_resp)

        # Register credentials in LOCAL_DEV_ACCOUNTS
        LOCAL_DEV_ACCOUNTS[payload.sap_id] = {
            "id": new_id,
            "tenant_id": current_hr.tenant_id,
            "username": payload.sap_id,
            "role": "EMPLOYEE",
            "name": payload.full_name,
            "sap_id": payload.sap_id,
            "passwords": ["hclpass123", "employee@hclpass"],
        }
        return emp_resp

    try:
        employee = AuthService.create_employee(
            db=db,
            tenant_id=current_hr.tenant_id,
            payload=payload,
            actor=current_hr,
        )
        return EmployeeResponse(
            id=employee.id,
            tenant_id=employee.tenant_id,
            user_id=employee.user_id,
            sap_id=employee.sap_id,
            full_name=employee.full_name,
            email=employee.email,
            department=employee.department,
            role="EMPLOYEE",
            is_active=employee.is_active,
            created_at=employee.created_at,
            last_login=None,
        )
    except Exception as exc:
        logger.warning("DB provision failed (%s). Storing locally.", exc)
        new_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        emp_resp = EmployeeResponse(
            id=new_id,
            tenant_id=current_hr.tenant_id,
            user_id=new_id,
            sap_id=payload.sap_id,
            full_name=payload.full_name,
            email=payload.email,
            department=payload.department,
            role="EMPLOYEE",
            is_active=True,
            created_at=now,
            last_login=None,
        )
        _save_local_employee(emp_resp)
        return emp_resp


@router.get("/employees/{employee_id}", response_model=EmployeeResponse)
def get_employee_details(
    employee_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Retrieves employee profile."""
    if not is_db_reachable():
        emp = next((e for e in _get_fallback_employees(current_user.tenant_id) if e.id == employee_id), None)
        if emp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found in organization",
            )
        return emp

    try:
        stmt = select(Employee).where(
            Employee.id == employee_id,
            Employee.tenant_id == current_user.tenant_id,
        )
        employee = db.execute(stmt).scalar_one_or_none()

        if employee is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found in organization",
            )

        if current_user.role != "HR" and employee.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: You can only view your own profile",
            )

        return EmployeeResponse(
            id=employee.id,
            tenant_id=employee.tenant_id,
            user_id=employee.user_id,
            sap_id=employee.sap_id,
            full_name=employee.full_name,
            email=employee.email,
            department=employee.department,
            role=employee.user.role if employee.user else "EMPLOYEE",
            is_active=employee.is_active,
            created_at=employee.created_at,
            last_login=employee.user.last_login if employee.user else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("DB query failed in get_employee_details: %s", exc)
        emp = next((e for e in _get_fallback_employees(current_user.tenant_id) if e.id == employee_id), None)
        if emp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found in organization",
            )
        return emp


@router.patch("/employees/{employee_id}/status", response_model=EmployeeResponse)
def update_employee_status(
    employee_id: uuid.UUID,
    payload: EmployeeStatusUpdateRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Activates or deactivates an employee account (HR only)."""
    target_active = payload.get_target_is_active()

    if not is_db_reachable():
        emp = next((e for e in _get_fallback_employees(current_hr.tenant_id) if e.id == employee_id), None)
        if emp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found in organization",
            )
        emp.is_active = target_active
        _save_local_employee(emp)
        return emp

    try:
        employee = AuthService.set_employee_status(
            db=db,
            tenant_id=current_hr.tenant_id,
            employee_id=employee_id,
            is_active=target_active,
            actor=current_hr,
        )
        return EmployeeResponse(
            id=employee.id,
            tenant_id=employee.tenant_id,
            user_id=employee.user_id,
            sap_id=employee.sap_id,
            full_name=employee.full_name,
            email=employee.email,
            department=employee.department,
            role=employee.user.role if employee.user else "EMPLOYEE",
            is_active=employee.is_active,
            created_at=employee.created_at,
            last_login=employee.user.last_login if employee.user else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("DB update failed in update_employee_status: %s", exc)
        emp = next((e for e in _get_fallback_employees(current_hr.tenant_id) if e.id == employee_id), None)
        if emp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found in organization",
            )
        emp.is_active = target_active
        _save_local_employee(emp)
        return emp
