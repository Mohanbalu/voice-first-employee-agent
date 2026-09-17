"""HR Employee Management Endpoints.

Protected routes for HR administrators to manage employee lifecycle,
provision accounts, and view workforce directories.
"""

from __future__ import annotations

import logging
import uuid
from typing import List

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
from backend.app.services.auth_service import AuthService

logger = logging.getLogger("routes.hr")

router = APIRouter(prefix="/api/hr", tags=["hr-management"])


@router.get("/employees", response_model=List[EmployeeResponse])
def list_employees(
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> List[EmployeeResponse]:
    """Lists all employees within the HR administrator's tenant.

    Strictly isolated to current_hr.tenant_id.
    """
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


@router.post("/employees", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
def create_employee(
    payload: EmployeeCreateRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Provisions a new employee and associated login account.

    Always enforces role = EMPLOYEE.
    """
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


@router.get("/employees/{employee_id}", response_model=EmployeeResponse)
def get_employee_details(
    employee_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Retrieves employee profile.

    Accessible by HR, or by the employee themselves for their own profile.
    """
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

    # If caller is not HR, verify they are retrieving their own record
    if current_user.role != "HR":
        if employee.user_id != current_user.id:
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


@router.patch("/employees/{employee_id}/status", response_model=EmployeeResponse)
def update_employee_status(
    employee_id: uuid.UUID,
    payload: EmployeeStatusUpdateRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> EmployeeResponse:
    """Activates or deactivates an employee account (HR only)."""
    target_active = payload.get_target_is_active()
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
