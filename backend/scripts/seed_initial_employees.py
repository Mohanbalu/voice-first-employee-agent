"""Idempotent seed script for initial HR administrator and employee accounts.

Seeds the development tenant (00000000-0000-0000-0000-000000000001) with:
- HR Administrator (username: hr@hclpass)
- 5 Initial Employees:
    SAP ID 56031439: Mohan Balu
    SAP ID 56031436: Mithesh
    SAP ID 56031957: Vishnu
    SAP ID 56031443: Siddhartha
    SAP ID 56031452: Pradeep

All passwords are securely hashed via Argon2id. Passwords are never logged.
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

# Add project root to sys.path
_current = Path(__file__).resolve()
_root = _current.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import select
from backend.app.database import get_session_factory
from backend.app.models.audit import AuditLog
from backend.app.models.employee import Employee
from backend.app.models.tenant import Tenant
from backend.app.models.user import User
from backend.app.utils.security import hash_password

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed.employees")

TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

INITIAL_EMPLOYEES = [
    {"sap_id": "56031439", "full_name": "Mohan Balu", "email": "mohan.balu@hcl.com", "department": "Engineering"},
    {"sap_id": "56031436", "full_name": "Mithesh", "email": "mithesh@hcl.com", "department": "Technology"},
    {"sap_id": "56031957", "full_name": "Vishnu", "email": "vishnu@hcl.com", "department": "Product"},
    {"sap_id": "56031443", "full_name": "Siddhartha", "email": "siddhartha@hcl.com", "department": "Engineering"},
    {"sap_id": "56031452", "full_name": "Pradeep", "email": "pradeep@hcl.com", "department": "Operations"},
]

HR_USER = {
    "username": "hr@hclpass",
    "full_name": "HR Administrator",
    "email": "hr@hcl.com",
    "department": "Human Resources",
}


def seed_initial_accounts():
    session_factory = get_session_factory()
    with session_factory() as session:
        # 1. Verify target tenant exists
        tenant = session.execute(
            select(Tenant).where(Tenant.id == TARGET_TENANT_ID)
        ).scalar_one_or_none()

        if tenant is None:
            logger.info("Creating development tenant %s", TARGET_TENANT_ID)
            tenant = Tenant(
                id=TARGET_TENANT_ID,
                name="HCL Enterprise",
                slug="dev-org",
            )
            session.add(tenant)
            session.flush()

        logger.info("Using tenant: id=%s slug=%s", tenant.id, tenant.slug)

        # 2. Seed HR User
        hr_user = session.execute(
            select(User).where(
                User.tenant_id == TARGET_TENANT_ID,
                User.username == HR_USER["username"],
            )
        ).scalar_one_or_none()

        if hr_user is None:
            logger.info("Creating HR user: %s", HR_USER["username"])
            hr_user = User(
                tenant_id=TARGET_TENANT_ID,
                username=HR_USER["username"],
                password_hash=hash_password("hclpass123"),
                role="HR",
                is_active=True,
                must_change_password=False,
            )
            session.add(hr_user)
            session.flush()

            # Optional HR employee record
            hr_emp = Employee(
                tenant_id=TARGET_TENANT_ID,
                user_id=hr_user.id,
                sap_id="56000001",
                full_name=HR_USER["full_name"],
                email=HR_USER["email"],
                department=HR_USER["department"],
                is_active=True,
            )
            session.add(hr_emp)
            session.add(
                AuditLog(
                    tenant_id=TARGET_TENANT_ID,
                    user_id=hr_user.id,
                    action="hr_user_seeded",
                    entity_type="user",
                    entity_id=str(hr_user.id),
                    details="Seeded initial HR administrator",
                )
            )
        else:
            logger.info("HR user already exists: %s", HR_USER["username"])

        # 3. Seed Initial Employees
        seeded_count = 0
        skipped_count = 0

        for emp_data in INITIAL_EMPLOYEES:
            sap_id = emp_data["sap_id"]

            # Check if employee already exists in tenant
            existing_emp = session.execute(
                select(Employee).where(
                    Employee.tenant_id == TARGET_TENANT_ID,
                    Employee.sap_id == sap_id,
                )
            ).scalar_one_or_none()

            if existing_emp is not None:
                logger.info("Employee %s (%s) already exists — skipping", sap_id, existing_emp.full_name)
                skipped_count += 1
                continue

            # Check if login user exists
            user = session.execute(
                select(User).where(
                    User.tenant_id == TARGET_TENANT_ID,
                    User.username == sap_id,
                )
            ).scalar_one_or_none()

            if user is None:
                user = User(
                    tenant_id=TARGET_TENANT_ID,
                    username=sap_id,
                    password_hash=hash_password("employee@hclpass"),
                    role="EMPLOYEE",
                    is_active=True,
                    must_change_password=True,
                )
                session.add(user)
                session.flush()

            employee = Employee(
                tenant_id=TARGET_TENANT_ID,
                user_id=user.id,
                sap_id=sap_id,
                full_name=emp_data["full_name"],
                email=emp_data["email"],
                department=emp_data["department"],
                is_active=True,
            )
            session.add(employee)
            session.add(
                AuditLog(
                    tenant_id=TARGET_TENANT_ID,
                    user_id=user.id,
                    action="employee_seeded",
                    entity_type="employee",
                    entity_id=str(employee.id),
                    details=f"Seeded employee SAP {sap_id} ({emp_data['full_name']})",
                )
            )
            seeded_count += 1
            logger.info("Seeded employee: SAP %s — %s", sap_id, emp_data["full_name"])

        session.commit()
        logger.info(
            "Seeding complete: %d employees seeded, %d skipped, HR ready.",
            seeded_count,
            skipped_count,
        )


if __name__ == "__main__":
    seed_initial_accounts()
