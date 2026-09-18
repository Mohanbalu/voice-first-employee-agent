"""Add schedules and timesheets tables.

Revision ID: 0004_schedules_timesheets
Revises: 0c84003d8d1c
Create Date: 2026-09-18 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004_schedules_timesheets"
down_revision: Union[str, None] = "0c84003d8d1c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. Create schedules table ────────────────────────────────────────────
    op.create_table(
        "schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("timezone", sa.String(length=50), server_default="Asia/Kolkata", nullable=False),
        sa.Column("recurrence_type", sa.String(length=20), server_default="NONE", nullable=False),
        sa.Column("recurrence_rule", sa.String(length=100), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="PENDING", nullable=False),
        sa.Column("reminder_type", sa.String(length=30), server_default="NOTIFICATION", nullable=False),
        sa.Column("duration_minutes", sa.Integer(), server_default="15", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_schedules_tenant_id", "schedules", ["tenant_id"])
    op.create_index("ix_schedules_user_id", "schedules", ["user_id"])
    op.create_index("ix_schedules_employee_id", "schedules", ["employee_id"])
    op.create_index("ix_schedules_scheduled_at", "schedules", ["scheduled_at"])
    op.create_index("ix_schedules_status", "schedules", ["status"])
    op.create_index("ix_schedules_tenant_user_status", "schedules", ["tenant_id", "user_id", "status"])
    op.create_index("ix_schedules_tenant_scheduled_at", "schedules", ["tenant_id", "scheduled_at"])

    # ── 2. Create timesheets table ───────────────────────────────────────────
    op.create_table(
        "timesheets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("employee_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("employees.id", ondelete="SET NULL"), nullable=True),
        sa.Column("work_date", sa.Date(), nullable=False),
        sa.Column("hours_worked", sa.Float(), server_default="8.0", nullable=False),
        sa.Column("task_description", sa.Text(), nullable=False),
        sa.Column("project_code", sa.String(length=50), server_default="GENERAL", nullable=False),
        sa.Column("status", sa.String(length=20), server_default="SUBMITTED", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_timesheets_tenant_id", "timesheets", ["tenant_id"])
    op.create_index("ix_timesheets_user_id", "timesheets", ["user_id"])
    op.create_index("ix_timesheets_employee_id", "timesheets", ["employee_id"])
    op.create_index("ix_timesheets_work_date", "timesheets", ["work_date"])
    op.create_index("ix_timesheets_status", "timesheets", ["status"])
    op.create_index("ix_timesheets_tenant_user_date", "timesheets", ["tenant_id", "user_id", "work_date"])


def downgrade() -> None:
    op.drop_table("timesheets")
    op.drop_table("schedules")
