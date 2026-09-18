"""Migration script to ensure schedules and timesheets tables exist in database."""

import sys
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from backend.app.database import get_engine
from sqlalchemy import text

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schedules (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        employee_id UUID REFERENCES employees(id) ON DELETE SET NULL,
        title VARCHAR(255) NOT NULL,
        description TEXT,
        scheduled_at TIMESTAMPTZ NOT NULL,
        timezone VARCHAR(50) DEFAULT 'Asia/Kolkata' NOT NULL,
        recurrence_type VARCHAR(20) DEFAULT 'NONE' NOT NULL,
        recurrence_rule VARCHAR(100),
        status VARCHAR(20) DEFAULT 'PENDING' NOT NULL,
        reminder_type VARCHAR(30) DEFAULT 'NOTIFICATION' NOT NULL,
        duration_minutes INTEGER DEFAULT 15 NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
        completed_at TIMESTAMPTZ,
        cancelled_at TIMESTAMPTZ
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_schedules_tenant_id ON schedules (tenant_id);",
    "CREATE INDEX IF NOT EXISTS ix_schedules_user_id ON schedules (user_id);",
    "CREATE INDEX IF NOT EXISTS ix_schedules_scheduled_at ON schedules (scheduled_at);",
    "CREATE INDEX IF NOT EXISTS ix_schedules_status ON schedules (status);",
    "CREATE INDEX IF NOT EXISTS ix_schedules_tenant_user_status ON schedules (tenant_id, user_id, status);",
    """
    CREATE TABLE IF NOT EXISTS timesheets (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        employee_id UUID REFERENCES employees(id) ON DELETE SET NULL,
        work_date DATE NOT NULL,
        hours_worked FLOAT DEFAULT 8.0 NOT NULL,
        task_description TEXT NOT NULL,
        project_code VARCHAR(50) DEFAULT 'GENERAL' NOT NULL,
        status VARCHAR(20) DEFAULT 'SUBMITTED' NOT NULL,
        created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS ix_timesheets_tenant_id ON timesheets (tenant_id);",
    "CREATE INDEX IF NOT EXISTS ix_timesheets_user_id ON timesheets (user_id);",
    "CREATE INDEX IF NOT EXISTS ix_timesheets_work_date ON timesheets (work_date);",
]

def run_migration():
    print("Starting direct DDL migration...", flush=True)
    engine = get_engine()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        print("Connected with AUTOCOMMIT!", flush=True)
        for stmt in DDL_STATEMENTS:
            conn.execute(text(stmt))
        print("DDL executed successfully!", flush=True)
        
        # Verify
        res = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")).fetchall()
        table_names = [r[0] for r in res]
        print("Verified tables present:", [t for t in ['schedules', 'timesheets', 'tickets', 'locations'] if t in table_names], flush=True)

if __name__ == "__main__":
    run_migration()
