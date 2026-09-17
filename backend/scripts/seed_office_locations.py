"""Idempotent seed script for official HCL campus office locations and geo-coordinates.

Seeds the development tenant (00000000-0000-0000-0000-000000000001) with:
1. Entrance Gate (lat: 16.534052, lon: 80.792049, type: CAMPUS_ENTRANCE)
2. SDC (lat: 16.533700, lon: 80.791341, type: BUILDING)
3. Tower 1 (lat: 16.534497, lon: 80.790976, type: BUILDING)
4. Tower 2 (lat: 16.534263, lon: 80.790065, type: BUILDING)

Idempotent: safe to run multiple times without creating duplicate locations.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Add project root to sys.path
_current = Path(__file__).resolve()
_root = _current.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from sqlalchemy import select
from backend.app.database import get_session_factory
from backend.app.models.audit import AuditLog
from backend.app.models.location import Location
from backend.app.models.tenant import Tenant

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed.locations")

TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

OFFICIAL_LOCATIONS = [
    {
        "name": "Entrance Gate",
        "location_type": "CAMPUS_ENTRANCE",
        "latitude": Decimal("16.534052"),
        "longitude": Decimal("80.792049"),
        "description": "Main HCL campus entrance gate.",
    },
    {
        "name": "SDC",
        "location_type": "BUILDING",
        "latitude": Decimal("16.533700"),
        "longitude": Decimal("80.791341"),
        "description": (
            "SDC building containing Techbees classrooms, reception, "
            "cafeteria, IT team, seminar halls, and laptop/technical "
            "support teams according to the office knowledge base."
        ),
    },
    {
        "name": "Tower 1",
        "location_type": "BUILDING",
        "latitude": Decimal("16.534497"),
        "longitude": Decimal("80.790976"),
        "description": (
            "Tower 1 containing trainee areas, shared ODCs, "
            "project-specific ODCs, IT team, parking and recreation/ "
            "breakout facilities according to the office knowledge base."
        ),
    },
    {
        "name": "Tower 2",
        "location_type": "BUILDING",
        "latitude": Decimal("16.534263"),
        "longitude": Decimal("80.790065"),
        "description": (
            "Tower 2 containing professional/experienced project ODCs, "
            "parking and recreation/breakout facilities according to the "
            "office knowledge base."
        ),
    },
]


def seed_office_locations() -> int:
    """Seeds or updates official office locations for the target tenant.

    Returns:
        Number of locations seeded/updated.
    """
    session_factory = get_session_factory()
    with session_factory() as session:
        # 1. Verify target tenant exists
        tenant = session.execute(
            select(Tenant).where(Tenant.id == TARGET_TENANT_ID)
        ).scalar_one_or_none()

        if tenant is None:
            logger.error("Target tenant %s not found. Aborting.", TARGET_TENANT_ID)
            sys.exit(1)

        logger.info("Seeding official office locations for tenant '%s' (%s)", tenant.name, tenant.id)

        count_inserted = 0
        count_updated = 0

        for loc_data in OFFICIAL_LOCATIONS:
            name = loc_data["name"]
            existing = session.execute(
                select(Location).where(
                    Location.tenant_id == TARGET_TENANT_ID,
                    Location.name == name,
                )
            ).scalar_one_or_none()

            if existing is None:
                new_loc = Location(
                    tenant_id=TARGET_TENANT_ID,
                    name=name,
                    location_type=loc_data["location_type"],
                    latitude=loc_data["latitude"],
                    longitude=loc_data["longitude"],
                    description=loc_data["description"],
                    is_active=True,
                )
                session.add(new_loc)
                session.flush()
                logger.info(
                    "  [INSERTED] %s (%s) -> Lat: %s, Lon: %s",
                    name,
                    loc_data["location_type"],
                    loc_data["latitude"],
                    loc_data["longitude"],
                )
                count_inserted += 1

                audit = AuditLog(
                    tenant_id=TARGET_TENANT_ID,
                    action="location_created",
                    entity_type="location",
                    entity_id=str(new_loc.id),
                    details=f"Office location '{name}' created with coordinates ({loc_data['latitude']}, {loc_data['longitude']})",
                )
                session.add(audit)
            else:
                # Update existing location to ensure exact canonical coordinates and description
                existing.location_type = loc_data["location_type"]
                existing.latitude = loc_data["latitude"]
                existing.longitude = loc_data["longitude"]
                existing.description = loc_data["description"]
                existing.is_active = True
                existing.updated_at = datetime.now(timezone.utc)
                logger.info(
                    "  [EXISTS/UPDATED] %s (%s) -> Lat: %s, Lon: %s",
                    name,
                    loc_data["location_type"],
                    loc_data["latitude"],
                    loc_data["longitude"],
                )
                count_updated += 1

        session.commit()
        logger.info(
            "Office locations seed complete. Inserted: %d, Updated/Verified: %d. Total: %d",
            count_inserted,
            count_updated,
            count_inserted + count_updated,
        )
        return count_inserted + count_updated


if __name__ == "__main__":
    seed_office_locations()
