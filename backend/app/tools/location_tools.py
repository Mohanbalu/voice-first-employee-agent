"""Location Tools for Agent and LangGraph Workflows.

These tools provide clean interfaces for retrieving office locations,
computing straight-line distances, and querying campus layout
without exposing raw SQL or direct DB queries to agent nodes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session

from backend.app.services.location_service import (
    LocationService,
    calculate_distance_km,
    calculate_distance_meters,
    format_straight_line_distance,
    normalize_location_name,
)

logger = logging.getLogger("location.tools")


def get_office_location_tool(
    name: str,
    tenant_id: uuid.UUID,
    db: Session,
) -> dict[str, Any]:
    """Retrieves authoritative geographic coordinates and details for a campus location.

    Args:
        name: Name of the building or entrance (e.g. 'SDC', 'Tower 1', 'Entrance Gate').
        tenant_id: Authenticated user's tenant UUID.
        db: Database session.

    Returns:
        Dict with location details or error message.
    """
    loc = LocationService.find_location_by_name(db, tenant_id, name)
    if loc is None:
        return {
            "found": False,
            "message": f"Location '{name}' was not found on campus.",
        }

    return {
        "found": True,
        "name": loc.name,
        "location_type": loc.location_type,
        "latitude": float(loc.latitude),
        "longitude": float(loc.longitude),
        "description": loc.description,
        "coordinates_str": f"{float(loc.latitude):.6f}, {float(loc.longitude):.6f}",
    }


def list_office_locations_tool(
    tenant_id: uuid.UUID,
    db: Session,
    location_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Lists all active office locations and buildings in the campus.

    Args:
        tenant_id: Authenticated user's tenant UUID.
        db: Database session.
        location_type: Optional filter (e.g. 'BUILDING', 'CAMPUS_ENTRANCE').

    Returns:
        List of location dictionaries.
    """
    locs = LocationService.list_locations(db, tenant_id, location_type=location_type, active_only=True)
    return [
        {
            "id": str(loc.id),
            "name": loc.name,
            "location_type": loc.location_type,
            "latitude": float(loc.latitude),
            "longitude": float(loc.longitude),
            "description": loc.description,
        }
        for loc in locs
    ]


def calculate_distance_tool(
    origin: str,
    destination: str,
    tenant_id: uuid.UUID,
    db: Session,
) -> dict[str, Any]:
    """Calculates the straight-line distance between two campus locations.

    Args:
        origin: Starting building or entrance name (e.g. 'Entrance Gate').
        destination: Target building or entrance name (e.g. 'Tower 2').
        tenant_id: Authenticated user's tenant UUID.
        db: Database session.

    Returns:
        Dict with straight-line distance in meters and km with descriptive formatting.
    """
    try:
        result = LocationService.calculate_distance_between(
            db, tenant_id, origin, destination
        )
        return {
            "success": True,
            **result,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }


def find_closest_building_tool(
    reference: str,
    tenant_id: uuid.UUID,
    db: Session,
    buildings_only: bool = True,
) -> dict[str, Any]:
    """Finds which building/location is closest to a reference point on campus.

    Args:
        reference: Reference location name (e.g. 'Entrance Gate').
        tenant_id: Authenticated user's tenant UUID.
        db: Database session.
        buildings_only: If True, only compares against buildings.

    Returns:
        Dict with closest location details and distance.
    """
    try:
        all_locs = LocationService.list_locations(db, tenant_id, active_only=True)
        ref_loc = LocationService.resolve_location(db, tenant_id, reference)

        candidates = [loc for loc in all_locs if loc.id != ref_loc.id]
        if buildings_only:
            candidates = [c for c in candidates if c.location_type == "BUILDING"]

        if not candidates:
            return {
                "success": False,
                "message": f"No candidates found to compare with {reference}.",
            }

        closest_loc = None
        min_m = float("inf")
        for c in candidates:
            m = calculate_distance_meters(ref_loc.latitude, ref_loc.longitude, c.latitude, c.longitude)
            if m < min_m:
                min_m = m
                closest_loc = c

        assert closest_loc is not None
        return {
            "success": True,
            "reference": ref_loc.name,
            "closest_building": closest_loc.name,
            "distance_meters": round(min_m, 2),
            "distance_km": round(min_m / 1000.0, 4),
            "formatted": (
                f"{closest_loc.name} is closest to {ref_loc.name} at "
                f"{format_straight_line_distance(min_m)}"
            ),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }
