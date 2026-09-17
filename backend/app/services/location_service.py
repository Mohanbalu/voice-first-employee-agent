"""Location Service for Office Geo-Coordinates, Distance Calculation, and Name Normalization.

This service manages authoritative geographic coordinates for the campus,
implements the Haversine great-circle distance algorithm, and provides
controlled normalization for natural language building queries.
"""

from __future__ import annotations

import logging
import math
import re
import uuid
from decimal import Decimal
from typing import Optional, Sequence

from fastapi import HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from backend.app.models.location import Location

logger = logging.getLogger("location.service")

# Mean Earth radius in kilometers (WGS 84 / IUGG standard)
EARTH_RADIUS_KM = 6371.0088

# Controlled canonical location names in HCL Campus
CANONICAL_LOCATIONS = {
    "ENTRANCE_GATE": "Entrance Gate",
    "SDC": "SDC",
    "TOWER_1": "Tower 1",
    "TOWER_2": "Tower 2",
}

# Controlled normalization lookup patterns
_NORMALIZATION_MAP: dict[str, str] = {
    # Entrance Gate aliases
    "entrance": "Entrance Gate",
    "entrance gate": "Entrance Gate",
    "main gate": "Entrance Gate",
    "gate": "Entrance Gate",
    "campus entrance": "Entrance Gate",
    "main entrance": "Entrance Gate",
    "entry gate": "Entrance Gate",
    "front gate": "Entrance Gate",
    # SDC aliases
    "sdc": "SDC",
    "sdc building": "SDC",
    "sdc block": "SDC",
    "software development center": "SDC",
    # Tower 1 aliases
    "tower 1": "Tower 1",
    "tower one": "Tower 1",
    "tower-1": "Tower 1",
    "tower 1 building": "Tower 1",
    "t1": "Tower 1",
    "tower1": "Tower 1",
    # Tower 2 aliases
    "tower 2": "Tower 2",
    "tower two": "Tower 2",
    "tower-2": "Tower 2",
    "tower 2 building": "Tower 2",
    "t2": "Tower 2",
    "tower2": "Tower 2",
}


def normalize_location_name(raw_name: str) -> str:
    """Normalizes natural language building/location references to their canonical name.

    Examples:
        "sdc building" -> "SDC"
        "tower one" -> "Tower 1"
        "main gate" -> "Entrance Gate"
        "Tower 2" -> "Tower 2"
    """
    if not raw_name:
        return raw_name

    cleaned = raw_name.strip().lower()
    # Remove common filler punctuation and normalize multiple spaces
    cleaned = re.sub(r"[^\w\s-]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Direct dictionary lookup
    if cleaned in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[cleaned]

    # Clean words like 'building', 'office', 'block' from end
    trimmed = re.sub(r"\b(building|block|office|campus)\b", "", cleaned).strip()
    if trimmed in _NORMALIZATION_MAP:
        return _NORMALIZATION_MAP[trimmed]

    # Word-number replacements
    if "one" in cleaned and "tower" in cleaned:
        return "Tower 1"
    if "two" in cleaned and "tower" in cleaned:
        return "Tower 2"

    # Return original trimmed string if no mapping matched
    return raw_name.strip()


def calculate_distance_km(lat1: float | Decimal, lon1: float | Decimal, lat2: float | Decimal, lon2: float | Decimal) -> float:
    """Calculates great-circle distance between two decimal coordinate pairs using the Haversine formula.

    Returns:
        Distance in kilometers.
    """
    f_lat1, f_lon1 = float(lat1), float(lon1)
    f_lat2, f_lon2 = float(lat2), float(lon2)

    phi1 = math.radians(f_lat1)
    phi2 = math.radians(f_lat2)
    delta_phi = math.radians(f_lat2 - f_lat1)
    delta_lambda = math.radians(f_lon2 - f_lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    # Ensure float precision bounds for atan2
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_KM * c


def calculate_distance_meters(lat1: float | Decimal, lon1: float | Decimal, lat2: float | Decimal, lon2: float | Decimal) -> float:
    """Calculates straight-line distance between two coordinate pairs in meters."""
    return calculate_distance_km(lat1, lon1, lat2, lon2) * 1000.0


def format_straight_line_distance(meters: float) -> str:
    """Formats a straight-line distance clearly according to project specifications.

    Terminology strictly states straight-line distance (never walking distance).
    """
    if meters >= 1000.0:
        km = meters / 1000.0
        return f"approximately {km:.2f} km in straight-line distance"
    return f"approximately {meters:.1f} meters in straight-line distance"


class LocationService:
    """Business logic and database operations for Office Locations."""

    @classmethod
    def list_locations(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        location_type: Optional[str] = None,
        active_only: bool = True,
    ) -> Sequence[Location]:
        """Lists all office locations belonging to a specific tenant."""
        stmt = select(Location).where(Location.tenant_id == tenant_id)
        if active_only:
            stmt = stmt.where(Location.is_active == True)  # noqa: E712
        if location_type:
            stmt = stmt.where(Location.location_type == location_type.strip().upper())
        stmt = stmt.order_by(Location.name.asc())
        return db.execute(stmt).scalars().all()

    @classmethod
    def get_location(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        location_id: uuid.UUID,
    ) -> Optional[Location]:
        """Fetches a single location by UUID ensuring strict tenant isolation."""
        stmt = select(Location).where(
            Location.id == location_id,
            Location.tenant_id == tenant_id,
        )
        return db.execute(stmt).scalar_one_or_none()

    @classmethod
    def find_location_by_name(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        query: str,
    ) -> Optional[Location]:
        """Finds a location by raw or normalized name for the given tenant."""
        if not query:
            return None

        normalized = normalize_location_name(query)

        # 1. Try exact match on normalized name
        stmt = select(Location).where(
            Location.tenant_id == tenant_id,
            func.lower(Location.name) == normalized.lower(),
        )
        loc = db.execute(stmt).scalar_one_or_none()
        if loc:
            return loc

        # 2. Try exact match on raw query
        stmt = select(Location).where(
            Location.tenant_id == tenant_id,
            func.lower(Location.name) == query.strip().lower(),
        )
        loc = db.execute(stmt).scalar_one_or_none()
        if loc:
            return loc

        # 3. Try partial substring match (e.g. 'tower 1' in name)
        stmt = select(Location).where(
            Location.tenant_id == tenant_id,
            Location.name.ilike(f"%{normalized}%"),
        )
        return db.execute(stmt).scalars().first()

    @classmethod
    def resolve_location(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        identifier: str,
    ) -> Location:
        """Resolves a location from either a UUID string or a location name.

        Raises HTTPException 404 if not found within the tenant.
        """
        # Try parsing as UUID
        try:
            loc_uuid = uuid.UUID(identifier)
            loc = cls.get_location(db, tenant_id, loc_uuid)
            if loc:
                return loc
        except (ValueError, AttributeError):
            pass

        # Try finding by name
        loc = cls.find_location_by_name(db, tenant_id, identifier)
        if loc:
            return loc

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{identifier}' not found in organization",
        )

    @classmethod
    def calculate_distance_between(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        origin_query: str,
        destination_query: str,
    ) -> dict:
        """Calculates straight-line distance between two locations in the campus."""
        loc1 = cls.resolve_location(db, tenant_id, origin_query)
        loc2 = cls.resolve_location(db, tenant_id, destination_query)

        dist_km = calculate_distance_km(loc1.latitude, loc1.longitude, loc2.latitude, loc2.longitude)
        dist_m = dist_km * 1000.0

        return {
            "origin": loc1.name,
            "destination": loc2.name,
            "origin_id": str(loc1.id),
            "destination_id": str(loc2.id),
            "distance_km": round(dist_km, 4),
            "distance_meters": round(dist_m, 2),
            "formatted": format_straight_line_distance(dist_m),
        }

    @classmethod
    def find_closest_location(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        reference_query: str,
        candidates_filter: Optional[list[str]] = None,
    ) -> dict:
        """Finds which building/location is closest to the given reference location."""
        ref_loc = cls.resolve_location(db, tenant_id, reference_query)
        all_locs = cls.list_locations(db, tenant_id, active_only=True)

        candidates = [loc for loc in all_locs if loc.id != ref_loc.id]

        if candidates_filter:
            allowed_names = {normalize_location_name(c).lower() for c in candidates_filter}
            candidates = [c for c in candidates if c.name.lower() in allowed_names]

        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No candidate locations found to compare with '{ref_loc.name}'",
            )

        closest_loc: Optional[Location] = None
        min_distance_m = float("inf")

        for cand in candidates:
            m = calculate_distance_meters(
                ref_loc.latitude, ref_loc.longitude, cand.latitude, cand.longitude
            )
            if m < min_distance_m:
                min_distance_m = m
                closest_loc = cand

        assert closest_loc is not None
        dist_km = min_distance_m / 1000.0

        return {
            "reference": ref_loc.name,
            "closest_location": closest_loc,
            "distance_km": round(dist_km, 4),
            "distance_meters": round(min_distance_m, 2),
            "formatted": (
                f"{closest_loc.name} is closest to {ref_loc.name} at "
                f"{format_straight_line_distance(min_distance_m)}"
            ),
        }

    @classmethod
    def resolve_current_location(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        latitude: float,
        longitude: float,
        accuracy: Optional[float] = None,
        query: Optional[str] = None,
    ) -> dict:
        """Determines current campus proximity from device GPS coordinates without persistence.

        - Compares employee coordinates against all known tenant locations.
        - Identifies the nearest location and evaluates proximity status.
        - Privacy: Does NOT store or log employee GPS coordinates.
        - Adheres strictly to: 'You appear to be near [Building]' (never claims 'inside').
        """
        all_locs = cls.list_locations(db, tenant_id, active_only=True)
        if not all_locs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No campus locations found for this organization.",
            )

        # 1. Calculate distance from current coordinates to every known location
        calculated_items = []
        loc_map = {}
        for loc in all_locs:
            dist_km = calculate_distance_km(latitude, longitude, loc.latitude, loc.longitude)
            dist_m = dist_km * 1000.0
            item = {
                "name": loc.name,
                "location_type": loc.location_type,
                "distance_meters": round(dist_m, 2),
                "distance_km": round(dist_km, 4),
                "formatted": format_straight_line_distance(dist_m),
            }
            calculated_items.append(item)
            loc_map[loc.name.lower()] = item

        # Sort by distance ascending
        calculated_items.sort(key=lambda x: x["distance_meters"])
        nearest = calculated_items[0]
        nearest_dist_m = nearest["distance_meters"]

        # 2. Determine categorical proximity status
        # Campus radius is roughly ~300 meters from center.
        # If nearest point is > 500m away, user is outside the HCL campus.
        if nearest_dist_m > 500.0:
            proximity_status = "OUTSIDE_CAMPUS"
            confidence_msg = (
                f"You appear to be outside the HCL campus. "
                f"The nearest campus location is {nearest['name']}, "
                f"approximately {format_straight_line_distance(nearest_dist_m)} away."
            )
        elif nearest_dist_m <= 30.0 or (accuracy is not None and nearest_dist_m <= accuracy + 15.0):
            proximity_status = "VERY_CLOSE"
            confidence_msg = f"You appear to be near {nearest['name']}."
            if accuracy is not None:
                confidence_msg += f" Your device reports an accuracy of approximately {int(accuracy)} meters."
        else:
            proximity_status = "NEAR"
            confidence_msg = (
                f"You appear to be near {nearest['name']}, "
                f"approximately {int(nearest_dist_m)} meters from the reference point."
            )
            if accuracy is not None:
                confidence_msg += f" Your device reports an accuracy of approximately {int(accuracy)} meters."

        # 3. Interpret specific question context (if user asked 'Am I near SDC?' or 'How far am I from Tower 1?')
        clean_q = (query or "").strip().lower()
        target_item = None

        # Check if a specific known building was queried
        for loc in all_locs:
            norm_name = loc.name.lower()
            # match name or aliases like 'sdc', 'tower 1', 'tower one', 'entrance'
            aliases = [norm_name]
            if "tower 1" in norm_name:
                aliases.extend(["tower one", "t1"])
            elif "tower 2" in norm_name:
                aliases.extend(["tower two", "t2"])
            elif "entrance" in norm_name:
                aliases.extend(["main gate", "gate", "campus entrance"])
            elif "sdc" in norm_name:
                aliases.extend(["sdc building", "sdc block"])

            if any(alias in clean_q for alias in aliases):
                target_item = loc_map.get(norm_name)
                break

        # Generate tailored natural language response text
        if target_item and ("how far" in clean_q or "distance" in clean_q):
            t_dist = target_item["distance_meters"]
            response_text = (
                f"Your current device location is approximately {int(t_dist)} meters "
                f"in straight-line distance from {target_item['name']}."
            )
        elif target_item and ("near" in clean_q or "close" in clean_q or "am i" in clean_q):
            t_dist = target_item["distance_meters"]
            # Consider "near" if within 100 meters or if it's the closest location
            if t_dist <= 100.0 or target_item["name"] == nearest["name"]:
                response_text = (
                    f"Yes, you appear to be near {target_item['name']}, "
                    f"approximately {int(t_dist)} meters away."
                )
            else:
                response_text = (
                    f"No, you are approximately {int(t_dist)} meters from {target_item['name']}. "
                    f"You are currently closer to {nearest['name']} "
                    f"(approximately {int(nearest_dist_m)} meters away)."
                )
        elif "which building" in clean_q or "which tower" in clean_q:
            bldgs = [item for item in calculated_items if item["location_type"] == "BUILDING"]
            nearest_bldg = bldgs[0] if bldgs else nearest
            response_text = (
                f"You are currently closest to {nearest_bldg['name']}, "
                f"approximately {int(nearest_bldg['distance_meters'])} meters away."
            )
        elif proximity_status == "OUTSIDE_CAMPUS":
            response_text = (
                f"You appear to be outside the HCL campus. Your current location is "
                f"approximately {format_straight_line_distance(nearest_dist_m)} from the nearest campus location ({nearest['name']})."
            )
        else:
            response_text = (
                f"You appear to be near {nearest['name']}. Your current location is "
                f"approximately {int(nearest_dist_m)} meters from the {nearest['name']} reference point."
            )
            if accuracy is not None:
                response_text += f" Your device reports an accuracy of approximately {int(accuracy)} meters."

        return {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "accuracy": float(accuracy) if accuracy is not None else None,
            "nearest_location": nearest,
            "proximity_status": proximity_status,
            "confidence_message": confidence_msg,
            "response_text": response_text,
            "target_location": target_item,
            "known_locations": calculated_items,
        }

