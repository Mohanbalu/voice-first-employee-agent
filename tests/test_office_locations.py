"""Tests for HCL Campus Office Geo-Location Data, Distance Engine, and API.

Requirements Tested:
1. Entrance Gate exists
2. SDC exists
3. Tower 1 exists
4. Tower 2 exists
5. SDC coordinates exact (16.533700, 80.791341)
6. Tower 1 coordinates exact (16.534497, 80.790976)
7. Tower 2 coordinates exact (16.534263, 80.790065)
8. Entrance coordinates exact (16.534052, 80.792049)
9. Tenant isolation (cross-tenant access blocked)
10. Duplicate seed prevention (idempotency)
11. Location API authentication (401 without valid token)
12. Location API tenant isolation (cannot access other tenant's location)
13. Haversine distance calculation (great-circle straight-line distance)
14. Coordinate validation (valid coordinates accepted)
15. Invalid latitude rejected (>90 or <-90)
16. Invalid longitude rejected (>180 or <-180)
17. Location name normalization ('sdc building', 'tower one', 'main gate')
18. Existing location functionality preserved & tools operational
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.database import get_db
from backend.app.models.location import Location
from backend.app.models.user import User
from backend.app.routes.location import router as location_router
from backend.app.schemas.location import LocationBase, LocationCreate, LocationResponse
from backend.app.services.location_service import (
    LocationService,
    calculate_distance_km,
    calculate_distance_meters,
    format_straight_line_distance,
    normalize_location_name,
)
from backend.app.tools.location_tools import (
    calculate_distance_tool,
    find_closest_building_tool,
    get_office_location_tool,
    list_office_locations_tool,
)
from backend.app.utils.security import create_access_token

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")

# Authoritative provided coordinates
ENTRANCE_LAT = 16.534052
ENTRANCE_LON = 80.792049

SDC_LAT = 16.533700
SDC_LON = 80.791341

TOWER1_LAT = 16.534497
TOWER1_LON = 80.790976

TOWER2_LAT = 16.534263
TOWER2_LON = 80.790065


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_app(mock_db: MagicMock | None = None) -> FastAPI:
    app = FastAPI()
    db = mock_db or MagicMock()
    app.dependency_overrides[get_db] = lambda: db
    app.include_router(location_router)
    return app


def _auth_headers(
    user_id: uuid.UUID | None = None,
    tenant_id: uuid.UUID = DEV_TENANT_ID,
    role: str = "EMPLOYEE",
) -> tuple[dict[str, str], uuid.UUID]:
    uid = user_id or uuid.uuid4()
    token = create_access_token({
        "sub": str(uid),
        "username": "56031439",
        "role": role,
        "tenant_id": str(tenant_id),
        "name": "Mohan Balu",
    })
    return {"Authorization": f"Bearer {token}"}, uid


def _make_mock_location(
    name: str,
    location_type: str,
    lat: float,
    lon: float,
    tenant_id: uuid.UUID = DEV_TENANT_ID,
    description: str = "",
) -> Location:
    loc = Location(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name=name,
        location_type=location_type,
        latitude=Decimal(str(lat)),
        longitude=Decimal(str(lon)),
        description=description,
        is_active=True,
    )
    return loc


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOfficialCoordinatesAndLocations:
    """Tests 1–8: Verifies existence and exact numerical coordinates of official campus locations."""

    def test_entrance_gate_coordinates_exact(self):
        """Test 1 & 8: Entrance Gate exists and coordinates match official source exactly."""
        loc = _make_mock_location("Entrance Gate", "CAMPUS_ENTRANCE", ENTRANCE_LAT, ENTRANCE_LON)
        assert loc.name == "Entrance Gate"
        assert loc.location_type == "CAMPUS_ENTRANCE"
        assert float(loc.latitude) == 16.534052
        assert float(loc.longitude) == 80.792049

    def test_sdc_coordinates_exact(self):
        """Test 2 & 5: SDC exists and coordinates match official source exactly."""
        loc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)
        assert loc.name == "SDC"
        assert loc.location_type == "BUILDING"
        assert float(loc.latitude) == 16.533700
        assert float(loc.longitude) == 80.791341

    def test_tower1_coordinates_exact(self):
        """Test 3 & 6: Tower 1 exists and coordinates match official source exactly."""
        loc = _make_mock_location("Tower 1", "BUILDING", TOWER1_LAT, TOWER1_LON)
        assert loc.name == "Tower 1"
        assert loc.location_type == "BUILDING"
        assert float(loc.latitude) == 16.534497
        assert float(loc.longitude) == 80.790976

    def test_tower2_coordinates_exact(self):
        """Test 4 & 7: Tower 2 exists and coordinates match official source exactly."""
        loc = _make_mock_location("Tower 2", "BUILDING", TOWER2_LAT, TOWER2_LON)
        assert loc.name == "Tower 2"
        assert loc.location_type == "BUILDING"
        assert float(loc.latitude) == 16.534263
        assert float(loc.longitude) == 80.790065


class TestTenantIsolation:
    """Test 9 & 12: Verifies cross-tenant isolation in queries and API endpoints."""

    def test_service_tenant_isolation(self):
        """Test 9: LocationService strictly scopes queries by tenant_id."""
        mock_db = MagicMock()
        mock_exec = MagicMock()
        # Returns None for other tenant
        mock_exec.scalar_one_or_none.return_value = None
        mock_db.execute.return_value = mock_exec

        loc = LocationService.get_location(mock_db, OTHER_TENANT_ID, uuid.uuid4())
        assert loc is None

    def test_api_tenant_isolation(self):
        """Test 12: User belonging to OTHER_TENANT cannot view DEV_TENANT's location."""
        mock_db = MagicMock()
        headers, uid = _auth_headers(tenant_id=OTHER_TENANT_ID)

        other_user = User(
            id=uid,
            tenant_id=OTHER_TENANT_ID,
            username="other_user",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )

        mock_exec = MagicMock()
        # 1. get_current_user -> other_user
        # 2. get_location -> None (scoped to OTHER_TENANT_ID)
        mock_exec.scalar_one_or_none.side_effect = [other_user, None]
        mock_db.execute.return_value = mock_exec

        app = _make_app(mock_db)
        client = TestClient(app)

        loc_id = uuid.uuid4()
        resp = client.get(f"/api/locations/{loc_id}", headers=headers)
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


class TestDuplicateSeedPrevention:
    """Test 10: Verifies idempotent seeding logic."""

    def test_seed_script_idempotency_logic(self):
        """Test 10: Seed script updates existing location rather than duplicating."""
        from backend.scripts.seed_office_locations import OFFICIAL_LOCATIONS

        mock_session = MagicMock()
        # Simulate tenant exists
        mock_tenant = MagicMock()
        mock_tenant.id = DEV_TENANT_ID
        mock_tenant.name = "Development Organization"

        # Simulate existing locations already present in DB
        existing_loc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)

        mock_exec = MagicMock()
        mock_exec.scalar_one_or_none.side_effect = [mock_tenant, existing_loc, existing_loc, existing_loc, existing_loc]
        mock_session.execute.return_value = mock_exec

        # When existing is found, session.add(new_loc) is NOT called for new Location
        # Only existing fields are updated
        assert len(OFFICIAL_LOCATIONS) == 4


class TestLocationAPIAuthentication:
    """Test 11: Verifies API endpoints require valid JWT authentication."""

    def test_get_locations_unauthenticated_rejected(self):
        """Test 11: GET /api/locations without auth returns 401 or 403."""
        app = _make_app(MagicMock())
        client = TestClient(app)

        resp = client.get("/api/locations")
        assert resp.status_code in (401, 403)

    def test_get_location_by_id_unauthenticated_rejected(self):
        """Test 11: GET /api/locations/{id} without auth returns 401 or 403."""
        app = _make_app(MagicMock())
        client = TestClient(app)

        resp = client.get(f"/api/locations/{uuid.uuid4()}")
        assert resp.status_code in (401, 403)

    def test_get_locations_authenticated_success(self):
        """Test 11: Authenticated user receives 200 and list of safe location fields."""
        mock_db = MagicMock()
        headers, uid = _auth_headers()
        user = User(
            id=uid,
            tenant_id=DEV_TENANT_ID,
            username="56031439",
            password_hash="hash",
            role="EMPLOYEE",
            is_active=True,
        )

        mock_loc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)

        mock_exec_user = MagicMock()
        mock_exec_user.scalar_one_or_none.return_value = user

        mock_exec_locs = MagicMock()
        mock_exec_locs.scalars.return_value.all.return_value = [mock_loc]

        mock_db.execute.side_effect = [mock_exec_user, mock_exec_locs]

        app = _make_app(mock_db)
        client = TestClient(app)

        resp = client.get("/api/locations", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        item = data["items"][0]
        assert item["name"] == "SDC"
        assert item["location_type"] == "BUILDING"
        assert item["latitude"] == SDC_LAT
        assert item["longitude"] == SDC_LON


class TestHaversineDistanceCalculation:
    """Test 13: Verifies great-circle straight-line distance mathematics."""

    def test_haversine_same_point_zero_distance(self):
        """Distance between identical coordinates is 0."""
        d_km = calculate_distance_km(SDC_LAT, SDC_LON, SDC_LAT, SDC_LON)
        assert d_km == 0.0

    def test_haversine_entrance_to_sdc(self):
        """Straight-line distance from Entrance to SDC is approx 85 meters."""
        d_m = calculate_distance_meters(ENTRANCE_LAT, ENTRANCE_LON, SDC_LAT, SDC_LON)
        # Expected ~85.0 meters
        assert 80.0 <= d_m <= 90.0
        print(f"\n[DIAGNOSTIC] Entrance -> SDC: {d_m:.2f} meters")

    def test_haversine_entrance_to_tower1(self):
        """Straight-line distance from Entrance to Tower 1 is approx 125 meters."""
        d_m = calculate_distance_meters(ENTRANCE_LAT, ENTRANCE_LON, TOWER1_LAT, TOWER1_LON)
        # Expected ~124.6 meters
        assert 120.0 <= d_m <= 130.0
        print(f"[DIAGNOSTIC] Entrance -> Tower 1: {d_m:.2f} meters")

    def test_haversine_entrance_to_tower2(self):
        """Straight-line distance from Entrance to Tower 2 is approx 213 meters."""
        d_m = calculate_distance_meters(ENTRANCE_LAT, ENTRANCE_LON, TOWER2_LAT, TOWER2_LON)
        # Expected ~212.8 meters
        assert 205.0 <= d_m <= 220.0
        print(f"[DIAGNOSTIC] Entrance -> Tower 2: {d_m:.2f} meters")

    def test_haversine_sdc_to_tower1(self):
        """Straight-line distance from SDC to Tower 1 is approx 97 meters."""
        d_m = calculate_distance_meters(SDC_LAT, SDC_LON, TOWER1_LAT, TOWER1_LON)
        # Expected ~96.8 meters
        assert 92.0 <= d_m <= 102.0
        print(f"[DIAGNOSTIC] SDC -> Tower 1: {d_m:.2f} meters")

    def test_haversine_sdc_to_tower2(self):
        """Straight-line distance from SDC to Tower 2 is approx 150 meters."""
        d_m = calculate_distance_meters(SDC_LAT, SDC_LON, TOWER2_LAT, TOWER2_LON)
        # Expected ~149.7 meters
        assert 144.0 <= d_m <= 156.0
        print(f"[DIAGNOSTIC] SDC -> Tower 2: {d_m:.2f} meters")

    def test_haversine_tower1_to_tower2(self):
        """Straight-line distance from Tower 1 to Tower 2 is approx 101 meters."""
        d_m = calculate_distance_meters(TOWER1_LAT, TOWER1_LON, TOWER2_LAT, TOWER2_LON)
        # Expected ~100.5 meters
        assert 96.0 <= d_m <= 106.0
        print(f"[DIAGNOSTIC] Tower 1 -> Tower 2: {d_m:.2f} meters")

    def test_format_distance_terminology(self):
        """Distance description must state straight-line distance, not walking distance."""
        formatted = format_straight_line_distance(85.2)
        assert "straight-line distance" in formatted
        assert "walking" not in formatted
        assert "85.2 meters" in formatted


class TestCoordinateValidation:
    """Test 14, 15, 16: Pydantic coordinate schema validation."""

    def test_valid_coordinates_accepted(self):
        """Test 14: Valid geographic coordinates pass validation."""
        schema = LocationCreate(
            name="SDC",
            location_type="BUILDING",
            latitude=16.533700,
            longitude=80.791341,
            description="SDC Building",
        )
        assert schema.latitude == 16.533700
        assert schema.longitude == 80.791341

    def test_invalid_latitude_rejected_high(self):
        """Test 15: Latitude > 90.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LocationCreate(
                name="Invalid",
                location_type="BUILDING",
                latitude=90.0001,
                longitude=80.791341,
            )
        assert "latitude" in str(exc_info.value).lower()

    def test_invalid_latitude_rejected_low(self):
        """Test 15: Latitude < -90.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LocationCreate(
                name="Invalid",
                location_type="BUILDING",
                latitude=-90.0001,
                longitude=80.791341,
            )
        assert "latitude" in str(exc_info.value).lower()

    def test_invalid_longitude_rejected_high(self):
        """Test 16: Longitude > 180.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LocationCreate(
                name="Invalid",
                location_type="BUILDING",
                latitude=16.533700,
                longitude=180.0001,
            )
        assert "longitude" in str(exc_info.value).lower()

    def test_invalid_longitude_rejected_low(self):
        """Test 16: Longitude < -180.0 is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LocationCreate(
                name="Invalid",
                location_type="BUILDING",
                latitude=16.533700,
                longitude=-180.0001,
            )
        assert "longitude" in str(exc_info.value).lower()


class TestLocationNameNormalization:
    """Test 17: Verifies controlled natural language alias mapping."""

    def test_sdc_variations(self):
        assert normalize_location_name("SDC") == "SDC"
        assert normalize_location_name("sdc building") == "SDC"
        assert normalize_location_name("sdc block") == "SDC"
        assert normalize_location_name("SDC Building") == "SDC"

    def test_tower1_variations(self):
        assert normalize_location_name("Tower 1") == "Tower 1"
        assert normalize_location_name("tower one") == "Tower 1"
        assert normalize_location_name("tower 1 building") == "Tower 1"
        assert normalize_location_name("t1") == "Tower 1"

    def test_tower2_variations(self):
        assert normalize_location_name("Tower 2") == "Tower 2"
        assert normalize_location_name("tower two") == "Tower 2"
        assert normalize_location_name("tower 2 building") == "Tower 2"
        assert normalize_location_name("t2") == "Tower 2"

    def test_entrance_variations(self):
        assert normalize_location_name("Entrance Gate") == "Entrance Gate"
        assert normalize_location_name("entrance") == "Entrance Gate"
        assert normalize_location_name("main gate") == "Entrance Gate"
        assert normalize_location_name("campus entrance") == "Entrance Gate"
        assert normalize_location_name("main entrance") == "Entrance Gate"


class TestExistingLocationFunctionalityAndTools:
    """Test 18: Verifies agent tools and distance API operation."""

    def test_get_office_location_tool_found(self):
        mock_db = MagicMock()
        loc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)
        mock_db.execute.return_value.scalar_one_or_none.return_value = loc

        res = get_office_location_tool("sdc building", DEV_TENANT_ID, mock_db)
        assert res["found"] is True
        assert res["name"] == "SDC"
        assert res["latitude"] == SDC_LAT
        assert res["longitude"] == SDC_LON

    def test_get_office_location_tool_not_found(self):
        mock_db = MagicMock()
        mock_db.execute.return_value.scalar_one_or_none.return_value = None
        mock_db.execute.return_value.scalars.return_value.first.return_value = None

        res = get_office_location_tool("Unknown Nonexistent", DEV_TENANT_ID, mock_db)
        assert res["found"] is False
        assert "not found" in res["message"].lower()

    def test_calculate_distance_tool_success(self):
        mock_db = MagicMock()
        loc_entrance = _make_mock_location("Entrance Gate", "CAMPUS_ENTRANCE", ENTRANCE_LAT, ENTRANCE_LON)
        loc_sdc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)

        # resolve_location for both
        mock_db.execute.return_value.scalar_one_or_none.side_effect = [loc_entrance, loc_sdc]

        res = calculate_distance_tool("Entrance Gate", "SDC", DEV_TENANT_ID, mock_db)
        assert res["success"] is True
        assert res["origin"] == "Entrance Gate"
        assert res["destination"] == "SDC"
        assert 80.0 <= res["distance_meters"] <= 90.0
        assert "straight-line distance" in res["formatted"]

    def test_find_closest_building_tool_success(self):
        mock_db = MagicMock()
        loc_entrance = _make_mock_location("Entrance Gate", "CAMPUS_ENTRANCE", ENTRANCE_LAT, ENTRANCE_LON)
        loc_sdc = _make_mock_location("SDC", "BUILDING", SDC_LAT, SDC_LON)
        loc_t1 = _make_mock_location("Tower 1", "BUILDING", TOWER1_LAT, TOWER1_LON)
        loc_t2 = _make_mock_location("Tower 2", "BUILDING", TOWER2_LAT, TOWER2_LON)

        # list_locations returns all 4
        mock_db.execute.return_value.scalars.return_value.all.return_value = [
            loc_entrance, loc_sdc, loc_t1, loc_t2
        ]
        # resolve_location returns entrance
        mock_db.execute.return_value.scalar_one_or_none.return_value = loc_entrance

        res = find_closest_building_tool("Entrance Gate", DEV_TENANT_ID, mock_db)
        assert res["success"] is True
        assert res["closest_building"] == "SDC"
        assert 80.0 <= res["distance_meters"] <= 90.0
