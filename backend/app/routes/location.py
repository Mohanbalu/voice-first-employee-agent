"""API Routes for Campus Office Locations and Geo-Coordinates."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.routes.auth import get_current_user
from backend.app.schemas.location import (
    ClosestLocationResponse,
    DistanceRequest,
    DistanceResponse,
    LocationListResponse,
    LocationResponse,
)
from backend.app.services.location_service import LocationService

logger = logging.getLogger("api.locations")

router = APIRouter(prefix="/api/locations", tags=["locations"])


@router.get("", response_model=LocationListResponse)
def list_locations(
    location_type: Optional[str] = Query(
        None,
        description="Optional filter by location type, e.g. CAMPUS_ENTRANCE or BUILDING",
    ),
    active_only: bool = Query(True, description="Filter for active locations only"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LocationListResponse:
    """Retrieves all campus office locations for the authenticated user's organization.

    Enforces strict tenant isolation — only locations for the caller's tenant are returned.
    """
    locs = LocationService.list_locations(
        db=db,
        tenant_id=current_user.tenant_id,
        location_type=location_type,
        active_only=active_only,
    )
    items = [LocationResponse.model_validate(loc) for loc in locs]
    return LocationListResponse(items=items, total=len(items))


@router.get("/search/{name}", response_model=LocationResponse)
def search_location_by_name(
    name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LocationResponse:
    """Finds an office location by natural language or canonical building name.

    Supports aliases such as 'SDC', 'sdc building', 'Tower 1', 'tower one', 'main gate'.
    """
    loc = LocationService.find_location_by_name(
        db=db,
        tenant_id=current_user.tenant_id,
        query=name,
    )
    if not loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location '{name}' not found in organization",
        )
    return LocationResponse.model_validate(loc)


@router.post("/distance", response_model=DistanceResponse)
def calculate_distance(
    payload: DistanceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DistanceResponse:
    """Calculates straight-line great-circle distance between two campus locations.

    Uses the Haversine formula based on authoritative coordinates stored in PostgreSQL.
    """
    result = LocationService.calculate_distance_between(
        db=db,
        tenant_id=current_user.tenant_id,
        origin_query=payload.origin,
        destination_query=payload.destination,
    )
    return DistanceResponse(
        origin=result["origin"],
        destination=result["destination"],
        distance_km=result["distance_km"],
        distance_meters=result["distance_meters"],
        formatted=result["formatted"],
    )


@router.get("/closest", response_model=ClosestLocationResponse)
def get_closest_location(
    reference: str = Query(..., description="Reference location name or UUID, e.g. Entrance Gate"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ClosestLocationResponse:
    """Calculates which building/location is physically closest to the reference point."""
    result = LocationService.find_closest_location(
        db=db,
        tenant_id=current_user.tenant_id,
        reference_query=reference,
    )
    return ClosestLocationResponse(
        reference=result["reference"],
        closest_location=LocationResponse.model_validate(result["closest_location"]),
        distance_km=result["distance_km"],
        distance_meters=result["distance_meters"],
        formatted=result["formatted"],
    )


@router.get("/{location_id}", response_model=LocationResponse)
def get_location(
    location_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> LocationResponse:
    """Retrieves a single location by UUID ensuring strict tenant isolation."""
    loc = LocationService.get_location(
        db=db,
        tenant_id=current_user.tenant_id,
        location_id=location_id,
    )
    if not loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Location not found in organization",
        )
    return LocationResponse.model_validate(loc)
