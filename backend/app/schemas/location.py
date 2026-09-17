"""Pydantic Schemas for Office Locations and Geo-Coordinates."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LocationBase(BaseModel):
    """Base schema for office location data."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Official name of the campus location or building",
        examples=["SDC", "Tower 1", "Tower 2", "Entrance Gate"],
    )
    location_type: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Location category, e.g. CAMPUS_ENTRANCE, BUILDING",
        examples=["BUILDING", "CAMPUS_ENTRANCE"],
    )
    latitude: float = Field(
        ...,
        description="Geographic latitude in decimal degrees (-90.0 to 90.0)",
        examples=[16.533700],
    )
    longitude: float = Field(
        ...,
        description="Geographic longitude in decimal degrees (-180.0 to 180.0)",
        examples=[80.791341],
    )
    description: Optional[str] = Field(
        None,
        description="Detailed description of the location, teams, or facilities",
    )
    is_active: bool = Field(
        default=True,
        description="Whether the location is currently active",
    )

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: float | Decimal) -> float:
        val = float(v)
        if not (-90.0 <= val <= 90.0):
            raise ValueError(f"Latitude must be between -90.0 and 90.0 degrees, got {val}")
        return val

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: float | Decimal) -> float:
        val = float(v)
        if not (-180.0 <= val <= 180.0):
            raise ValueError(f"Longitude must be between -180.0 and 180.0 degrees, got {val}")
        return val


class LocationCreate(LocationBase):
    """Schema for creating a new location."""
    pass


class LocationUpdate(BaseModel):
    """Schema for updating an existing location."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location_type: Optional[str] = Field(None, min_length=1, max_length=50)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

    @field_validator("latitude")
    @classmethod
    def validate_latitude(cls, v: Optional[float | Decimal]) -> Optional[float]:
        if v is not None:
            val = float(v)
            if not (-90.0 <= val <= 90.0):
                raise ValueError(f"Latitude must be between -90.0 and 90.0 degrees, got {val}")
            return val
        return None

    @field_validator("longitude")
    @classmethod
    def validate_longitude(cls, v: Optional[float | Decimal]) -> Optional[float]:
        if v is not None:
            val = float(v)
            if not (-180.0 <= val <= 180.0):
                raise ValueError(f"Longitude must be between -180.0 and 180.0 degrees, got {val}")
            return val
        return None


class LocationResponse(BaseModel):
    """Safe schema returned to API clients."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    location_type: str
    latitude: float
    longitude: float
    description: Optional[str] = None
    is_active: bool
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def convert_decimal_to_float(cls, v: object) -> float:
        if v is not None:
            return float(v)
        return 0.0


class LocationListResponse(BaseModel):
    """Response containing a list of locations."""

    items: list[LocationResponse]
    total: int


class DistanceRequest(BaseModel):
    """Request to calculate straight-line distance between two locations."""

    origin: str = Field(..., description="Name or UUID of the starting location")
    destination: str = Field(..., description="Name or UUID of the destination location")


class DistanceResponse(BaseModel):
    """Response containing straight-line distance between two locations."""

    origin: str
    destination: str
    distance_km: float
    distance_meters: float
    formatted: str = Field(
        ...,
        description="Human-readable description stating straight-line distance",
        examples=["approximately 85.2 meters in straight-line distance"],
    )


class ClosestLocationResponse(BaseModel):
    """Response indicating which building/location is closest to a reference point."""

    reference: str
    closest_location: LocationResponse
    distance_km: float
    distance_meters: float
    formatted: str


class LocationDistanceItem(BaseModel):
    """Represents a known campus location paired with calculated distance."""

    name: str
    location_type: str
    distance_meters: float
    distance_km: float
    formatted: str


class CurrentLocationRequest(BaseModel):
    """Payload sent by device/browser containing GPS coordinates."""

    latitude: float = Field(
        ...,
        description="Current device latitude in decimal degrees (-90.0 to 90.0)",
        examples=[16.533700],
    )
    longitude: float = Field(
        ...,
        description="Current device longitude in decimal degrees (-180.0 to 180.0)",
        examples=[80.791341],
    )
    accuracy: Optional[float] = Field(
        default=None,
        description="Device reported GPS horizontal accuracy in meters (radius)",
        examples=[10.0],
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional employee question (e.g. 'Where am I?', 'Am I near SDC?')",
        examples=["Where am I?"],
    )

    @field_validator("latitude")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        val = float(v)
        if not (-90.0 <= val <= 90.0):
            raise ValueError(f"Latitude must be between -90.0 and 90.0 degrees, got {val}")
        return val

    @field_validator("longitude")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        val = float(v)
        if not (-180.0 <= val <= 180.0):
            raise ValueError(f"Longitude must be between -180.0 and 180.0 degrees, got {val}")
        return val

    @field_validator("accuracy")
    @classmethod
    def validate_acc(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            val = float(v)
            if val < 0:
                raise ValueError(f"Accuracy radius must be non-negative, got {val}")
            return val
        return None


class CurrentLocationResponse(BaseModel):
    """Structured response comparing employee GPS with campus layout."""

    latitude: float
    longitude: float
    accuracy: Optional[float] = None
    nearest_location: LocationDistanceItem
    proximity_status: str = Field(
        ...,
        description="Categorical proximity: 'VERY_CLOSE', 'NEAR', or 'OUTSIDE_CAMPUS'",
    )
    confidence_message: str = Field(
        ...,
        description="Deterministic context considering distance and GPS accuracy without overclaiming",
    )
    response_text: str = Field(
        ...,
        description="Natural language answer tailored to the user's specific query",
    )
    target_location: Optional[LocationDistanceItem] = Field(
        default=None,
        description="Calculated distance for specific building if user asked 'Am I near X?'",
    )
    known_locations: list[LocationDistanceItem] = Field(
        default_factory=list,
        description="All known campus locations sorted by proximity",
    )

