"""Schedule Routes — Module 6.

REST API endpoints for workplace schedules, reminders, and recurring tasks.
Enforces strict JWT-authenticated tenant and user isolation.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import get_current_user
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.schedule import (
    ScheduleActionResponse,
    ScheduleCreate,
    ScheduleListResponse,
    ScheduleResponse,
    ScheduleUpdate,
)
from backend.app.services.scheduling_service import SchedulingService

logger = logging.getLogger("routes.schedules")
router = APIRouter(prefix="/api/schedules", tags=["schedules"])


def _get_scheduling_service(db: Session = Depends(get_db)) -> SchedulingService:
    return SchedulingService(db=db)


@router.post(
    "",
    response_model=ScheduleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Schedule / Reminder",
    description="Creates a new scheduled reminder or task for the authenticated employee.",
)
def create_schedule(
    schedule_in: ScheduleCreate,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleResponse:
    schedule = service.create_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        data=schedule_in,
    )
    return ScheduleResponse.model_validate(schedule)


@router.get(
    "",
    response_model=ScheduleListResponse,
    summary="List Employee Schedules",
    description="Lists schedules for the current user with optional status and time filtering.",
)
def list_schedules(
    status: Optional[str] = Query(None, description="PENDING | COMPLETED | CANCELLED | MISSED"),
    upcoming_only: bool = Query(False, description="Filter only future pending items"),
    today_only: bool = Query(False, description="Filter only items scheduled for today"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleListResponse:
    total, items = service.list_schedules(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        status_filter=status,
        upcoming_only=upcoming_only,
        today_only=today_only,
        limit=limit,
        offset=offset,
    )
    return ScheduleListResponse(
        total=total,
        items=[ScheduleResponse.model_validate(item) for item in items],
    )


@router.get(
    "/{schedule_id}",
    response_model=ScheduleResponse,
    summary="Get Schedule Detail",
)
def get_schedule(
    schedule_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleResponse:
    schedule = service.get_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        schedule_id=schedule_id,
    )
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found or unauthorized.",
        )
    return ScheduleResponse.model_validate(schedule)


@router.patch(
    "/{schedule_id}",
    response_model=ScheduleResponse,
    summary="Update Schedule",
)
def update_schedule(
    schedule_id: uuid.UUID,
    update_in: ScheduleUpdate,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleResponse:
    schedule = service.update_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        schedule_id=schedule_id,
        data=update_in,
    )
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found or unauthorized.",
        )
    return ScheduleResponse.model_validate(schedule)


@router.post(
    "/{schedule_id}/complete",
    response_model=ScheduleActionResponse,
    summary="Complete Schedule",
    description="Marks schedule as completed and spawns the next occurrence if recurring.",
)
def complete_schedule(
    schedule_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleActionResponse:
    schedule = service.complete_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        schedule_id=schedule_id,
    )
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found or unauthorized.",
        )
    return ScheduleActionResponse(
        success=True,
        message="Schedule marked as completed.",
        schedule=ScheduleResponse.model_validate(schedule),
    )


@router.post(
    "/{schedule_id}/cancel",
    response_model=ScheduleActionResponse,
    summary="Cancel Schedule",
)
def cancel_schedule(
    schedule_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
) -> ScheduleActionResponse:
    schedule = service.cancel_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        schedule_id=schedule_id,
    )
    if not schedule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found or unauthorized.",
        )
    return ScheduleActionResponse(
        success=True,
        message="Schedule cancelled.",
        schedule=ScheduleResponse.model_validate(schedule),
    )


@router.delete(
    "/{schedule_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Schedule",
)
def delete_schedule(
    schedule_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    service: SchedulingService = Depends(_get_scheduling_service),
):
    deleted = service.delete_schedule(
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
        schedule_id=schedule_id,
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Schedule not found or unauthorized.",
        )
    return None
