"""Ticket Management Endpoints.

Protected routes for ticket submission, tracking, and resolution.
Employees can create tickets and view their own.
HR can view all organizational tickets and manage statuses.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.auth.dependencies import get_current_user, require_hr
from backend.app.database import get_db
from backend.app.models.user import User
from backend.app.schemas.ticket import (
    TicketAnswerRequest,
    TicketCreateRequest,
    TicketListResponse,
    TicketResponse,
    TicketStatusUpdateRequest,
)
from backend.app.services.ticket_service import TicketService

logger = logging.getLogger("routes.tickets")

router = APIRouter(prefix="/api/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, status_code=status.HTTP_201_CREATED)
def create_ticket(
    payload: TicketCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TicketResponse:
    """Creates a new support ticket.

    tenant_id and created_by are set automatically from authenticated user context.
    """
    ticket = TicketService.create_ticket(
        db=db,
        tenant_id=current_user.tenant_id,
        creator=current_user,
        payload=payload,
    )
    return TicketService.to_response(db, ticket)


@router.get("", response_model=TicketListResponse)
def list_all_tickets(
    status_filter: Optional[str] = Query(None, alias="status"),
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> TicketListResponse:
    """Lists all tickets for the organization (HR only)."""
    tickets = TicketService.list_tickets_for_tenant(
        db=db,
        tenant_id=current_hr.tenant_id,
        status_filter=status_filter,
    )
    responses = [TicketService.to_response(db, t) for t in tickets]
    return TicketListResponse(total=len(responses), tickets=responses)


@router.get("/me", response_model=TicketListResponse)
def list_my_tickets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TicketListResponse:
    """Lists only tickets created by the authenticated employee."""
    tickets = TicketService.list_tickets_for_user(
        db=db,
        tenant_id=current_user.tenant_id,
        user_id=current_user.id,
    )
    responses = [TicketService.to_response(db, t) for t in tickets]
    return TicketListResponse(total=len(responses), tickets=responses)


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket_details(
    ticket_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TicketResponse:
    """Retrieves ticket details.

    Allowed for HR administrators or the employee who created the ticket.
    """
    ticket = TicketService.get_ticket(
        db=db,
        tenant_id=current_user.tenant_id,
        ticket_id=ticket_id,
    )
    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Ticket not found in organization",
        )

    if current_user.role != "HR" and ticket.created_by != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: You cannot view another employee's ticket",
        )

    return TicketService.to_response(db, ticket)


@router.patch("/{ticket_id}/status", response_model=TicketResponse)
def update_ticket_status(
    ticket_id: uuid.UUID,
    payload: TicketStatusUpdateRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> TicketResponse:
    """Updates a ticket's status and adds resolution notes (HR only)."""
    ticket = TicketService.update_ticket_status(
        db=db,
        tenant_id=current_hr.tenant_id,
        ticket_id=ticket_id,
        new_status=payload.status,
        resolution_notes=payload.resolution_notes,
        actor=current_hr,
    )
    return TicketService.to_response(db, ticket)


@router.post("/{ticket_id}/answer", response_model=TicketResponse)
def provide_ticket_answer(
    ticket_id: uuid.UUID,
    payload: TicketAnswerRequest,
    current_hr: User = Depends(require_hr),
    db: Session = Depends(get_db),
) -> TicketResponse:
    """HR provides an official answer to an open ticket.

    - Saves the answer on the ticket and marks it RESOLVED.
    - Ingests the Q&A as a new knowledge chunk into the RAG knowledge base
      so the AI assistant can use this answer for future similar questions.
    """
    logger.info(
        "HR %s providing answer for ticket %s (ingest_to_kb=%s)",
        current_hr.id, ticket_id, payload.ingest_to_kb,
    )
    ticket = TicketService.provide_answer(
        db=db,
        tenant_id=current_hr.tenant_id,
        ticket_id=ticket_id,
        payload=payload,
        actor=current_hr,
    )
    return TicketService.to_response(db, ticket)
