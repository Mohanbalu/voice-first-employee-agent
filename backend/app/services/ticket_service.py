"""Ticket Service — Production ticket lifecycle and numbering.

Generates collision-safe, human-readable ticket numbers:
HCL-{CATEGORY}-{SEQUENCE:06d}
Maintains strict multi-tenant isolation and auditable lifecycle transitions.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.models.audit import AuditLog
from backend.app.models.employee import Employee
from backend.app.models.ticket import Ticket
from backend.app.models.user import User
from backend.app.schemas.ticket import (
    TicketAnswerRequest,
    TicketCreateRequest,
    TicketResponse,
    VALID_CATEGORIES,
    VALID_PRIORITIES,
    VALID_STATUSES,
)
from backend.app.services.kb_ingest_service import ingest_ticket_answer_to_kb

logger = logging.getLogger("tickets.service")


class TicketService:
    """Service handling support ticket creation, routing, and management."""

    @staticmethod
    def _generate_ticket_number(db: Session, tenant_id: uuid.UUID, category: str) -> str:
        """Generates a human-friendly collision-safe ticket number: HCL-{CAT}-{SEQUENCE:06d}."""
        clean_cat = category.strip().upper()
        count_stmt = select(func.count(Ticket.id)).where(
            Ticket.tenant_id == tenant_id,
            Ticket.category == clean_cat,
        )
        raw_count = db.execute(count_stmt).scalar()
        try:
            current_count = int(raw_count) if raw_count is not None else 0
        except (ValueError, TypeError):
            current_count = 0
        sequence = current_count + 1

        # Check for potential collision (e.g. from deleted tickets or concurrent creation)
        candidate = f"HCL-{clean_cat}-{sequence:06d}"
        exists_stmt = select(Ticket.id).where(
            Ticket.tenant_id == tenant_id,
            Ticket.ticket_number == candidate,
        )
        collision = db.execute(exists_stmt).scalar_one_or_none()
        while collision is not None:
            sequence += 1
            candidate = f"HCL-{clean_cat}-{sequence:06d}"
            collision = db.execute(
                select(Ticket.id).where(
                    Ticket.tenant_id == tenant_id,
                    Ticket.ticket_number == candidate,
                )
            ).scalar_one_or_none()

        return candidate

    @classmethod
    def create_ticket(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        creator: User,
        payload: TicketCreateRequest,
    ) -> Ticket:
        """Creates a new ticket scoped strictly to the authenticated user and tenant."""
        category = payload.category.strip().upper()
        if category not in VALID_CATEGORIES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category: {category}",
            )

        ticket_number = cls._generate_ticket_number(db, tenant_id, category)
        now = datetime.now(timezone.utc)

        ticket = Ticket(
            id=uuid.uuid4(),
            ticket_number=ticket_number,
            tenant_id=tenant_id,
            created_by=creator.id,
            category=category,
            subject=payload.subject.strip(),
            description=payload.description.strip(),
            priority=payload.priority.strip().upper(),
            status="OPEN",
            created_at=now,
            updated_at=now,
        )
        db.add(ticket)
        db.flush()

        audit = AuditLog(
            tenant_id=tenant_id,
            user_id=creator.id,
            action="ticket_created",
            entity_type="ticket",
            entity_id=str(ticket.id),
            details=f"Created ticket {ticket.ticket_number}: {ticket.subject}",
        )
        db.add(audit)

        db.commit()
        db.refresh(ticket)

        logger.info("Created ticket %s for user_id=%s tenant_id=%s", ticket.ticket_number, creator.id, tenant_id)
        return ticket

    @staticmethod
    def get_ticket(
        db: Session,
        tenant_id: uuid.UUID,
        ticket_id: uuid.UUID,
    ) -> Optional[Ticket]:
        """Retrieves a single ticket within tenant boundaries."""
        stmt = select(Ticket).where(
            Ticket.id == ticket_id,
            Ticket.tenant_id == tenant_id,
        )
        return db.execute(stmt).scalar_one_or_none()

    @staticmethod
    def list_tickets_for_tenant(
        db: Session,
        tenant_id: uuid.UUID,
        status_filter: Optional[str] = None,
    ) -> List[Ticket]:
        """Returns tickets across the entire tenant (HR only)."""
        stmt = select(Ticket).where(Ticket.tenant_id == tenant_id)
        if status_filter and status_filter.strip().upper() not in ("ALL", ""):
            stmt = stmt.where(Ticket.status == status_filter.strip().upper())

        stmt = stmt.order_by(Ticket.created_at.desc())
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def list_tickets_for_user(
        db: Session,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> List[Ticket]:
        """Returns only the tickets created by the given employee."""
        stmt = (
            select(Ticket)
            .where(
                Ticket.tenant_id == tenant_id,
                Ticket.created_by == user_id,
            )
            .order_by(Ticket.created_at.desc())
        )
        return list(db.execute(stmt).scalars().all())

    @staticmethod
    def update_ticket_status(
        db: Session,
        tenant_id: uuid.UUID,
        ticket_id: uuid.UUID,
        new_status: str,
        resolution_notes: Optional[str] = None,
        actor: Optional[User] = None,
    ) -> Ticket:
        """Updates ticket status and optional resolution notes with audit tracking."""
        clean_status = new_status.strip().upper()
        if clean_status not in VALID_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {clean_status}",
            )

        ticket = db.execute(
            select(Ticket).where(
                Ticket.id == ticket_id,
                Ticket.tenant_id == tenant_id,
            )
        ).scalar_one_or_none()

        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found in organization",
            )

        old_status = ticket.status
        ticket.status = clean_status
        ticket.updated_at = datetime.now(timezone.utc)

        if resolution_notes:
            ticket.resolution_notes = resolution_notes.strip()

        if clean_status in ("RESOLVED", "CLOSED"):
            ticket.resolved_at = datetime.now(timezone.utc)

        audit = AuditLog(
            tenant_id=tenant_id,
            user_id=actor.id if actor else None,
            action="ticket_status_updated",
            entity_type="ticket",
            entity_id=str(ticket.id),
            details=f"Ticket {ticket.ticket_number} status changed from {old_status} to {clean_status}",
        )
        db.add(audit)

        db.commit()
        db.refresh(ticket)
        return ticket

    @classmethod
    def provide_answer(
        cls,
        db: Session,
        tenant_id: uuid.UUID,
        ticket_id: uuid.UUID,
        payload: TicketAnswerRequest,
        actor: Optional[User] = None,
    ) -> Ticket:
        """HR provides an official answer to a ticket.

        - Stores the answer on the ticket (hr_answer field).
        - Marks the ticket as RESOLVED.
        - Optionally ingests the Q&A pair into the RAG knowledge base.
        """
        ticket = db.execute(
            select(Ticket).where(
                Ticket.id == ticket_id,
                Ticket.tenant_id == tenant_id,
            )
        ).scalar_one_or_none()

        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found in organization",
            )

        old_status = ticket.status
        ticket.hr_answer = payload.answer.strip()
        ticket.status = "RESOLVED"
        ticket.resolution_notes = payload.answer.strip()
        ticket.resolved_at = datetime.now(timezone.utc)
        ticket.updated_at = datetime.now(timezone.utc)

        chunk_id: Optional[str] = None
        if payload.ingest_to_kb:
            try:
                # Build the question from the original ticket subject + description
                question_text = ticket.subject
                if ticket.description and ticket.description.strip() != ticket.subject:
                    question_text = f"{ticket.subject}\n{ticket.description}"

                chunk_id = ingest_ticket_answer_to_kb(
                    db=db,
                    tenant_id=tenant_id,
                    ticket_number=ticket.ticket_number,
                    question=question_text,
                    answer=payload.answer.strip(),
                    category=ticket.category,
                )
                ticket.kb_chunk_id = chunk_id
                logger.info(
                    "Ticket %s answer ingested to KB as chunk_id=%s",
                    ticket.ticket_number, chunk_id,
                )
            except Exception as exc:
                logger.error(
                    "KB ingestion failed for ticket %s: %s — answer saved without KB indexing",
                    ticket.ticket_number, exc,
                )

        audit = AuditLog(
            tenant_id=tenant_id,
            user_id=actor.id if actor else None,
            action="ticket_answered",
            entity_type="ticket",
            entity_id=str(ticket.id),
            details=(
                f"Ticket {ticket.ticket_number} answered by HR (was {old_status}). "
                f"KB ingested: {chunk_id is not None}"
            ),
        )
        db.add(audit)

        db.commit()
        db.refresh(ticket)
        return ticket

    @staticmethod
    def to_response(db: Session, ticket: Ticket) -> TicketResponse:
        """Converts a Ticket model to a TicketResponse, attaching creator's display name."""
        creator_name = None
        if ticket.creator:
            # Check for linked employee
            emp = db.execute(
                select(Employee).where(Employee.user_id == ticket.creator.id)
            ).scalar_one_or_none()
            if emp:
                creator_name = emp.full_name
            else:
                creator_name = ticket.creator.username

        return TicketResponse(
            id=ticket.id,
            ticket_number=ticket.ticket_number,
            tenant_id=ticket.tenant_id,
            created_by=ticket.created_by,
            creator_name=creator_name,
            assigned_to=ticket.assigned_to,
            category=ticket.category,
            subject=ticket.subject,
            description=ticket.description,
            priority=ticket.priority,
            status=ticket.status,
            resolution_notes=ticket.resolution_notes,
            hr_answer=ticket.hr_answer,
            kb_chunk_id=ticket.kb_chunk_id,
            created_at=ticket.created_at,
            updated_at=ticket.updated_at,
            resolved_at=ticket.resolved_at,
        )
