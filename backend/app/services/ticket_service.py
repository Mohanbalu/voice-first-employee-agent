"""Ticket Service — Production ticket lifecycle and numbering.

Generates collision-safe, human-readable ticket numbers:
HCL-{CATEGORY}-{SEQUENCE:06d}
Maintains strict multi-tenant isolation and auditable lifecycle transitions.
Includes local JSON cache fallback when PostgreSQL database is unreachable.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

try:
    from backend.app.database import is_db_reachable
except ImportError:
    try:
        from app.database import is_db_reachable
    except ImportError:
        def is_db_reachable() -> bool:
            return False

logger = logging.getLogger("tickets.service")

_LOCAL_TICKETS_FILE = Path(__file__).resolve().parents[3] / "data" / "tickets_local.json"
_IN_MEMORY_TICKETS: Dict[str, List[Ticket]] = {}


def _serialize_ticket(t: Ticket) -> Dict[str, Any]:
    return {
        "id": str(t.id),
        "ticket_number": t.ticket_number,
        "tenant_id": str(t.tenant_id),
        "created_by": str(t.created_by),
        "assigned_to": str(t.assigned_to) if t.assigned_to else None,
        "category": t.category,
        "subject": t.subject,
        "description": t.description,
        "priority": t.priority,
        "status": t.status,
        "resolution_notes": t.resolution_notes,
        "hr_answer": t.hr_answer,
        "kb_chunk_id": t.kb_chunk_id,
        "created_at": t.created_at.isoformat() if t.created_at else None,
        "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        "resolved_at": t.resolved_at.isoformat() if t.resolved_at else None,
    }


def _deserialize_ticket(d: Dict[str, Any]) -> Ticket:
    def _parse_dt(v: Optional[str]) -> Optional[datetime]:
        if not v:
            return None
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return None

    return Ticket(
        id=uuid.UUID(d["id"]),
        ticket_number=d["ticket_number"],
        tenant_id=uuid.UUID(d["tenant_id"]),
        created_by=uuid.UUID(d["created_by"]),
        assigned_to=uuid.UUID(d["assigned_to"]) if d.get("assigned_to") else None,
        category=d.get("category") or "IT",
        subject=d.get("subject") or "Support Ticket",
        description=d.get("description") or "",
        priority=d.get("priority") or "MEDIUM",
        status=d.get("status") or "OPEN",
        resolution_notes=d.get("resolution_notes"),
        hr_answer=d.get("hr_answer"),
        kb_chunk_id=d.get("kb_chunk_id"),
        created_at=_parse_dt(d.get("created_at")) or datetime.now(timezone.utc),
        updated_at=_parse_dt(d.get("updated_at")) or datetime.now(timezone.utc),
        resolved_at=_parse_dt(d.get("resolved_at")),
    )


def _load_local_tickets():
    global _IN_MEMORY_TICKETS
    if not _LOCAL_TICKETS_FILE.exists():
        # Seed initial sample tickets for local development
        default_tenant = "00000000-0000-0000-0000-000000000001"
        siddhartha_id = "00000000-0000-0000-0000-000000000006"
        now = datetime.now(timezone.utc).isoformat()
        sample = {
            default_tenant: [
                {
                    "id": str(uuid.uuid4()),
                    "ticket_number": "HCL-IT-000001",
                    "tenant_id": default_tenant,
                    "created_by": siddhartha_id,
                    "assigned_to": None,
                    "category": "IT",
                    "subject": "VPN Access Configuration Request",
                    "description": "Need Cisco AnyConnect profile configured for remote access.",
                    "priority": "HIGH",
                    "status": "OPEN",
                    "resolution_notes": None,
                    "hr_answer": None,
                    "kb_chunk_id": None,
                    "created_at": now,
                    "updated_at": now,
                    "resolved_at": None,
                },
                {
                    "id": str(uuid.uuid4()),
                    "ticket_number": "HCL-HR-000001",
                    "tenant_id": default_tenant,
                    "created_by": siddhartha_id,
                    "assigned_to": None,
                    "category": "HR",
                    "subject": "Annual Leave Calculation Clarification",
                    "description": "Can you verify if remaining earned leaves carry over to next quarter?",
                    "priority": "MEDIUM",
                    "status": "RESOLVED",
                    "resolution_notes": "Leave policy answered by HR team.",
                    "hr_answer": "As per the company leave policy, up to 8 earned leaves carry forward into Q1 automatically.",
                    "kb_chunk_id": None,
                    "created_at": now,
                    "updated_at": now,
                    "resolved_at": now,
                },
            ]
        }
        try:
            _LOCAL_TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LOCAL_TICKETS_FILE.write_text(json.dumps(sample, indent=2), encoding="utf-8")
        except Exception:
            pass

    try:
        raw_text = _LOCAL_TICKETS_FILE.read_text(encoding="utf-8")
        raw_dict = json.loads(raw_text)
        loaded: Dict[str, List[Ticket]] = {}
        for key, items in raw_dict.items():
            loaded[key] = [_deserialize_ticket(it) for it in items]
        _IN_MEMORY_TICKETS = loaded
    except Exception as exc:
        logger.warning("Failed to load local tickets file: %s", exc)


def _save_local_tickets():
    try:
        _LOCAL_TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
        dump_dict: Dict[str, List[Dict[str, Any]]] = {}
        for key, items in _IN_MEMORY_TICKETS.items():
            dump_dict[key] = [_serialize_ticket(it) for it in items]
        _LOCAL_TICKETS_FILE.write_text(json.dumps(dump_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save local tickets file: %s", exc)


_load_local_tickets()


class TicketService:
    """Service handling support ticket creation, routing, and management."""

    @staticmethod
    def _get_local_list(tenant_id: uuid.UUID) -> List[Ticket]:
        _load_local_tickets()
        t_str = str(tenant_id)
        if t_str not in _IN_MEMORY_TICKETS:
            _IN_MEMORY_TICKETS[t_str] = []
        return _IN_MEMORY_TICKETS[t_str]

    @classmethod
    def _generate_ticket_number(cls, db: Optional[Session], tenant_id: uuid.UUID, category: str) -> str:
        """Generates a human-friendly collision-safe ticket number: HCL-{CAT}-{SEQUENCE:06d}."""
        clean_cat = category.strip().upper()
        if db is not None and is_db_reachable():
            try:
                count_stmt = select(func.count(Ticket.id)).where(
                    Ticket.tenant_id == tenant_id,
                    Ticket.category == clean_cat,
                )
                raw_count = db.execute(count_stmt).scalar()
                current_count = int(raw_count) if raw_count is not None else 0
                sequence = current_count + 1
                candidate = f"HCL-{clean_cat}-{sequence:06d}"
                return candidate
            except Exception:
                pass

        # Offline fallback numbering
        local_items = cls._get_local_list(tenant_id)
        cat_count = len([t for t in local_items if t.category == clean_cat])
        return f"HCL-{clean_cat}-{cat_count + 1:06d}"

    @classmethod
    def create_ticket(
        cls,
        db: Optional[Session],
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

        # Save to local store
        cls._get_local_list(tenant_id).insert(0, ticket)
        _save_local_tickets()

        if db is not None and is_db_reachable():
            try:
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
            except Exception as exc:
                logger.warning("DB write failed in create_ticket (%s). Retained in local store.", exc)

        logger.info("Created ticket %s for user_id=%s tenant_id=%s", ticket.ticket_number, creator.id, tenant_id)
        return ticket

    @classmethod
    def get_ticket(
        cls,
        db: Optional[Session],
        tenant_id: uuid.UUID,
        ticket_id: uuid.UUID,
    ) -> Optional[Ticket]:
        """Retrieves a single ticket within tenant boundaries."""
        if db is not None and is_db_reachable():
            try:
                stmt = select(Ticket).where(
                    Ticket.id == ticket_id,
                    Ticket.tenant_id == tenant_id,
                )
                item = db.execute(stmt).scalar_one_or_none()
                if item:
                    return item
            except Exception:
                pass

        for t in cls._get_local_list(tenant_id):
            if t.id == ticket_id:
                return t
        return None

    @classmethod
    def list_tickets_for_tenant(
        cls,
        db: Optional[Session],
        tenant_id: uuid.UUID,
        status_filter: Optional[str] = None,
    ) -> List[Ticket]:
        """Returns tickets across the entire tenant (HR only)."""
        if db is not None and is_db_reachable():
            try:
                stmt = select(Ticket).where(Ticket.tenant_id == tenant_id)
                if status_filter and status_filter.strip().upper() not in ("ALL", ""):
                    stmt = stmt.where(Ticket.status == status_filter.strip().upper())
                stmt = stmt.order_by(Ticket.created_at.desc())
                return list(db.execute(stmt).scalars().all())
            except Exception as exc:
                logger.warning("DB query failed in list_tickets_for_tenant: %s", exc)

        items = cls._get_local_list(tenant_id)
        if status_filter and status_filter.strip().upper() not in ("ALL", ""):
            clean = status_filter.strip().upper()
            return [t for t in items if t.status == clean]
        return list(items)

    @classmethod
    def list_tickets_for_user(
        cls,
        db: Optional[Session],
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> List[Ticket]:
        """Returns only the tickets created by the given employee."""
        if db is not None and is_db_reachable():
            try:
                stmt = (
                    select(Ticket)
                    .where(
                        Ticket.tenant_id == tenant_id,
                        Ticket.created_by == user_id,
                    )
                    .order_by(Ticket.created_at.desc())
                )
                return list(db.execute(stmt).scalars().all())
            except Exception as exc:
                logger.warning("DB query failed in list_tickets_for_user: %s", exc)

        items = cls._get_local_list(tenant_id)
        return [t for t in items if t.created_by == user_id]

    @classmethod
    def update_ticket_status(
        cls,
        db: Optional[Session],
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

        ticket = cls.get_ticket(db, tenant_id, ticket_id)
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

        _save_local_tickets()

        if db is not None and is_db_reachable():
            try:
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
            except Exception as exc:
                logger.warning("DB write failed in update_ticket_status: %s", exc)

        return ticket

    @classmethod
    def answer_ticket(
        cls,
        db: Optional[Session],
        tenant_id: uuid.UUID,
        ticket_id: uuid.UUID,
        payload: TicketAnswerRequest,
        actor: User,
    ) -> Ticket:
        """HR official answer submission on a ticket."""
        ticket = cls.get_ticket(db, tenant_id, ticket_id)
        if ticket is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Ticket not found in organization",
            )

        now = datetime.now(timezone.utc)
        ticket.hr_answer = payload.answer.strip()
        ticket.status = "RESOLVED"
        ticket.resolved_at = now
        ticket.updated_at = now

        _save_local_tickets()

        if db is not None and is_db_reachable():
            try:
                db.commit()
                db.refresh(ticket)
            except Exception as exc:
                logger.warning("DB write failed in answer_ticket: %s", exc)

        return ticket

    @classmethod
    def to_response(cls, db: Optional[Session], ticket: Ticket) -> TicketResponse:
        """Converts a Ticket model to a TicketResponse, attaching creator's display name safely."""
        creator_name = None
        if db is not None and is_db_reachable():
            try:
                emp = db.execute(
                    select(Employee).where(Employee.user_id == ticket.created_by)
                ).scalar_one_or_none()
                if emp:
                    creator_name = emp.full_name
            except Exception:
                pass

        if not creator_name:
            # Check LOCAL_DEV_ACCOUNTS
            from backend.app.services.auth_service import LOCAL_DEV_ACCOUNTS
            for acc in LOCAL_DEV_ACCOUNTS.values():
                if acc["id"] == ticket.created_by:
                    creator_name = acc["name"]
                    break

        if not creator_name:
            creator_name = "Employee"

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
