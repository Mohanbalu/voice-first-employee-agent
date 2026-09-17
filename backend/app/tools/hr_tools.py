"""Workplace & IT Tools — Integration boundary for LangGraph Agent.

Keeps database and business logic decoupled from agent graph nodes.
Agent -> Tool -> Service -> Database.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.app.models.ticket import Ticket
from backend.app.models.user import User
from backend.app.schemas.ticket import TicketCreateRequest
from backend.app.services.ticket_service import TicketService

logger = logging.getLogger("tools.hr_it")


def create_support_ticket_tool(
    db: Session,
    tenant_id: uuid.UUID,
    creator: User,
    category: str,
    subject: str,
    description: str,
    priority: str = "MEDIUM",
) -> Dict[str, Any]:
    """Tool function invoked by the agent to create a support ticket.

    Returns structured dict with ticket details and human-friendly response.
    """
    payload = TicketCreateRequest(
        category=category,
        subject=subject,
        description=description,
        priority=priority,
    )
    ticket = TicketService.create_ticket(
        db=db,
        tenant_id=tenant_id,
        creator=creator,
        payload=payload,
    )

    logger.info("Tool created ticket %s for user %s", ticket.ticket_number, creator.id)

    return {
        "status": "ticket_created",
        "ticket_id": str(ticket.id),
        "ticket_number": ticket.ticket_number,
        "category": ticket.category,
        "subject": ticket.subject,
        "message": (
            f"Your {ticket.category} support ticket has been created.\n"
            f"Ticket Number: {ticket.ticket_number}\n"
            f"Subject: {ticket.subject}\n"
            f"Status: OPEN"
        ),
    }
