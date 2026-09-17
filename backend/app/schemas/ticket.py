"""Ticket System Pydantic Schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


VALID_CATEGORIES = {"IT", "HR", "FACILITIES", "GENERAL"}
VALID_PRIORITIES = {"LOW", "MEDIUM", "HIGH", "URGENT"}
VALID_STATUSES = {"OPEN", "IN_PROGRESS", "WAITING_FOR_EMPLOYEE", "RESOLVED", "CLOSED"}


class TicketCreateRequest(BaseModel):
    """Payload for creating a new support ticket."""

    category: str = Field(..., description="IT, HR, FACILITIES, or GENERAL")
    subject: str = Field(..., min_length=3, max_length=255, description="Brief description of the issue")
    description: str = Field(..., min_length=5, description="Full details of the request")
    priority: str = Field("MEDIUM", description="LOW, MEDIUM, HIGH, or URGENT")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in VALID_CATEGORIES:
            raise ValueError(f"Category must be one of: {', '.join(sorted(VALID_CATEGORIES))}")
        return upper

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in VALID_PRIORITIES:
            raise ValueError(f"Priority must be one of: {', '.join(sorted(VALID_PRIORITIES))}")
        return upper


class TicketStatusUpdateRequest(BaseModel):
    """Payload for HR updating a ticket status and resolution notes."""

    status: str = Field(..., description="OPEN, IN_PROGRESS, WAITING_FOR_EMPLOYEE, RESOLVED, or CLOSED")
    resolution_notes: Optional[str] = Field(None, description="Action taken or response notes")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        upper = v.strip().upper()
        if upper not in VALID_STATUSES:
            raise ValueError(f"Status must be one of: {', '.join(sorted(VALID_STATUSES))}")
        return upper


class TicketAnswerRequest(BaseModel):
    """HR submits an official answer that gets stored on the ticket AND ingested into the RAG knowledge base."""

    answer: str = Field(..., min_length=10, description="HR's official resolution answer")
    ingest_to_kb: bool = Field(True, description="Whether to add this Q&A to the RAG knowledge base")


class TicketResponse(BaseModel):
    """Safe ticket representation returned to employees and HR."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ticket_number: str
    tenant_id: uuid.UUID
    created_by: uuid.UUID
    creator_name: Optional[str] = None
    assigned_to: Optional[uuid.UUID] = None
    category: str
    subject: str
    description: str
    priority: str
    status: str
    resolution_notes: Optional[str] = None
    hr_answer: Optional[str] = None
    kb_chunk_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class TicketListResponse(BaseModel):
    """List of tickets with total counter."""

    total: int
    tickets: List[TicketResponse]
