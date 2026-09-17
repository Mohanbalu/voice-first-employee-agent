"""Tests for Agent ↔ Ticket Integration.

Validates:
- Tool execution: create_support_ticket_tool creates tickets in DB with user & tenant context.
- ToolRouterNode integration: creates tickets when user & db session are present.
- Duplicate ticket prevention when ticket_created is already present in graph state.
- Graceful stub fallback when db session or user context is absent.
- Informational/policy queries do not trigger ticket creation.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from backend.app.agents.intent_classifier import IntentType
from backend.app.agents.tool_router import ToolRouterNode
from backend.app.models.ticket import Ticket
from backend.app.models.user import User
from backend.app.tools.hr_tools import create_support_ticket_tool

DEV_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_user() -> User:
    return User(
        id=uuid.uuid4(),
        tenant_id=DEV_TENANT_ID,
        username="56031439",
        password_hash="hash",
        role="EMPLOYEE",
        is_active=True,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestAgentTicketIntegration:
    """Agent tool routing and ticket integration test suite."""

    def test_create_support_ticket_tool_direct(self, mock_user):
        """Validates that create_support_ticket_tool delegates to TicketService and returns structured response."""
        mock_db = MagicMock()
        mock_ticket = Ticket(
            id=uuid.uuid4(),
            ticket_number="HCL-IT-000001",
            tenant_id=DEV_TENANT_ID,
            created_by=mock_user.id,
            category="IT",
            subject="Test monitor issue",
            description="Details of monitor issue",
            priority="HIGH",
            status="OPEN",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        with patch("backend.app.tools.hr_tools.TicketService.create_ticket", return_value=mock_ticket) as mock_create:
            res = create_support_ticket_tool(
                db=mock_db,
                tenant_id=DEV_TENANT_ID,
                creator=mock_user,
                category="IT",
                subject="Test monitor issue",
                description="Details of monitor issue",
                priority="HIGH",
            )

        assert res["status"] == "ticket_created"
        assert res["ticket_number"] == "HCL-IT-000001"
        assert "OPEN" in res["message"]
        mock_create.assert_called_once()

    def test_tool_router_creates_real_ticket_when_context_provided(self, mock_user):
        """Validates ToolRouterNode triggers create_support_ticket_tool when db_session and current_user exist."""
        mock_db = MagicMock()
        router_node = ToolRouterNode()

        mock_tool_res = {
            "status": "ticket_created",
            "ticket_id": str(uuid.uuid4()),
            "ticket_number": "HCL-IT-000002",
            "category": "IT",
            "subject": "IT Support: Helpdesk issue",
            "message": "Your IT support ticket has been created.\nTicket Number: HCL-IT-000002",
        }

        with patch("backend.app.tools.hr_tools.create_support_ticket_tool", return_value=mock_tool_res) as mock_tool:
            state = {
                "intent": "it_support",
                "request": "I have an IT issue, please raise a ticket for my keyboard",
                "db_session": mock_db,
                "current_user": mock_user,
                "tenant_id": DEV_TENANT_ID,
            }
            output = router_node(state)

        assert output["agent_mode"] == "tool_intent"
        assert output["ticket_created"]["ticket_number"] == "HCL-IT-000002"
        assert "HCL-IT-000002" in output["final_response"]
        mock_tool.assert_called_once()

    def test_tool_router_prevents_duplicate_ticket_when_already_created(self, mock_user):
        """Validates ToolRouterNode is idempotent and will not recreate ticket if ticket_created is in state."""
        router_node = ToolRouterNode()
        existing_ticket = {
            "status": "ticket_created",
            "ticket_number": "HCL-IT-000001",
            "message": "Ticket HCL-IT-000001 already logged",
        }

        with patch("backend.app.tools.hr_tools.create_support_ticket_tool") as mock_tool:
            state = {
                "intent": "it_support",
                "request": "Please raise a ticket for my keyboard",
                "db_session": MagicMock(),
                "current_user": mock_user,
                "tenant_id": DEV_TENANT_ID,
                "ticket_created": existing_ticket,
            }
            output = router_node(state)

        mock_tool.assert_not_called()
        assert output["final_response"] == "Ticket HCL-IT-000001 already logged"

    def test_tool_router_fallback_to_stub_without_session(self):
        """Validates ToolRouterNode falls back cleanly to informational stub when DB session is absent."""
        router_node = ToolRouterNode()
        state = {
            "intent": "it_support",
            "request": "How do I request a keyboard?",
        }
        output = router_node(state)

        assert output["agent_mode"] == "tool_intent"
        assert output["tool_intents"][0]["status"] == "stub_pending"
        assert "IT support" in output["final_response"]

    def test_informational_query_does_not_trigger_ticket_creation(self, mock_user):
        """Validates that informational/non-ticket queries do not create tickets."""
        router_node = ToolRouterNode()

        with patch("backend.app.tools.hr_tools.create_support_ticket_tool") as mock_tool:
            state = {
                "intent": "it_support",
                "request": "Where can I find the IT support policies?",
                "db_session": MagicMock(),
                "current_user": mock_user,
                "tenant_id": DEV_TENANT_ID,
            }
            output = router_node(state)

        # "policies" is informational without action triggers like "ticket", "issue", "broken"
        mock_tool.assert_not_called()
        assert output["tool_intents"][0]["status"] == "stub_pending"
