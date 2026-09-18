"""Tool Router — Module 5.

Routes non-RAG tool intents to the appropriate stub handler.
All handlers are SAFE STUBS — no real tool execution in Module 5.

Supported intents (all stubbed):
  LEAVE_REQUEST     → HRAgentNode
  IT_SUPPORT        → IT support stub
  CALENDAR_QUERY    → Calendar stub
  TASK_MANAGEMENT   → Task stub
  TIMESHEET         → Timesheet stub
  NAVIGATION        → Navigation stub
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    from backend.app.agents.agent_state import AgentState, IntentType
    from backend.app.agents.hr_agent import HRAgentNode
    from backend.app.agents.scheduling_agent import SchedulingAgentNode
except ImportError:
    from app.agents.agent_state import AgentState, IntentType
    from app.agents.hr_agent import HRAgentNode
    from app.agents.scheduling_agent import SchedulingAgentNode

logger = logging.getLogger("agents.tool_router")


# ── Stub response builders ────────────────────────────────────────────────────

def _it_stub(request: str) -> Dict[str, Any]:
    return {
        "tool": "it_helpdesk_system",
        "intent": "it_support",
        "status": "stub_pending",
        "message": (
            "I can see you need IT support. "
            "Once the IT helpdesk integration is available (Module 6), I'll be able to:\n"
            "  • Raise a helpdesk ticket automatically\n"
            "  • Check the status of your existing tickets\n"
            "  • Escalate urgent issues\n\n"
            "For now, please contact the IT helpdesk directly or use the IT portal."
        ),
        "requires_action": True,
        "action_description": "Raise IT support ticket",
        "module_planned": "Module 6 — IT Integration",
    }


def _calendar_stub(request: str) -> Dict[str, Any]:
    return {
        "tool": "calendar_system",
        "intent": "calendar_query",
        "status": "stub_pending",
        "message": (
            "I can see you'd like to manage your calendar. "
            "Once the calendar integration is available (Module 6), I'll be able to:\n"
            "  • Schedule and book meetings\n"
            "  • Check room availability\n"
            "  • Send calendar invites\n\n"
            "For now, please use your calendar application directly."
        ),
        "requires_action": True,
        "action_description": "Manage calendar / book meeting",
        "module_planned": "Module 6 — Calendar Integration",
    }


def _task_stub(request: str) -> Dict[str, Any]:
    return {
        "tool": "task_management_system",
        "intent": "task_management",
        "status": "stub_pending",
        "message": (
            "I can see you'd like to manage tasks. "
            "Once the task integration is available (Module 6), I'll be able to:\n"
            "  • Create and assign tasks\n"
            "  • View your task list\n"
            "  • Update task status\n\n"
            "For now, please use your task management application directly."
        ),
        "requires_action": True,
        "action_description": "Manage tasks / action items",
        "module_planned": "Module 6 — Task Integration",
    }


def _timesheet_stub(request: str) -> Dict[str, Any]:
    return {
        "tool": "timesheet_system",
        "intent": "timesheet",
        "status": "stub_pending",
        "message": (
            "I can see you'd like to manage your timesheet. "
            "Once the timesheet integration is available (Module 6), I'll be able to:\n"
            "  • Log your working hours\n"
            "  • View attendance records\n"
            "  • Submit timesheets for approval\n\n"
            "For now, please use the timesheet system directly."
        ),
        "requires_action": True,
        "action_description": "Log / manage timesheet hours",
        "module_planned": "Module 6 — Timesheet Integration",
    }


def _navigation_stub(request: str) -> Dict[str, Any]:
    return {
        "tool": "indoor_navigation_system",
        "intent": "navigation",
        "status": "stub_pending",
        "message": (
            "I can see you'd like directions or room information. "
            "Once the indoor navigation integration is available (Module 6), I'll be able to:\n"
            "  • Provide step-by-step directions to rooms and facilities\n"
            "  • Show floor maps\n"
            "  • Locate colleagues' desks\n\n"
            "For now, please refer to the office map on the company intranet."
        ),
        "requires_action": True,
        "action_description": "Indoor navigation / room finding",
        "module_planned": "Module 6 — Navigation Integration",
    }


_TOOL_STUBS = {
    IntentType.IT_SUPPORT: _it_stub,
    IntentType.CALENDAR_QUERY: _calendar_stub,
    IntentType.TASK_MANAGEMENT: _task_stub,
    IntentType.TIMESHEET: _timesheet_stub,
    IntentType.NAVIGATION: _navigation_stub,
}


class ToolRouterNode:
    """
    LangGraph node: routes tool intents to the correct stub handler.
    LEAVE_REQUEST is routed to HRAgentNode.
    All other tool intents are handled by inline stubs.
    """

    def __init__(self):
        self._hr_node = HRAgentNode()
        self._scheduling_node = SchedulingAgentNode()

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        intent_str = state.get("intent", "")
        request = state.get("request", "")

        try:
            intent = IntentType(intent_str)
        except ValueError:
            logger.warning("Unknown tool intent: %s", intent_str)
            return {
                "agent_mode": "tool_intent",
                "tool_intents": [],
                "final_response": (
                    "I recognised your request but couldn't determine the right tool. "
                    "Please try rephrasing or contact support."
                ),
                "error": f"Unknown tool intent: {intent_str!r}",
            }

        # Route leave requests to dedicated HR node
        if intent == IntentType.LEAVE_REQUEST:
            return self._hr_node(state)

        # Route scheduling & reminder requests directly to SchedulingAgentNode
        if intent == IntentType.SCHEDULING:
            return self._scheduling_node(state)

        # Route task management or calendar requests with timing/meeting expressions to SchedulingAgentNode
        if intent in (IntentType.TASK_MANAGEMENT, IntentType.CALENDAR_QUERY):
            lower_req = request.lower()
            if any(k in lower_req for k in ("remind", "schedule", "meeting", "at ", "tomorrow", "today", "in ", "every ", "due ", "am", "pm", "clock", ":")):
                return self._scheduling_node(state)

        # Real ticket creation integration for IT_SUPPORT when session & user provided
        if intent == IntentType.IT_SUPPORT and state.get("db_session") and state.get("current_user"):
            if state.get("ticket_created"):
                existing = state["ticket_created"]
                return {
                    "agent_mode": "tool_intent",
                    "tool_intents": [existing],
                    "final_response": existing.get("message", "Your ticket has already been logged."),
                }
            lower_req = request.lower()
            if any(phrase in lower_req for phrase in ("ticket", "helpdesk", "broken", "issue", "not working", "need support", "need help")):
                try:
                    import uuid
                    tenant_id = uuid.UUID(str(state.get("tenant_id")))
                    user = state["current_user"]
                    session = state["db_session"]
                    from backend.app.tools.hr_tools import create_support_ticket_tool
                    res = create_support_ticket_tool(
                        db=session,
                        tenant_id=tenant_id,
                        creator=user,
                        category="IT",
                        subject=f"IT Support: {request[:60]}",
                        description=request,
                        priority="MEDIUM",
                    )
                    return {
                        "agent_mode": "tool_intent",
                        "tool_intents": [res],
                        "ticket_created": res,
                        "final_response": res["message"],
                    }
                except Exception as exc:
                    logger.error("Ticket tool execution error: %s", exc)

        # All other tool intents handled by stubs
        stub_builder = _TOOL_STUBS.get(intent)
        if stub_builder is None:
            logger.warning("No stub defined for intent: %s", intent_str)
            return {
                "agent_mode": "tool_intent",
                "tool_intents": [],
                "final_response": "I'm not able to handle that request yet. Please check back later.",
                "error": f"No stub for intent: {intent_str}",
            }

        stub = stub_builder(request)
        logger.info("Tool stub called: %s", stub["tool"])

        return {
            "agent_mode": "tool_intent",
            "tool_intents": [stub],
            "final_response": stub["message"],
        }
