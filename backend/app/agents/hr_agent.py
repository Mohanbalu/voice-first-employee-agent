"""HR Agent — Module 5.

LangGraph node: handles LEAVE_REQUEST intents.

IMPORTANT: This is a SAFE STUB. It does NOT call any real HR system.
Real HR/leave API integration will be implemented in Module 6.

The stub returns a structured ToolIntentResponse that:
1. Acknowledges the employee's intent.
2. Describes exactly what Module 6 will do.
3. Never pretends the action has been taken.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    from backend.app.agents.agent_state import AgentState
except ImportError:
    from app.agents.agent_state import AgentState

logger = logging.getLogger("agents.hr_agent")


def _build_leave_stub_response(request: str) -> Dict[str, Any]:
    """Builds a structured ToolIntentResponse for leave requests."""
    return {
        "tool": "hr_leave_system",
        "intent": "leave_request",
        "status": "stub_pending",
        "message": (
            "I can see you'd like to submit a leave request. "
            "Once the HR integration is available (Module 6), I'll be able to:\n"
            "  • Submit your leave request directly to the HR system\n"
            "  • Check your remaining leave balance\n"
            "  • Notify your manager for approval\n\n"
            "For now, please contact HR directly or use the HR portal."
        ),
        "requires_action": True,
        "action_description": "Submit leave request via HR system",
        "module_planned": "Module 6 — HR Integration",
    }


class HRAgentNode:
    """
    LangGraph node: acknowledges and stubs leave/HR requests.
    """

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        request = state.get("request", "")
        stub = _build_leave_stub_response(request)

        logger.info("HR agent stub called for intent: leave_request")

        return {
            "agent_mode": "tool_intent",
            "tool_intents": [stub],
            "final_response": stub["message"],
        }
