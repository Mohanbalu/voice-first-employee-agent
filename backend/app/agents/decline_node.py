"""Decline Node — Module 5.

LangGraph node for when the employee request is outside the assistant's scope.
Returns a polite, honest refusal — never pretends capability it doesn't have.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

try:
    from backend.app.agents.agent_state import AgentState
except ImportError:
    from app.agents.agent_state import AgentState

logger = logging.getLogger("agents.decline_node")

_DECLINE_RESPONSE = (
    "I'm sorry, that request is outside the scope of what I can help with as an "
    "employee workplace assistant. I'm designed to help with company policies, "
    "IT support, leave requests, scheduling, and other workplace topics.\n\n"
    "If you have a workplace-related question, please feel free to ask!"
)


class DeclineNode:
    """
    LangGraph node: politely declines out-of-scope requests.

    Never pretends capability. Never attempts to answer out-of-scope topics.
    Custom response injectable for deterministic testing.
    """

    def __init__(self, custom_response: str = ""):
        self._custom_response = custom_response

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Node callable — writes decline response to state."""
        response = self._custom_response or _DECLINE_RESPONSE
        intent = state.get("intent", "out_of_scope")
        logger.info("Decline node triggered for intent: %s", intent)

        return {
            "agent_mode": "decline",
            "final_response": response,
            "requires_clarification": False,
        }
