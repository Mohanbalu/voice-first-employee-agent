"""Agents Package — Module 5.

LangGraph-based agent orchestrator and node definitions.
"""

from __future__ import annotations

from backend.app.agents.agent_state import (
    AgentState,
    IntentType,
    TOOL_INTENTS,
    make_initial_state,
)
from backend.app.agents.intent_classifier import (
    IntentClassifier,
    IntentClassification,
)
from backend.app.agents.knowledge_agent import KnowledgeAgentNode
from backend.app.agents.hr_agent import HRAgentNode
from backend.app.agents.tool_router import ToolRouterNode
from backend.app.agents.clarify_node import ClarifyNode
from backend.app.agents.decline_node import DeclineNode
from backend.app.agents.orchestrator import AgentOrchestrator

__all__ = [
    "AgentState",
    "IntentType",
    "TOOL_INTENTS",
    "make_initial_state",
    "IntentClassifier",
    "IntentClassification",
    "KnowledgeAgentNode",
    "HRAgentNode",
    "ToolRouterNode",
    "ClarifyNode",
    "DeclineNode",
    "AgentOrchestrator",
]
