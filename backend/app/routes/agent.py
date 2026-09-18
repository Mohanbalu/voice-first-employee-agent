"""Agent Route — Module 5.

POST /api/agent — unified entry point for the LangGraph orchestrator.

⚠ DEVELOPMENT NOTE:
tenant_id is currently accepted from the request body for development and testing.
Module 8 (Authentication) will replace this with JWT-derived tenant identity.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

try:
    from backend.app.config import config
    from backend.app.database import get_db
    from backend.app.agents.orchestrator import AgentOrchestrator
    from backend.app.agents.agent_state import AgentState
    from backend.app.schemas.agent import AgentRequest, AgentResponse, ToolIntentRef
    from backend.app.rag.answer_generator import clean_markdown_asterisks
    from backend.app.auth.dependencies import get_optional_current_user
    from backend.app.models.user import User
except ImportError:
    from app.config import config
    from app.database import get_db
    from app.agents.orchestrator import AgentOrchestrator
    from app.agents.agent_state import AgentState
    from app.schemas.agent import AgentRequest, AgentResponse, ToolIntentRef
    from app.rag.answer_generator import clean_markdown_asterisks
    from app.auth.dependencies import get_optional_current_user
    from app.models.user import User

logger = logging.getLogger("routes.agent")
router = APIRouter(prefix="/api", tags=["agent"])

# Shared orchestrator (stateless — safe to share across requests)
# NOTE: For tests, the orchestrator is replaced via dependency override.
_orchestrator: Optional[AgentOrchestrator] = None


def _get_orchestrator(db: Session = Depends(get_db)) -> AgentOrchestrator:
    """Returns a per-request orchestrator bound to the current DB session."""
    # We create a new orchestrator per request to bind the DB session.
    # The compiled LangGraph graph is cheap to re-create; heavy resources
    # (models, DB connection pool) are singletons in their respective layers.
    return AgentOrchestrator(db_session=db)


def _resolve_tenant_id(request: AgentRequest) -> str:
    """
    Resolves the tenant UUID from the request.

    Development: use request.tenant_id or fall back to DEV_TENANT_ID.
    Module 8 will replace this with JWT-derived identity.
    """
    raw = request.tenant_id or config.tenant.default_id
    try:
        uuid.UUID(raw)   # validate format
        return raw
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tenant_id: '{raw}'. Must be a valid UUID.",
        )


@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Agent Orchestrator",
    description=(
        "Unified agent endpoint. Classifies the employee's request and routes it to "
        "the appropriate handler: RAG knowledge Q&A, tool intent stub, "
        "clarification, or safe decline.\n\n"
        "**Development note**: `tenant_id` is accepted from the request body until "
        "Module 8 authentication is implemented."
    ),
)
async def agent(
    request: AgentRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    orchestrator: AgentOrchestrator = Depends(_get_orchestrator),
) -> AgentResponse:
    """LangGraph agent orchestrator endpoint."""
    tenant_id = _resolve_tenant_id(request)
    if current_user and str(current_user.tenant_id):
        tenant_id = str(current_user.tenant_id)

    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info(
        "Agent request. tenant=%s user=%s request_len=%d conversation=%s",
        tenant_id,
        getattr(current_user, "username", "anon"),
        len(request.request),
        conversation_id,
    )

    try:
        loop = asyncio.get_event_loop()
        final_state: AgentState = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: orchestrator.run(
                request=request.request,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                current_user=current_user,
            )),
            timeout=25.0,  # Hard 25s cap — Render free tier limit
        )
    except asyncio.TimeoutError:
        logger.error("Orchestrator timed out after 25s for tenant=%s", tenant_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="The assistant took too long to respond. Please try again in a moment.",
        )
    except Exception as exc:
        logger.error("Orchestrator failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent service error. Please try again.",
        )

    # Extract RAG sources from rag_response if present
    rag_sources: List[Dict[str, Any]] = []
    rag_resp = final_state.get("rag_response")
    if rag_resp and isinstance(rag_resp, dict):
        rag_sources = rag_resp.get("sources", [])

    # Convert tool_intents to Pydantic models
    tool_intents: List[ToolIntentRef] = [
        ToolIntentRef(**ti)
        for ti in (final_state.get("tool_intents") or [])
    ]

    raw_response = final_state.get("final_response") or ""
    clean_resp = clean_markdown_asterisks(raw_response)

    # Detect if the assistant was unable to answer or needs human ticket escalation
    agent_mode = final_state.get("agent_mode") or ""
    requires_clarification = bool(final_state.get("requires_clarification") or False)
    suggest_ticket = False
    if agent_mode in ("clarify", "decline") or requires_clarification:
        suggest_ticket = True
    else:
        resp_lower = clean_resp.lower()
        fallback_phrases = [
            "could not find",
            "unable to find",
            "cannot find",
            "not find relevant information",
            "no relevant information",
            "not available in the company knowledge",
            "not mentioned in the available",
            "do not have information",
            "don't have information",
            "not found in the records",
            "not specified in current company records",
            "not specified in the provided records",
            "please raise a ticket",
            "contact it support",
            "contact hr",
            "outside the scope of what i can help with",
        ]
        if any(phrase in resp_lower for phrase in fallback_phrases):
            suggest_ticket = True

    return AgentResponse(
        response=clean_resp,
        agent_mode=agent_mode,
        intent=final_state.get("intent"),
        intent_confidence=final_state.get("intent_confidence") or 0.0,
        requires_clarification=requires_clarification,
        clarification_question=final_state.get("clarification_question"),
        rag_sources=rag_sources,
        tool_intents=tool_intents,
        error=final_state.get("error"),
        suggest_ticket=suggest_ticket,
        schedule_data=final_state.get("schedule_created"),
    )
