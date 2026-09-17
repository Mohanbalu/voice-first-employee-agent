"""Knowledge Agent Node — Module 5.

LangGraph node that answers knowledge/policy questions via RAGService.
Wraps Module 4's RAGService and writes the result into AgentState.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

try:
    from backend.app.agents.agent_state import AgentState
    from backend.app.rag.rag_service import RAGService, RAGResponse
    from backend.app.rag.retriever import RetrievalConfig
except ImportError:
    from app.agents.agent_state import AgentState
    from app.rag.rag_service import RAGService, RAGResponse
    from app.rag.retriever import RetrievalConfig

logger = logging.getLogger("agents.knowledge_agent")


class KnowledgeAgentNode:
    """
    LangGraph node: routes KNOWLEDGE_QUERY intents through the RAG pipeline.

    The RAGService is fully injectable — in tests, a mock service is passed in.
    In production, the node creates a RAGService using the global config.
    """

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        session: Optional[Session] = None,
    ):
        self._rag_service = rag_service
        self._session = session  # injected session for tests

    def _get_service(self) -> RAGService:
        if self._rag_service is not None:
            return self._rag_service
        return RAGService(retrieval_config=RetrievalConfig())

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Node callable for LangGraph.
        Returns a partial state dict with the fields this node writes.
        """
        question = state.get("request", "")
        tenant_id_str = state.get("tenant_id", "")

        try:
            tenant_id = uuid.UUID(tenant_id_str)
        except (ValueError, AttributeError):
            logger.error("Invalid tenant_id in state: %r", tenant_id_str)
            return {
                "agent_mode": "rag",
                "final_response": (
                    "I could not find relevant information for your question in the "
                    "available company knowledge base."
                ),
                "error": f"Invalid tenant_id: {tenant_id_str!r}",
                "rag_response": None,
            }

        service = self._get_service()

        try:
            if self._session is not None:
                rag_resp: RAGResponse = service.answer_question(
                    tenant_id=tenant_id,
                    question=question,
                    session=self._session,
                )
            else:
                rag_resp = service.answer_question(
                    tenant_id=tenant_id,
                    question=question,
                )
        except Exception as exc:
            logger.error("RAGService failed in knowledge node: %s", exc)
            return {
                "agent_mode": "rag",
                "final_response": (
                    "I encountered an error while searching the knowledge base. "
                    "Please try again."
                ),
                "error": f"RAG service error: {exc}",
                "rag_response": None,
            }

        logger.info(
            "Knowledge node: confidence=%s chunks=%d",
            rag_resp.confidence,
            rag_resp.used_context_count,
        )

        return {
            "agent_mode": "rag",
            "rag_response": rag_resp.to_dict(),
            "final_response": rag_resp.answer,
            "error": rag_resp.error,
        }
