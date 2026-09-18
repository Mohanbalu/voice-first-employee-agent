"""Scheduling Agent Node — LangGraph Module.

Parses natural language scheduling requests, handles ambiguity clarification,
and executes schedule creation via SchedulingService.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from backend.app.agents.agent_state import AgentState
from backend.app.schemas.schedule import ScheduleCreate
from backend.app.services.scheduling_service import SchedulingService
from backend.app.utils.nl_time_parser import parse_schedule_request

logger = logging.getLogger("agents.scheduling")


class SchedulingAgentNode:
    """LangGraph node: handles employee scheduling, reminders, and recurring tasks."""

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        request = state.get("request") or state.get("raw_query") or ""
        tenant_id_str = state.get("tenant_id") or "00000000-0000-0000-0000-000000000001"
        db_session = state.get("db_session")
        current_user = state.get("current_user")

        # 1. Parse natural language time & recurrence
        parsed = parse_schedule_request(request)

        # 2. Check for ambiguity
        if parsed.is_ambiguous:
            question = parsed.clarification_question or "Could you clarify what date and time you'd like me to schedule this for?"
            logger.info("Scheduling request is ambiguous: %r -> %s", request, question)
            return {
                "agent_mode": "clarify",
                "requires_clarification": True,
                "clarification_question": question,
                "final_response": question,
            }

        # 3. Format human-friendly time description
        time_str = ""
        if parsed.scheduled_at:
            # Format nicely e.g. "tomorrow at 10:00 AM" or "Friday, Sep 18 at 5:00 PM"
            time_str = parsed.scheduled_at.strftime("%A, %b %d at %I:%M %p")

        recurrence_desc = ""
        if parsed.recurrence_type == "DAILY":
            recurrence_desc = " (repeats daily)"
        elif parsed.recurrence_type == "WEEKLY":
            recurrence_desc = f" (repeats weekly)"
        elif parsed.recurrence_type == "WEEKDAYS":
            recurrence_desc = " (repeats every weekday)"
        elif parsed.recurrence_type == "MONTHLY":
            recurrence_desc = " (repeats monthly)"

        confirmation_message = (
            f"I've scheduled a reminder for '{parsed.title}' on {time_str}{recurrence_desc}."
        )

        schedule_dict = {
            "tool": "scheduling_system",
            "intent": "scheduling",
            "status": "scheduled",
            "title": parsed.title,
            "scheduled_at": parsed.scheduled_at.isoformat() if parsed.scheduled_at else None,
            "timezone": parsed.timezone_name,
            "recurrence_type": parsed.recurrence_type,
            "recurrence_rule": parsed.recurrence_rule,
            "reminder_type": parsed.reminder_type,
            "message": confirmation_message,
        }

        # 4. Resolve user & tenant, and persist schedule
        try:
            t_id = uuid.UUID(str(tenant_id_str or "00000000-0000-0000-0000-000000000001"))
        except Exception:
            t_id = uuid.UUID("00000000-0000-0000-0000-000000000001")

        # Fallback to Siddhartha if no active user context
        user_id = uuid.UUID("00000000-0000-0000-0000-000000000006")
        if current_user:
            if hasattr(current_user, "id") and current_user.id:
                user_id = current_user.id
            elif isinstance(current_user, dict):
                raw_u = current_user.get("id") or current_user.get("user_id")
                if raw_u:
                    try:
                        user_id = uuid.UUID(str(raw_u))
                    except Exception:
                        pass
            else:
                try:
                    user_id = uuid.UUID(str(current_user))
                except Exception:
                    pass

        try:
            service = SchedulingService(db=db_session)
            create_payload = ScheduleCreate(
                title=parsed.title,
                scheduled_at=parsed.scheduled_at,
                timezone=parsed.timezone_name,
                recurrence_type=parsed.recurrence_type,
                recurrence_rule=parsed.recurrence_rule,
                reminder_type=parsed.reminder_type,
                duration_minutes=parsed.duration_minutes,
            )
            created = service.create_schedule(
                tenant_id=t_id,
                user_id=user_id,
                data=create_payload,
            )
            schedule_dict["id"] = str(created.id)
            schedule_dict["status"] = created.status
            logger.info("Persisted schedule %s in SchedulingService for user %s", created.id, user_id)
        except Exception as exc:
            logger.error("Failed to persist schedule: %s", exc)

        return {
            "agent_mode": "tool_intent",
            "tool_intents": [schedule_dict],
            "schedule_created": schedule_dict,
            "final_response": confirmation_message,
        }
