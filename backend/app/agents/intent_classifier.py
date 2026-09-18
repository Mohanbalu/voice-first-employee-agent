"""Intent Classifier — Module 5 & Puter Integration.

Classifies an employee request into one of the IntentType values.

Three-tier classification hierarchy:
1. Injected mock provider (tests — zero external calls).
2. Configured AI provider (Puter AI or OpenAI).
3. Deterministic keyword heuristic (fallback when provider unavailable/unconfigured).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

try:
    from backend.app.agents.agent_state import IntentType
    from backend.app.ai.provider import AIProvider
    from backend.app.ai.factory import get_ai_provider
except ImportError:
    from app.agents.agent_state import IntentType
    from app.ai.provider import AIProvider
    from app.ai.factory import get_ai_provider

logger = logging.getLogger("agents.intent_classifier")

# ── Heuristic keyword maps ────────────────────────────────────────────────────
# Used as fallback when LLM is not available.
# Keywords are lowercased and checked via substring match.

_KEYWORD_MAP: List[tuple[set[str], IntentType]] = [
    # Leave / HR
    ({
        "leave", "annual leave", "sick leave", "holiday", "time off",
        "pto", "vacation", "maternity", "paternity", "absence",
        "days off", "apply for leave",
    }, IntentType.LEAVE_REQUEST),
    # IT Support
    ({
        "it support", "helpdesk", "laptop", "password", "vpn",
        "software", "hardware", "access", "printer", "computer",
        "ticket", "it ticket", "raise a ticket", "tech support",
    }, IntentType.IT_SUPPORT),
    # Scheduling & Reminders
    ({
        "remind me", "reminder", "schedule a reminder", "set a reminder",
        "remind", "set reminder", "create reminder", "my reminders",
        "schedule reminder",
    }, IntentType.SCHEDULING),
    # Tasks
    ({
        "task", "to-do", "todo", "assign task", "action item",
        "my tasks", "project task",
    }, IntentType.TASK_MANAGEMENT),
    # Calendar & Meetings
    ({
        "meeting", "calendar", "book a room", "room booking",
        "appointment", "book a meeting", "event", "invite",
        "schedule a meeting", "schedule meeting",
    }, IntentType.CALENDAR_QUERY),
    # Timesheet
    ({
        "timesheet", "log hours", "hours worked", "clock in",
        "clock out", "attendance", "time entry", "overtime",
    }, IntentType.TIMESHEET),
    # Navigation & GPS
    ({
        "where is", "where are", "where can i find", "which floor", "find room", "room location", "floor", "navigate",
        "directions", "office map", "how to get to", "building", "location", "located",
        "nearest", "toilet", "canteen", "cafeteria", "sdc", "tower 1", "tower 2",
        "techbees", "seminar hall", "odc", "play area", "breakout room", "it team",
        "where am i", "my location", "current location",
    }, IntentType.NAVIGATION),
    # Policy / Knowledge
    ({
        "policy", "policies", "rules", "guideline", "guidelines",
        "procedure", "what is", "how does", "explain", "tell me about",
        "what are", "can i", "am i allowed", "entitle", "regulation",
        "health insurance", "benefit", "wfh", "work from home",
        "remote", "travel", "expense", "reimbursement", "disciplinary",
        "environmental", "anti-bribery", "corruption", "human rights",
        "water", "business policy", "ai", "leadership",
    }, IntentType.KNOWLEDGE_QUERY),
]

_CLARIFY_KEYWORDS: set[str] = {
    "help", "something", "anything", "whatever", "stuff", "thing",
    "not sure", "maybe", "idk", "i don't know",
}

_OUT_OF_SCOPE_KEYWORDS: set[str] = {
    "buy", "sell", "stock", "trade", "invest", "crypto",
    "weather", "sports", "news", "movie", "music", "game",
    "joke", "story", "recipe", "cooking", "health advice",
    "medical advice", "diagnosis",
}

# ── LLM classification prompt ─────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an intent classifier for an enterprise employee assistant.

Classify the employee's message into EXACTLY ONE of these intents:
- knowledge_query: asking about company policy, rules, benefits, procedures, office facilities, building locations
- leave_request: requesting or asking about leave, time off, holidays
- it_support: reporting IT issues, requesting tech help or access (e.g. broken laptop, password reset, VPN issue)
- calendar_query: scheduling meetings, booking rooms, checking calendar
- task_management: creating or querying tasks, action items, to-dos
- scheduling: setting reminders, scheduling future alerts, recurring reminders (e.g. remind me tomorrow at 10 AM, set reminder in 30 minutes, remind me every Monday)
- timesheet: logging hours, checking attendance, overtime queries
- navigation: finding rooms, offices, facilities, cafeteria, floors, teams, or desks in the building (e.g. where is the IT team, where are Techbees classrooms)
- clarify_needed: message is too vague or ambiguous to classify
- out_of_scope: completely unrelated to work or employee assistant

Respond ONLY with valid JSON in this exact format:
{
  "intent": "<one of the intent names above>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<one sentence max>"
}
"""


@dataclass
class IntentClassification:
    """Result of intent classification."""

    intent: IntentType
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
        }


def _heuristic_classify(text: str) -> IntentClassification:
    """
    Deterministic keyword-based intent classification.
    Used as fallback when LLM is unavailable, and as basis for tests.
    """
    lowered = text.lower()

    # Check out-of-scope first
    if any(kw in lowered for kw in _OUT_OF_SCOPE_KEYWORDS):
        return IntentClassification(
            intent=IntentType.OUT_OF_SCOPE,
            confidence=0.80,
            reasoning="Request appears to be outside the scope of the employee assistant.",
        )

    # Check very short/vague messages
    words = lowered.split()
    if len(words) <= 2 or any(kw in lowered for kw in _CLARIFY_KEYWORDS):
        return IntentClassification(
            intent=IntentType.CLARIFY_NEEDED,
            confidence=0.70,
            reasoning="Message is too short or vague to determine intent.",
        )

    # Match keyword categories — pick highest-match category
    best_intent: Optional[IntentType] = None
    best_count = 0

    for keywords, intent in _KEYWORD_MAP:
        match_count = sum(1 for kw in keywords if kw in lowered)
        if match_count > best_count:
            best_count = match_count
            best_intent = intent

    if best_intent is not None and best_count > 0:
        confidence = min(0.75 + 0.05 * best_count, 0.95)
        return IntentClassification(
            intent=best_intent,
            confidence=round(confidence, 3),
            reasoning=f"Matched {best_count} keyword(s) for {best_intent.value}.",
        )

    # Default: clarify if nothing matched
    return IntentClassification(
        intent=IntentType.CLARIFY_NEEDED,
        confidence=0.50,
        reasoning="Could not determine a clear intent from the message.",
    )


class IntentClassifier:
    """
    Intent classifier with injectable provider (for testing and multi-provider AI).

    Usage:
      classifier = IntentClassifier()                           # auto: Puter, OpenAI, or heuristic
      classifier = IntentClassifier(provider=stub)              # injected callable (tests)
      classifier = IntentClassifier(ai_provider=mock_provider)  # injected AIProvider
    """

    def __init__(
        self,
        provider: Optional[Any] = None,
        model: Optional[str] = None,
        ai_provider: Optional[AIProvider] = None,
    ):
        self._provider = provider
        self._model = model or os.getenv("AGENT_INTENT_MODEL", "gpt-4o-mini")
        self._ai_provider = ai_provider

    def classify(self, text: str) -> IntentClassification:
        """Classifies the employee request text into an IntentType."""
        if not text or not text.strip():
            return IntentClassification(
                intent=IntentType.CLARIFY_NEEDED,
                confidence=1.0,
                reasoning="Empty message — cannot classify.",
            )

        text = text.strip()

        # 1. Injected provider (e.g., test stub function)
        if self._provider is not None:
            result = self._provider(text)
            if isinstance(result, IntentClassification):
                return result
            if isinstance(result, dict):
                return IntentClassification(
                    intent=IntentType(result.get("intent", "clarify_needed")),
                    confidence=float(result.get("confidence", 0.80)),
                    reasoning=str(result.get("reasoning", "")),
                )
            raise TypeError(f"Provider must return IntentClassification or dict, got {type(result)}")

        # 2. Injected AIProvider instance
        if self._ai_provider is not None:
            return self._classify_via_provider(self._ai_provider, text)

        # 3. Resolve configured AI Provider (Groq, Puter AI, OpenAI, Mock)
        ai_provider_name = os.getenv("AI_PROVIDER", "groq").strip().lower()

        if ai_provider_name == "groq":
            groq_key = os.getenv("GROQ_API_KEY", "")
            is_placeholder = not groq_key or groq_key == "your_groq_api_key_placeholder"
            if not is_placeholder:
                try:
                    provider = get_ai_provider("groq")
                    return self._classify_via_provider(provider, text)
                except Exception as exc:
                    logger.warning("Groq AI classification failed (%s), using heuristic.", exc)
                    return _heuristic_classify(text)

        elif ai_provider_name == "puter":
            puter_token = os.getenv("PUTER_AUTH_TOKEN", "")
            is_placeholder = not puter_token or puter_token == "your_puter_auth_token_placeholder"
            if not is_placeholder:
                try:
                    provider = get_ai_provider("puter")
                    return self._classify_via_provider(provider, text)
                except Exception as exc:
                    logger.warning("Puter AI classification failed (%s), using heuristic.", exc)
                    return _heuristic_classify(text)

        elif ai_provider_name == "openai":
            openai_key = os.getenv("OPENAI_API_KEY", "")
            is_placeholder = not openai_key or openai_key == "your_openai_api_key_placeholder"
            if not is_placeholder:
                try:
                    provider = get_ai_provider("openai")
                    return self._classify_via_provider(provider, text)
                except Exception as exc:
                    logger.warning("OpenAI classification failed (%s), using heuristic.", exc)
                    return _heuristic_classify(text)

        elif ai_provider_name == "mock":
            try:
                provider = get_ai_provider("mock")
                return self._classify_via_provider(provider, text)
            except Exception as exc:
                logger.warning("Mock AI classification failed (%s), using heuristic.", exc)
                return _heuristic_classify(text)

        # 4. Heuristic fallback when no valid API keys/tokens are configured
        logger.info("No active AI provider credentials — using heuristic intent classifier.")
        return _heuristic_classify(text)

    def _classify_via_provider(self, provider: AIProvider, text: str) -> IntentClassification:
        """Executes classification using an AIProvider with safe JSON fallback."""
        try:
            data = provider.classify_intent(text=text, system_prompt=_SYSTEM_PROMPT)
            if not data or not isinstance(data, dict):
                logger.warning("Provider returned invalid intent payload, using heuristic fallback.")
                return _heuristic_classify(text)

            intent_str = data.get("intent", "clarify_needed")
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.CLARIFY_NEEDED

            return IntentClassification(
                intent=intent,
                confidence=float(data.get("confidence", 0.70)),
                reasoning=str(data.get("reasoning", ""))[:200],
            )
        except Exception as exc:
            logger.warning("AI provider classification error (%s), using heuristic.", exc)
            return _heuristic_classify(text)
