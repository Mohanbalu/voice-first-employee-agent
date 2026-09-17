"""RAG Answer Generator — Module 4 & 5.

Calls the configured AI Provider (Puter AI or OpenAI) with a strictly-grounded
system prompt to generate answers based ONLY on the supplied company knowledge context.
Never uses the LLM's general pretrained knowledge as company policy.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from backend.app.ai.provider import AIProvider
    from backend.app.ai.factory import get_ai_provider
except ImportError:
    from app.ai.provider import AIProvider
    from app.ai.factory import get_ai_provider

logger = logging.getLogger("rag.answer_generator")

# ── Configuration ─────────────────────────────────────────────────────────────

LLM_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))

import re

# Strict grounding system prompt — do NOT change without review
SYSTEM_PROMPT = """\
You are an AI assistant for company employees. You answer questions ONLY using \
the company knowledge excerpts provided below.

RULES (mandatory — never violate):
1. Answer ONLY based on the provided [SOURCE] excerpts.
2. Do NOT use your general pretrained knowledge as company policy or facts.
3. If the excerpts do not contain enough information, say:
   "I could not find this information in the available company knowledge base."
4. Do NOT invent policies, page numbers, dates, floors, buildings, or facilities.
5. If multiple sources seem to conflict, present both and note the discrepancy.
6. When answering questions about office locations, facilities, or teams:
   - If multiple buildings/floors match the query (e.g., IT team in multiple buildings), list each location clearly (e.g., SDC: 2nd floor; Tower 1: 1st floor).
   - Distinguish general IT teams from laptop/hardware technical support (e.g., SDC 3rd floor).
   - If a specific building or team's floor is not mentioned or specified in the records (such as Tower 2 experienced teams), explicitly state that the floor or detail is not specified in current company records. Never guess or fabricate a floor.
7. Cite the document name and page range when relevant.
8. Be concise, direct, and well-structured.
9. Do NOT fabricate citations, internal database IDs, technical table names, or provider names.
10. Preserve important conditions, exceptions, and qualifications from the policy.
11. If a question is completely unrelated to the provided excerpts, respond:
    "I could not find this information in the available company knowledge base."
12. FORMATTING RESTRICTION: Do NOT use asterisks (*) or star symbols in your response. Never use asterisks for bolding (e.g., do not write **bold**), italics (*italic*), or bullet points (* bullet). Output clean, readable plain text. For bullet points or lists, use plain dashes (-) or numbers (1., 2.), never asterisks (*).
"""

NO_CONTEXT_ANSWER = (
    "I could not find relevant information for your question in the available "
    "company knowledge base. Please contact HR or your manager for guidance."
)


def clean_markdown_asterisks(text: str) -> str:
    """Removes asterisk markdown symbols (*, **, ***) from text for clean voice and UI display."""
    if not text:
        return ""
    # Convert bullet points starting with * or • to -
    cleaned = re.sub(r"^\s*[*•]\s+", "- ", text, flags=re.MULTILINE)
    # Remove any remaining asterisks
    cleaned = cleaned.replace("*", "")
    return cleaned


@dataclass
class GeneratedAnswer:
    """Structured result from the LLM answer generator."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }


class AnswerGenerator:
    """
    Generates answers from retrieved context using the configured AI Provider.
    - Never exposes API keys/tokens.
    - Bounded retries with exponential backoff.
    - Returns NO_CONTEXT_ANSWER if context is empty (no retrieval results).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = LLM_TIMEOUT,
        max_retries: int = LLM_MAX_RETRIES,
        provider: Optional[AIProvider] = None,
    ):
        self._explicit_api_key = api_key is not None
        self._api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self._model = model or LLM_MODEL
        self._timeout = timeout
        self._max_retries = max_retries
        self._client: Optional[Any] = None
        self._provider: Optional[AIProvider] = provider

    def _get_client(self) -> Any:
        """Lazily initialises the OpenAI client."""
        if self._client is not None:
            return self._client

        if not self._api_key or self._api_key == "your_openai_api_key_placeholder":
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it in backend/.env to enable "
                "LLM answer generation."
            )

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
            ) from exc

        return self._client

    def generate(self, question: str, context_text: str) -> GeneratedAnswer:
        """
        Generates a grounded answer. If context_text is empty, returns the
        NO_CONTEXT_ANSWER without calling the LLM.
        """
        if not context_text or not context_text.strip():
            logger.info("Empty context — returning no-match answer without LLM call.")
            return GeneratedAnswer(
                answer=NO_CONTEXT_ANSWER,
                model=self._model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        user_message = (
            f"COMPANY KNOWLEDGE EXCERPTS:\n\n{context_text}\n\n"
            f"EMPLOYEE QUESTION:\n{question}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # 1. Use injected or configured AIProvider if available
        if self._provider is not None:
            resp = self._provider.generate_text(
                messages=messages,
                temperature=0.1,
            )
            clean_ans = clean_markdown_asterisks(resp.text or NO_CONTEXT_ANSWER)
            return GeneratedAnswer(
                answer=clean_ans,
                model=resp.model,
                prompt_tokens=resp.usage.prompt_tokens,
                completion_tokens=resp.usage.completion_tokens,
                total_tokens=resp.usage.total_tokens,
                error=resp.error,
            )

        # 2. Check if AI_PROVIDER is set to Groq, Puter, or Mock and not explicitly overridden with OpenAI key
        ai_provider_env = os.getenv("AI_PROVIDER", "groq").strip().lower()
        if ai_provider_env in ("groq", "puter", "mock") and not self._explicit_api_key and not (self._client is not None):
            try:
                active_provider = get_ai_provider(ai_provider_env)
                resp = active_provider.generate_text(
                    messages=messages,
                    temperature=0.1,
                )
                clean_ans = clean_markdown_asterisks(resp.text or NO_CONTEXT_ANSWER)
                return GeneratedAnswer(
                    answer=clean_ans,
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                    total_tokens=resp.usage.total_tokens,
                    error=resp.error,
                )
            except Exception as exc:
                logger.error(f"{ai_provider_env.capitalize()} generation failed: %s", exc)
                raise

        # 3. Direct OpenAI client execution (backward-compatible with existing tests)
        client = self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries + 1):
            try:
                from openai import RateLimitError, APIConnectionError, InternalServerError

                response = client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=0.1,  # low temperature for factual grounding
                )
                usage = response.usage
                raw_ans = response.choices[0].message.content or NO_CONTEXT_ANSWER
                clean_ans = clean_markdown_asterisks(raw_ans)

                logger.info(
                    "LLM answer generated. model=%s tokens=%d",
                    self._model,
                    usage.total_tokens if usage else 0,
                )

                return GeneratedAnswer(
                    answer=clean_ans,
                    model=self._model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )

            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    logger.error("LLM max retries exceeded: %s", exc)
                    raise
                backoff = (2 ** attempt) * 1.5
                logger.warning(
                    "Retryable LLM error (%s). Backing off %.1fs (attempt %d/%d)...",
                    exc.__class__.__name__, backoff, attempt + 1, self._max_retries,
                )
                time.sleep(backoff)

            except Exception as exc:
                logger.error("Non-retryable LLM error: %s", exc)
                raise

        if last_error:
            raise last_error

        return GeneratedAnswer(
            answer=NO_CONTEXT_ANSWER,
            model=self._model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        )
