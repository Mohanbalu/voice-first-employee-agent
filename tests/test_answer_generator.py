"""Tests for RAG Answer Generator — Module 4.

All tests mock the OpenAI client. No real API calls are made.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from backend.app.rag.answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
    NO_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_completion(content: str = "You have 20 days of annual leave.", model: str = "gpt-4o-mini"):
    """Creates a fake OpenAI chat completion response object."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.total_tokens = 150
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_generator(api_key: str = "sk-test-key") -> AnswerGenerator:
    return AnswerGenerator(api_key=api_key, model="gpt-4o-mini", timeout=10, max_retries=1)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNoContextBehavior:
    """Generator must NOT call the LLM when context is empty."""

    def test_empty_context_returns_no_match_without_llm(self):
        gen = _make_generator()
        result = gen.generate(question="What is our Mars policy?", context_text="")
        assert result.answer == NO_CONTEXT_ANSWER
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_whitespace_context_returns_no_match_without_llm(self):
        gen = _make_generator()
        result = gen.generate(question="Mars policy?", context_text="   \n  ")
        assert result.answer == NO_CONTEXT_ANSWER

    def test_no_api_key_raises_when_context_present(self):
        gen = AnswerGenerator(api_key="", model="gpt-4o-mini")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            gen.generate(question="Leave policy?", context_text="Employees get 20 days.")

    def test_placeholder_api_key_raises(self):
        gen = AnswerGenerator(api_key="your_openai_api_key_placeholder", model="gpt-4o-mini")
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            gen.generate(question="Leave?", context_text="Employees get 20 days.")


class TestLLMGroundingPrompt:
    """Verifies the grounding system prompt reaches the LLM."""

    def test_system_prompt_sent_to_llm(self):
        gen = _make_generator()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        gen._client = fake_client

        gen.generate(question="Leave?", context_text="Employees get 20 days annual leave.")

        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages") or call_args[0][1]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert "only" in system_messages[0]["content"].lower() or "ONLY" in system_messages[0]["content"]

    def test_context_included_in_user_message(self):
        gen = _make_generator()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        gen._client = fake_client

        context = "Employees are entitled to 20 days annual leave."
        gen.generate(question="Leave policy?", context_text=context)

        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][1]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert context in user_messages[0]["content"]

    def test_question_included_in_user_message(self):
        gen = _make_generator()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        gen._client = fake_client

        question = "How many leave days do I get?"
        gen.generate(question=question, context_text="Context text here.")

        call_args = fake_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[0][1]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert question in user_messages[0]["content"]

    def test_low_temperature_used(self):
        """Temperature should be low (<=0.2) for factual grounding."""
        gen = _make_generator()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        gen._client = fake_client

        gen.generate(question="Leave?", context_text="Context.")

        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        temp = call_kwargs.get("temperature", 1.0)
        assert temp <= 0.2, f"Expected low temperature for grounding, got {temp}"


class TestSuccessfulGeneration:
    """Verifies successful answer generation and token counting."""

    def test_answer_returned_correctly(self):
        gen = _make_generator()
        fake_client = MagicMock()
        expected_answer = "You have 20 days of annual leave."
        fake_client.chat.completions.create.return_value = _fake_completion(content=expected_answer)
        gen._client = fake_client

        result = gen.generate(question="Leave?", context_text="Context.")
        assert result.answer == expected_answer

    def test_token_counts_populated(self):
        gen = _make_generator()
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion()
        gen._client = fake_client

        result = gen.generate(question="Leave?", context_text="Context.")
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150

    def test_model_name_preserved(self):
        gen = AnswerGenerator(api_key="sk-test", model="gpt-4o-mini")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion(model="gpt-4o-mini")
        gen._client = fake_client

        result = gen.generate(question="Leave?", context_text="Context.")
        assert result.model == "gpt-4o-mini"


class TestRetryBehavior:
    """Verifies exponential backoff on transient LLM errors."""

    def test_rate_limit_retried(self):
        from openai import RateLimitError

        gen = AnswerGenerator(api_key="sk-test", model="gpt-4o-mini", max_retries=1)
        fake_client = MagicMock()

        rate_err = RateLimitError.__new__(RateLimitError)
        # First call raises, second succeeds
        fake_client.chat.completions.create.side_effect = [
            rate_err,
            _fake_completion(content="Answer after retry."),
        ]
        gen._client = fake_client

        with patch("backend.app.rag.answer_generator.time.sleep"):
            result = gen.generate(question="Leave?", context_text="Context.")

        assert result.answer == "Answer after retry."
        assert fake_client.chat.completions.create.call_count == 2

    def test_max_retries_exceeded_raises(self):
        from openai import RateLimitError

        gen = AnswerGenerator(api_key="sk-test", model="gpt-4o-mini", max_retries=1)
        fake_client = MagicMock()
        rate_err = RateLimitError.__new__(RateLimitError)
        fake_client.chat.completions.create.side_effect = rate_err
        gen._client = fake_client

        with patch("backend.app.rag.answer_generator.time.sleep"):
            with pytest.raises(RateLimitError):
                gen.generate(question="Leave?", context_text="Context.")


class TestAPIKeySafety:
    """API key must never be leaked in errors or outputs."""

    def test_api_key_not_in_generated_answer(self):
        gen = AnswerGenerator(api_key="sk-supersecretkey", model="gpt-4o-mini")
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = _fake_completion(
            content="Leave policy says 20 days."
        )
        gen._client = fake_client

        result = gen.generate(question="Leave?", context_text="Context.")
        assert "sk-supersecretkey" not in result.answer
        assert "sk-supersecretkey" not in result.model
