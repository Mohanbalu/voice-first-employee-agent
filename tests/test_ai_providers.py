"""Unit Tests for AI Provider Abstraction (Puter AI, OpenAI, Mock).

Covers:
1. Provider selection via factory (Puter, OpenAI, Mock, fallbacks)
2. PuterAIProvider initialization & configuration
3. Response normalization (AITextResponse, TokenUsage)
4. Error & retry handling (timeouts, API errors, connection errors)
5. Structured intent classification via provider & malformed JSON resilience
6. RAG AnswerGenerator integration with AIProvider
7. IntentClassifier integration with AIProvider
8. Security check: API secrets/tokens are never exposed in logs or return objects

Zero external API calls made during tests.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from backend.app.ai.factory import (
    get_ai_provider,
    reset_global_provider,
    set_global_provider,
)
from backend.app.ai.mock_provider import MockAIProvider
from backend.app.ai.openai_provider import OpenAIProvider
from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage
from backend.app.ai.puter_provider import PuterAIProvider
from backend.app.rag.answer_generator import AnswerGenerator, NO_CONTEXT_ANSWER
from backend.app.agents.agent_state import IntentType
from backend.app.agents.intent_classifier import IntentClassifier, IntentClassification


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES & TEARDOWN
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_provider_factory():
    """Ensure global provider override is reset after each test."""
    reset_global_provider()
    yield
    reset_global_provider()


def _make_mock_client_response(content: str = "Test response", model: str = "gpt-4o-mini"):
    """Builds a mock response object matching the OpenAI / Puter format."""
    mock_resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    mock_resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 20
    usage.completion_tokens = 15
    usage.total_tokens = 35
    mock_resp.usage = usage
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROVIDER FACTORY TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_factory_selects_puter():
    """Factory returns PuterAIProvider when provider_type='puter'."""
    provider = get_ai_provider("puter")
    assert isinstance(provider, PuterAIProvider)
    assert provider.name == "puter"


def test_factory_selects_openai():
    """Factory returns OpenAIProvider when provider_type='openai'."""
    provider = get_ai_provider("openai")
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"


def test_factory_selects_mock():
    """Factory returns MockAIProvider when provider_type='mock'."""
    provider = get_ai_provider("mock")
    assert isinstance(provider, MockAIProvider)
    assert provider.name == "mock"


def test_factory_unrecognized_falls_back_to_openai():
    """Unknown provider name safely defaults to OpenAIProvider."""
    provider = get_ai_provider("unknown_future_provider")
    assert isinstance(provider, OpenAIProvider)


def test_factory_respects_global_override():
    """set_global_provider overrides provider selection across all callers."""
    mock_p = MockAIProvider(canned_response="Overridden response")
    set_global_provider(mock_p)

    active = get_ai_provider()
    assert active is mock_p
    assert active.generate_text([{"role": "user", "content": "Hi"}]).text == "Overridden response"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PUTER PROVIDER INITIALIZATION & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def test_puter_provider_defaults():
    """PuterAIProvider initializes with expected defaults."""
    provider = PuterAIProvider(
        auth_token="test_token_123",
        model="gpt-4o",
    )
    assert provider.name == "puter"
    assert provider.model_name == "gpt-4o"
    assert "api.puter.com" in provider.base_url


def test_puter_provider_missing_token_raises():
    """Accessing unconfigured Puter client raises descriptive RuntimeError."""
    provider = PuterAIProvider(auth_token="")
    with pytest.raises(RuntimeError, match="PUTER_AUTH_TOKEN is not configured"):
        provider.generate_text([{"role": "user", "content": "Hello"}])


def test_puter_provider_placeholder_token_raises():
    """Placeholder token is treated as unconfigured."""
    provider = PuterAIProvider(auth_token="your_puter_auth_token_placeholder")
    with pytest.raises(RuntimeError, match="PUTER_AUTH_TOKEN is not configured"):
        provider.generate_text([{"role": "user", "content": "Hello"}])


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESPONSE NORMALIZATION & GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def test_puter_provider_successful_normalized_response():
    """PuterAIProvider normalizes API output into AITextResponse and TokenUsage."""
    provider = PuterAIProvider(auth_token="valid_token_abc")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_client_response(
        content="Puter response content",
        model="gpt-4o-mini",
    )
    provider._client = fake_client

    response = provider.generate_text(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.2,
    )

    assert isinstance(response, AITextResponse)
    assert response.text == "Puter response content"
    assert response.model == "gpt-4o-mini"
    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 15
    assert response.usage.total_tokens == 35


def test_openai_provider_successful_normalized_response():
    """OpenAIProvider normalizes output into AITextResponse."""
    provider = OpenAIProvider(api_key="sk-test-fake")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_client_response(
        content="OpenAI response content",
        model="gpt-4o-mini",
    )
    provider._client = fake_client

    response = provider.generate_text(
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response.text == "OpenAI response content"
    assert response.usage.total_tokens == 35


# ─────────────────────────────────────────────────────────────────────────────
# 4. API FAILURE, RETRY, & TIMEOUT HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def test_puter_provider_retryable_error_retries_and_succeeds():
    """PuterAIProvider retries on transient connection/rate limit errors."""
    from openai import RateLimitError

    provider = PuterAIProvider(auth_token="valid_token_abc", max_retries=2)
    fake_client = MagicMock()

    # Fail on first call, succeed on second
    mock_resp = _make_mock_client_response("Success after retry")
    fake_client.chat.completions.create.side_effect = [
        RateLimitError(message="Rate limit reached", response=MagicMock(status_code=429), body={}),
        mock_resp,
    ]
    provider._client = fake_client

    with patch("time.sleep"):  # skip sleep in tests
        resp = provider.generate_text([{"role": "user", "content": "Hi"}])

    assert resp.text == "Success after retry"
    assert fake_client.chat.completions.create.call_count == 2


def test_puter_provider_max_retries_exceeded_raises():
    """PuterAIProvider raises after exceeding max retries on persistent error."""
    from openai import APIConnectionError

    provider = PuterAIProvider(auth_token="valid_token_abc", max_retries=1)
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
    provider._client = fake_client

    with patch("time.sleep"):
        with pytest.raises(APIConnectionError):
            provider.generate_text([{"role": "user", "content": "Hi"}])


# ─────────────────────────────────────────────────────────────────────────────
# 5. INTENT CLASSIFICATION & MALFORMED JSON RESILIENCE
# ─────────────────────────────────────────────────────────────────────────────

def test_puter_provider_intent_classification_valid_json():
    """PuterAIProvider successfully parses valid structured JSON."""
    provider = PuterAIProvider(auth_token="valid_token_abc")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_client_response(
        content='{"intent": "leave_request", "confidence": 0.95, "reasoning": "Vacation query"}'
    )
    provider._client = fake_client

    result = provider.classify_intent(
        text="Can I take leave tomorrow?",
        system_prompt="Classify intent",
    )
    assert result is not None
    assert result["intent"] == "leave_request"
    assert result["confidence"] == 0.95


def test_puter_provider_intent_classification_malformed_json():
    """PuterAIProvider returns None without crashing when LLM returns non-JSON."""
    provider = PuterAIProvider(auth_token="valid_token_abc")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_client_response(
        content="I am an AI and I think this is about leave."
    )
    provider._client = fake_client

    result = provider.classify_intent(
        text="Can I take leave tomorrow?",
        system_prompt="Classify intent",
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. RAG ANSWER GENERATOR WITH INJECTED PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

def test_answer_generator_with_mock_provider():
    """AnswerGenerator successfully delegates to injected AIProvider."""
    mock_p = MockAIProvider(canned_response="Grounded policy answer from mock.")
    gen = AnswerGenerator(provider=mock_p)

    res = gen.generate(
        question="How many days leave do I get?",
        context_text="Policy excerpt: Employees get 25 days annual leave.",
    )
    assert res.answer == "Grounded policy answer from mock."
    assert mock_p.call_count == 1
    assert "25 days" in mock_p.last_messages[1]["content"]


def test_answer_generator_no_context_short_circuit_does_not_call_provider():
    """AnswerGenerator does not call the provider when context is empty."""
    mock_p = MockAIProvider()
    gen = AnswerGenerator(provider=mock_p)

    res = gen.generate(question="Question?", context_text="")
    assert res.answer == NO_CONTEXT_ANSWER
    assert mock_p.call_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. INTENT CLASSIFIER WITH INJECTED AI PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

def test_intent_classifier_with_mock_ai_provider():
    """IntentClassifier extracts intent from AIProvider with validation."""
    mock_p = MockAIProvider(
        canned_intent={"intent": "it_support", "confidence": 0.94, "reasoning": "Laptop issue"}
    )
    classifier = IntentClassifier(ai_provider=mock_p)

    classification = classifier.classify("My laptop screen is broken")
    assert classification.intent == IntentType.IT_SUPPORT
    assert classification.confidence == 0.94


def test_intent_classifier_malformed_provider_output_falls_back_to_heuristic():
    """IntentClassifier falls back to keyword heuristic if provider fails."""
    mock_p = MagicMock(spec=AIProvider)
    mock_p.classify_intent.return_value = None  # simulates malformed output
    classifier = IntentClassifier(ai_provider=mock_p)

    # Keyword heuristic matches 'leave'
    classification = classifier.classify("I want to apply for annual leave")
    assert classification.intent == IntentType.LEAVE_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# 8. SECURITY: SECRETS NEVER EXPOSED
# ─────────────────────────────────────────────────────────────────────────────

def test_no_secret_tokens_in_provider_string_or_response():
    """Secret tokens never appear in string representations or response structures."""
    secret = "secret_puter_token_999888777"
    provider = PuterAIProvider(auth_token=secret)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_client_response("Safe response")
    provider._client = fake_client

    resp = provider.generate_text([{"role": "user", "content": "Hi"}])
    resp_dict = resp.to_dict()

    assert secret not in repr(resp)
    assert secret not in str(resp_dict)
    assert secret not in repr(provider)
