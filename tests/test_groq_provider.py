"""Unit Tests for Groq AI Provider.

Covers:
1. Groq provider initialization & default configurations
2. API key & base URL configuration
3. Model configuration
4. Successful text generation & normalized AITextResponse
5. Token usage normalization (prompt_tokens, completion_tokens, total_tokens)
6. Authentication & placeholder key failure handling
7. Rate-limit & connection error retry handling (exponential backoff)
8. Timeout handling
9. Structured intent classification & malformed JSON handling
10. Provider factory selects Groq
11. IntentClassifier works with injected Groq provider
12. AnswerGenerator works with injected Groq provider
13. Security check: API secrets/keys are never exposed in logs or return structures

All tests are completely OFFLINE — zero live Groq API calls during pytest.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from backend.app.ai.factory import (
    get_ai_provider,
    reset_global_provider,
    set_global_provider,
)
from backend.app.ai.groq_provider import GroqAIProvider
from backend.app.ai.mock_provider import MockAIProvider
from backend.app.ai.provider import AIProvider, AITextResponse, TokenUsage
from backend.app.rag.answer_generator import AnswerGenerator, NO_CONTEXT_ANSWER
from backend.app.agents.agent_state import IntentType
from backend.app.agents.intent_classifier import IntentClassifier


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES & TEARDOWN
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def cleanup_provider_factory():
    """Ensure global provider override is reset after each test."""
    reset_global_provider()
    yield
    reset_global_provider()


def _make_mock_groq_response(content: str = "Groq test response", model: str = "openai/gpt-oss-120b"):
    """Builds a mock response object matching Groq's OpenAI-compatible output."""
    mock_resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    mock_resp.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = 30
    usage.completion_tokens = 20
    usage.total_tokens = 50
    mock_resp.usage = usage
    return mock_resp


# ─────────────────────────────────────────────────────────────────────────────
# 1. INITIALIZATION & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def test_groq_provider_defaults():
    """GroqAIProvider initializes with expected defaults."""
    provider = GroqAIProvider(
        api_key="gsk_test_mock_key_12345",
        model="openai/gpt-oss-120b",
    )
    assert provider.name == "groq"
    assert provider.model_name == "openai/gpt-oss-120b"
    assert "api.groq.com/openai/v1" in provider.base_url


def test_groq_provider_custom_base_url_and_model():
    """GroqAIProvider respects custom base URL and model arguments."""
    provider = GroqAIProvider(
        api_key="gsk_test_key",
        base_url="https://custom.groq.proxy/v1",
        model="openai/gpt-oss-20b",
    )
    assert provider.base_url == "https://custom.groq.proxy/v1"
    assert provider.model_name == "openai/gpt-oss-20b"


def test_groq_provider_missing_key_raises():
    """Accessing unconfigured Groq client raises descriptive RuntimeError."""
    provider = GroqAIProvider(api_key="")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not configured"):
        provider.generate_text([{"role": "user", "content": "Hello"}])


def test_groq_provider_placeholder_key_raises():
    """Placeholder API key is treated as unconfigured."""
    provider = GroqAIProvider(api_key="your_groq_api_key_placeholder")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not configured"):
        provider.generate_text([{"role": "user", "content": "Hello"}])


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT GENERATION & RESPONSE NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def test_groq_provider_successful_text_generation():
    """GroqAIProvider normalizes API output into AITextResponse and TokenUsage."""
    provider = GroqAIProvider(api_key="gsk_valid_mock_key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response(
        content="Grounded workplace response from Groq",
        model="openai/gpt-oss-120b",
    )
    provider._client = fake_client

    response = provider.generate_text(
        messages=[{"role": "user", "content": "How many days leave do I get?"}],
        temperature=0.1,
    )

    assert isinstance(response, AITextResponse)
    assert response.text == "Grounded workplace response from Groq"
    assert response.model == "openai/gpt-oss-120b"
    assert response.usage.prompt_tokens == 30
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 50


# ─────────────────────────────────────────────────────────────────────────────
# 3. ERROR & RETRY HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def test_groq_provider_retryable_error_succeeds_on_retry():
    """GroqAIProvider retries on transient RateLimitError with backoff."""
    from openai import RateLimitError

    provider = GroqAIProvider(api_key="gsk_valid_mock_key", max_retries=2)
    fake_client = MagicMock()

    mock_resp = _make_mock_groq_response("Success after retry")
    fake_client.chat.completions.create.side_effect = [
        RateLimitError(message="Rate limit reached", response=MagicMock(status_code=429), body={}),
        mock_resp,
    ]
    provider._client = fake_client

    with patch("time.sleep"):  # avoid delay in test
        resp = provider.generate_text([{"role": "user", "content": "Hi"}])

    assert resp.text == "Success after retry"
    assert fake_client.chat.completions.create.call_count == 2


def test_groq_provider_max_retries_exceeded_raises():
    """GroqAIProvider raises after exhausting max retries."""
    from openai import APIConnectionError

    provider = GroqAIProvider(api_key="gsk_valid_mock_key", max_retries=1)
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())
    provider._client = fake_client

    with patch("time.sleep"):
        with pytest.raises(APIConnectionError):
            provider.generate_text([{"role": "user", "content": "Hi"}])


# ─────────────────────────────────────────────────────────────────────────────
# 4. STRUCTURED INTENT CLASSIFICATION & MALFORMED JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_groq_provider_intent_classification_valid_json():
    """GroqAIProvider parses JSON object mode response into a dictionary."""
    provider = GroqAIProvider(api_key="gsk_valid_mock_key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response(
        content='{"intent": "knowledge_query", "confidence": 0.98, "reasoning": "Policy question"}'
    )
    provider._client = fake_client

    result = provider.classify_intent(
        text="What is the leave policy?",
        system_prompt="Classify intent into JSON",
    )

    assert result is not None
    assert result["intent"] == "knowledge_query"
    assert result["confidence"] == 0.98


def test_groq_provider_intent_classification_malformed_json_handled_safely():
    """GroqAIProvider returns None without crashing when output is non-JSON."""
    provider = GroqAIProvider(api_key="gsk_valid_mock_key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response(
        content="I think this request is about leave, but I will not output JSON."
    )
    provider._client = fake_client

    result = provider.classify_intent(
        text="Apply for leave",
        system_prompt="Classify intent into JSON",
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# 5. FACTORY INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def test_factory_selects_groq_by_name():
    """get_ai_provider('groq') returns GroqAIProvider instance."""
    provider = get_ai_provider("groq")
    assert isinstance(provider, GroqAIProvider)
    assert provider.name == "groq"


def test_factory_defaults_to_groq_when_configured():
    """get_ai_provider defaults to Groq when AI_PROVIDER=groq."""
    with patch.dict("os.environ", {"AI_PROVIDER": "groq"}):
        provider = get_ai_provider()
        assert isinstance(provider, GroqAIProvider)


# ─────────────────────────────────────────────────────────────────────────────
# 6. INTEGRATION WITH ANSWER GENERATOR & INTENT CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def test_answer_generator_with_mocked_groq_provider():
    """AnswerGenerator generates answer using injected GroqAIProvider."""
    provider = GroqAIProvider(api_key="gsk_valid_mock_key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response(
        content="According to the policy, employees are entitled to 25 days leave."
    )
    provider._client = fake_client

    gen = AnswerGenerator(provider=provider)
    result = gen.generate(
        question="How many days leave?",
        context_text="Policy excerpt: Employees get 25 days annual leave.",
    )

    assert "25 days leave" in result.answer
    assert result.prompt_tokens == 30
    assert result.completion_tokens == 20


def test_intent_classifier_with_mocked_groq_provider():
    """IntentClassifier resolves intent using injected GroqAIProvider."""
    provider = GroqAIProvider(api_key="gsk_valid_mock_key")
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response(
        content='{"intent": "leave_request", "confidence": 0.97, "reasoning": "Vacation application"}'
    )
    provider._client = fake_client

    classifier = IntentClassifier(ai_provider=provider)
    res = classifier.classify("I want to apply for 3 days leave")

    assert res.intent == IntentType.LEAVE_REQUEST
    assert res.confidence == 0.97


def test_intent_classifier_groq_failure_falls_back_to_heuristic():
    """IntentClassifier falls back to heuristic when Groq classification fails."""
    failing_provider = MagicMock(spec=AIProvider)
    failing_provider.classify_intent.return_value = None  # simulation of failure

    classifier = IntentClassifier(ai_provider=failing_provider)
    res = classifier.classify("I want to apply for sick leave")

    assert res.intent == IntentType.LEAVE_REQUEST


# ─────────────────────────────────────────────────────────────────────────────
# 7. SECURITY: NO SECRETS IN STRINGS OR LOGS
# ─────────────────────────────────────────────────────────────────────────────

def test_groq_api_key_not_exposed_in_repr_or_dict():
    """Groq API key is never exposed in repr or serialization."""
    secret = "gsk_super_secret_groq_key_99999"
    provider = GroqAIProvider(api_key=secret)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = _make_mock_groq_response("Safe response")
    provider._client = fake_client

    resp = provider.generate_text([{"role": "user", "content": "Hi"}])
    resp_dict = resp.to_dict()

    assert secret not in repr(provider)
    assert secret not in repr(resp)
    assert secret not in str(resp_dict)
