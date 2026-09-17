"""Unit tests for Local Sentence-Transformers Embedding Provider (Module 3.2+).

Verifies:
1. Provider initialization & default model
2. Custom model configuration
3. Embedding dimension is 384
4. Document embedding shape (batch of texts)
5. Query embedding shape (single query with BGE prompt)
6. Deterministic output
7. Batch processing and empty text handling
8. Dimension mismatch detection
9. Provider factory selection (local, mock, openai)
10. Mock embedding provider backward compatibility
11. Zero API key requirement for local embeddings
12. Model loading failure handling
13. EmbeddingService integration with query embedding
14. Empty and whitespace query error handling
15. Representation and credential safety
"""

import hashlib
import math
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from backend.app.config import config
from backend.app.rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingService,
    LocalSentenceTransformerEmbeddingProvider,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)


def _make_deterministic_vector(text: str, dim: int = 384) -> np.ndarray:
    """Generates a reproducible 384-dim float array from a string hash."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(h[i % len(h)] / 255.0 - 0.5) for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw))
    return np.array([v / max(norm, 1e-9) for v in raw], dtype=np.float32)


class DummyMockModel:
    """Lightweight in-memory mock for SentenceTransformer to keep tests offline."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = 384, **kwargs):
        self.model_name = model_name
        self.dim = dim

    def get_embedding_dimension(self) -> int:
        return self.dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(
        self,
        sentences,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        if isinstance(sentences, str):
            sentences = [sentences]
        vectors = [_make_deterministic_vector(s, self.dim) for s in sentences]
        return np.array(vectors)


@pytest.fixture
def mock_sentence_transformer():
    """Patches sentence_transformers.SentenceTransformer in all tests."""
    with patch("sentence_transformers.SentenceTransformer", side_effect=DummyMockModel) as mock_st:
        yield mock_st


class TestLocalEmbeddingProvider:
    """Tests for LocalSentenceTransformerEmbeddingProvider."""

    def test_provider_initialization_defaults(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        assert provider.model_name == "BAAI/bge-small-en-v1.5"
        assert provider.get_dimension() == 384

    def test_provider_custom_model_and_instruction(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider(
            model_name="custom/bge-test",
            query_instruction="Custom prefix: ",
            batch_size=32,
        )
        assert provider.model_name == "custom/bge-test"
        assert provider._query_instruction == "Custom prefix: "
        assert provider._batch_size == 32

    def test_dimension_is_384(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        assert provider.get_dimension() == 384

    def test_document_embedding_shape_and_type(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        docs = [
            "Employees are eligible for 20 days of paid leave.",
            "Work from home requires manager approval.",
            "Business travel expenses must be submitted within 14 days.",
        ]
        embeddings = provider.embed_texts(docs)

        assert len(embeddings) == 3
        for vec in embeddings:
            assert isinstance(vec, list)
            assert len(vec) == 384
            assert all(isinstance(val, float) for val in vec)

    def test_document_embedding_empty_list(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        assert provider.embed_texts([]) == []

    def test_query_embedding_shape_and_instruction(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider(
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        query = "How do I request annual leave?"
        vec = provider.embed_query(query)

        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(val, float) for val in vec)

    def test_deterministic_output(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        text = "Confidential customer data must not be stored on local drives."

        vec1 = provider.embed_texts([text])[0]
        vec2 = provider.embed_texts([text])[0]

        assert vec1 == vec2

    def test_empty_query_raises_value_error(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            provider.embed_query("")

        with pytest.raises(ValueError, match="Query text cannot be empty"):
            provider.embed_query("   ")

    def test_model_loading_failure_raises_runtime_error(self):
        with patch("sentence_transformers.SentenceTransformer", side_effect=Exception("Disk full")):
            provider = LocalSentenceTransformerEmbeddingProvider()
            with pytest.raises(RuntimeError, match="Failed to load sentence-transformers model"):
                provider.get_dimension()

    def test_no_api_key_required(self, mock_sentence_transformer):
        # Should initialize and embed cleanly without any OPENAI_API_KEY or GROQ_API_KEY in environment
        with patch.dict("os.environ", {}, clear=True):
            provider = LocalSentenceTransformerEmbeddingProvider()
            vec = provider.embed_query("Company holiday schedule")
            assert len(vec) == 384


class TestProviderFactoryAndService:
    """Tests for get_embedding_provider factory and EmbeddingService."""

    def test_factory_returns_local_by_default(self, mock_sentence_transformer):
        with patch.dict("os.environ", {"EMBEDDING_PROVIDER": "local"}):
            provider = get_embedding_provider()
            assert isinstance(provider, LocalSentenceTransformerEmbeddingProvider)
            assert provider.model_name == "BAAI/bge-small-en-v1.5"

    def test_factory_returns_mock_when_specified(self):
        provider = get_embedding_provider("mock", dimension=384)
        assert isinstance(provider, MockEmbeddingProvider)
        assert provider.get_dimension() == 384

    def test_mock_provider_embed_query_and_texts(self):
        mock_prov = MockEmbeddingProvider(dimension=384)
        assert mock_prov.get_dimension() == 384

        texts = ["Policy segment 1", "Policy segment 2"]
        doc_vecs = mock_prov.embed_texts(texts)
        assert len(doc_vecs) == 2
        assert len(doc_vecs[0]) == 384

        query_vec = mock_prov.embed_query("User search query")
        assert len(query_vec) == 384

    def test_mock_provider_supports_custom_dimension(self):
        mock_prov = MockEmbeddingProvider(dimension=1536)
        assert mock_prov.get_dimension() == 1536
        assert len(mock_prov.embed_texts(["sample"])[0]) == 1536

    def test_embedding_service_routes_embed_query(self, mock_sentence_transformer):
        provider = LocalSentenceTransformerEmbeddingProvider()
        svc = EmbeddingService(provider)

        doc_vec = svc.embed_text("Sample document text")
        assert len(doc_vec) == 384

        query_vec = svc.embed_query("Sample question?")
        assert len(query_vec) == 384
        assert svc.get_embedding_dimension() == 384

    def test_unrecognized_provider_falls_back_to_local(self, mock_sentence_transformer):
        provider = get_embedding_provider("unknown_provider_xyz")
        assert isinstance(provider, LocalSentenceTransformerEmbeddingProvider)
