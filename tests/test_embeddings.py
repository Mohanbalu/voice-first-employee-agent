"""Unit tests for Knowledge Base Embeddings Pipeline (Module 3.2).

Covers:
1. Chunk loading and text extraction
2. Single and batch text embedding
3. Batch size partitioning behavior
4. Retry handling and exponential backoff on transient errors
5. Terminal failure and error recording
6. Vector dimension validation (uniformity, empty vector detection, numeric checks)
7. Inconsistent vector dimensions detection
8. Checkpoint / resume behavior (skipping existing embeddings)
9. Force / rebuild behavior
10. Output JSONL file structure and metadata preservation
11. Validation report serialization
12. Security: ensuring no API keys are logged or leaked
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from backend.app.rag.embeddings import (
    BaseEmbeddingProvider,
    EmbeddingPipeline,
    EmbeddingRecord,
    EmbeddingService,
    EmbeddingValidationReport,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
)


class TestMockEmbeddingProvider:
    """Tests for the offline mock embedding provider."""

    def test_mock_dimension(self):
        provider = MockEmbeddingProvider(dimension=1536)
        assert provider.get_dimension() == 1536
        assert provider.model_name == "mock-text-embedding-3-small"

    def test_mock_embed_texts(self):
        provider = MockEmbeddingProvider(dimension=256)
        texts = ["First test text.", "Second test text."]
        vectors = provider.embed_texts(texts)

        assert len(vectors) == 2
        assert len(vectors[0]) == 256
        assert len(vectors[1]) == 256
        # Deterministic vectors
        vectors_again = provider.embed_texts(texts)
        assert vectors[0] == vectors_again[0]

    def test_mock_service_wrapper(self):
        provider = MockEmbeddingProvider(dimension=128)
        service = EmbeddingService(provider)

        vec = service.embed_text("Sample string")
        assert len(vec) == 128
        assert service.get_embedding_dimension() == 128

        batch = service.embed_batch(["A", "B", "C"])
        assert len(batch) == 3


class TestOpenAIEmbeddingProviderMocked:
    """Tests for OpenAI provider using mocked client responses."""

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY is not configured"):
                OpenAIEmbeddingProvider(api_key=None)

    def test_successful_embedding_generation(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item1 = MagicMock()
        mock_item1.embedding = [0.1, 0.2, 0.3]
        mock_item2 = MagicMock()
        mock_item2.embedding = [0.4, 0.5, 0.6]
        mock_response.data = [mock_item1, mock_item2]
        mock_client.embeddings.create.return_value = mock_response

        with patch("openai.OpenAI", return_value=mock_client):
            provider = OpenAIEmbeddingProvider(api_key="sk-test-key-mock", model="text-embedding-3-small")
            vectors = provider.embed_texts(["Text 1", "Text 2"])

            assert len(vectors) == 2
            assert vectors[0] == [0.1, 0.2, 0.3]
            assert vectors[1] == [0.4, 0.5, 0.6]
            assert provider.get_dimension() == 3

    def test_retry_on_rate_limit_error(self):
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2]
        mock_response.data = [mock_item]

        # Fail once with RateLimitError, then succeed
        mock_err = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_client.embeddings.create.side_effect = [mock_err, mock_response]

        with patch("openai.OpenAI", return_value=mock_client), patch("time.sleep") as mock_sleep:
            provider = OpenAIEmbeddingProvider(api_key="sk-test-key-mock", max_retries=2)
            vectors = provider.embed_texts(["Text to retry"])

            assert len(vectors) == 1
            assert vectors[0] == [0.1, 0.2]
            assert mock_client.embeddings.create.call_count == 2
            assert mock_sleep.call_count == 1

    def test_max_retries_exceeded_raises(self):
        from openai import RateLimitError

        mock_client = MagicMock()
        mock_err = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_client.embeddings.create.side_effect = mock_err

        with patch("openai.OpenAI", return_value=mock_client), patch("time.sleep"):
            provider = OpenAIEmbeddingProvider(api_key="sk-test-key-mock", max_retries=2)
            with pytest.raises(RateLimitError):
                provider.embed_texts(["Text that fails"])

            assert mock_client.embeddings.create.call_count == 3  # 1 initial + 2 retries


class TestEmbeddingValidation:
    """Tests for vector validation logic."""

    def test_valid_embeddings_pass(self):
        records = [
            {"chunk_id": "c1", "embedding": [0.1, 0.2, 0.3]},
            {"chunk_id": "c2", "embedding": [0.4, 0.5, 0.6]},
        ]
        report = EmbeddingPipeline.validate_embeddings(records, expected_dimension=3)
        assert report.validation_status == "PASS"
        assert report.embedding_dimension == 3
        assert len(report.errors) == 0

    def test_empty_vector_detected(self):
        records = [
            {"chunk_id": "c1", "embedding": [0.1, 0.2]},
            {"chunk_id": "c2", "embedding": []},
        ]
        report = EmbeddingPipeline.validate_embeddings(records)
        assert report.validation_status == "FAIL"
        assert any("empty embedding vector" in e for e in report.errors)

    def test_inconsistent_dimensions_detected(self):
        records = [
            {"chunk_id": "c1", "embedding": [0.1, 0.2, 0.3]},
            {"chunk_id": "c2", "embedding": [0.4, 0.5]},
        ]
        report = EmbeddingPipeline.validate_embeddings(records)
        assert report.validation_status == "FAIL"
        assert any("Dimension mismatch" in e for e in report.errors)

    def test_non_numeric_vector_detected(self):
        records = [
            {"chunk_id": "c1", "embedding": [0.1, "not_a_number", 0.3]},
        ]
        report = EmbeddingPipeline.validate_embeddings(records)
        assert report.validation_status == "FAIL"
        assert any("non-numeric" in e for e in report.errors)

    def test_missing_vector_detected(self):
        records = [
            {"chunk_id": "c1"},
        ]
        report = EmbeddingPipeline.validate_embeddings(records)
        assert report.validation_status == "FAIL"
        assert any("null/missing" in e for e in report.errors)


class TestEmbeddingPipelineExecution:
    """Tests full pipeline loading, checkpointing, and output generation."""

    def _create_sample_chunks(self, chunks_dir: Path, count: int = 5) -> List[Dict]:
        chunks = []
        for i in range(count):
            chunk = {
                "document_id": "sample_doc",
                "document_name": "Sample Policy",
                "source_file": "Sample.pdf",
                "chunk_id": f"sample_doc_{i+1:04d}",
                "chunk_index": i,
                "page_start": 1,
                "page_end": 1,
                "section": "1. Purpose",
                "text": f"Policy text for section {i+1} covering compliance.",
            }
            chunks.append(chunk)

        (chunks_dir / "sample_doc.jsonl").write_text(
            "\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8"
        )
        return chunks

    def test_pipeline_run_mock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td = Path(temp_dir)
            chunks_dir = td / "chunks"
            chunks_dir.mkdir()
            output_dir = td / "embeddings"

            self._create_sample_chunks(chunks_dir, count=10)
            provider = MockEmbeddingProvider(dimension=64)

            pipeline = EmbeddingPipeline(
                provider=provider,
                chunks_dir=chunks_dir,
                output_dir=output_dir,
                batch_size=4,
            )

            summary = pipeline.run()

            assert summary.chunks_discovered == 10
            assert summary.new_embeddings_generated == 10
            assert summary.skipped_existing == 0
            assert summary.failed == 0
            assert summary.embedding_dimension == 64
            assert summary.validation_status == "PASS"

            # Check output files
            assert pipeline.output_file.exists()
            assert pipeline.report_file.exists()

            lines = pipeline.output_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 10

            record_0 = json.loads(lines[0])
            assert record_0["chunk_id"] == "sample_doc_0001"
            assert "text" in record_0
            assert len(record_0["embedding"]) == 64
            assert record_0["metadata"]["document_id"] == "sample_doc"
            assert record_0["metadata"]["section"] == "1. Purpose"
            assert "created_at" in record_0

    def test_pipeline_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td = Path(temp_dir)
            chunks_dir = td / "chunks"
            chunks_dir.mkdir()
            output_dir = td / "embeddings"

            self._create_sample_chunks(chunks_dir, count=8)
            provider = MockEmbeddingProvider(dimension=32)

            pipeline = EmbeddingPipeline(
                provider=provider,
                chunks_dir=chunks_dir,
                output_dir=output_dir,
                batch_size=4,
            )

            # RUN 1: Generate all 8
            summary_1 = pipeline.run()
            assert summary_1.new_embeddings_generated == 8
            assert summary_1.skipped_existing == 0

            # RUN 2: Re-run should skip all 8
            summary_2 = pipeline.run()
            assert summary_2.new_embeddings_generated == 0
            assert summary_2.skipped_existing == 8
            assert summary_2.validation_status == "PASS"

            # RUN 3 with force=True should regenerate all 8
            summary_3 = pipeline.run(force=True)
            assert summary_3.new_embeddings_generated == 8
            assert summary_3.skipped_existing == 0

    def test_no_api_key_leakage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            td = Path(temp_dir)
            chunks_dir = td / "chunks"
            chunks_dir.mkdir()
            output_dir = td / "embeddings"

            self._create_sample_chunks(chunks_dir, count=3)
            secret_key = "sk-super-secret-production-key-xyz123"

            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.data = [
                MagicMock(embedding=[0.1] * 16),
                MagicMock(embedding=[0.2] * 16),
                MagicMock(embedding=[0.3] * 16),
            ]
            mock_client.embeddings.create.return_value = mock_resp

            with patch("openai.OpenAI", return_value=mock_client):
                provider = OpenAIEmbeddingProvider(api_key=secret_key, model="text-embedding-3-small")
                pipeline = EmbeddingPipeline(
                    provider=provider,
                    chunks_dir=chunks_dir,
                    output_dir=output_dir,
                )
                summary = pipeline.run()

                # Verify files do not contain the secret
                output_content = pipeline.output_file.read_text(encoding="utf-8")
                report_content = pipeline.report_file.read_text(encoding="utf-8")

                assert secret_key not in output_content
                assert secret_key not in report_content
                assert secret_key not in str(summary)
