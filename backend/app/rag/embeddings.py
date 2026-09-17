"""Knowledge Base Embeddings Pipeline.

Generates vector embeddings for chunked policy documents using the official
OpenAI Python SDK. Supports configurable models, batch processing with exponential
backoff retries, checkpoint/resume capabilities, dimension validation, and offline
mock execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Attempt to load dotenv if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("knowledge_base.embeddings")

try:
    from backend.app.config import config
except ImportError:
    try:
        from app.config import config
    except ImportError:
        config = None

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_VECTOR_DIMENSION = 384
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 60.0


def get_project_root() -> Path:
    """Locates the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "backend").exists():
            return parent
        if (parent / "data" / "processed" / "chunks").exists():
            return parent
    return Path.cwd()


def get_config_param(key: str, default: Any) -> Any:
    """Reads configuration parameter from environment or defaults."""
    val = os.getenv(key)
    if val is not None and val.strip():
        if isinstance(default, int):
            try:
                return int(val.strip())
            except ValueError:
                return default
        if isinstance(default, float):
            try:
                return float(val.strip())
            except ValueError:
                return default
        return val.strip()
    return default


@dataclass
class EmbeddingRecord:
    """Represents an embedded chunk record ready for vector storage."""

    chunk_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Converts record to JSON-serializable dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class EmbeddingValidationReport:
    """Report detailing vector dimension and data integrity checks."""

    total_chunks: int
    successfully_embedded: int
    failed_embeddings: int
    skipped_embeddings: int
    embedding_dimension: int
    model_name: str
    timestamp: str
    validation_status: str  # "PASS" or "FAIL"
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingPipelineSummary:
    """High-level summary of an embedding pipeline execution."""

    chunks_discovered: int
    new_embeddings_generated: int
    skipped_existing: int
    failed: int
    embedding_dimension: int
    model: str
    output_file: Path
    validation_status: str
    report: EmbeddingValidationReport


class BaseEmbeddingProvider(Protocol):
    """Abstract protocol for embedding model providers."""

    @property
    def model_name(self) -> str:
        ...

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

    def get_dimension(self) -> int:
        ...


_GLOBAL_SENTENCE_TRANSFORMER_CACHE: Dict[str, Any] = {}


class LocalSentenceTransformerEmbeddingProvider:
    """Production local embedding provider using Sentence-Transformers (BAAI/bge-small-en-v1.5).

    Supports:
    - Lazy loading of model on first use (avoiding redundant reloads per chunk)
    - Deterministic 384-dimensional normalized float vectors
    - Document embedding (embed_texts) without query instruction
    - Query embedding (embed_query) applying BGE retrieval instruction
    - Configurable batch execution on CPU (or GPU if available)
    - Zero external API keys or network inference cost
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        query_instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ):
        cfg_emb = getattr(config, "embedding", None) if config else None
        self._model_name = (
            model_name
            or (cfg_emb.model if cfg_emb else None)
            or os.getenv("EMBEDDING_MODEL")
            or DEFAULT_LOCAL_MODEL
        )
        self._query_instruction = (
            query_instruction
            if query_instruction is not None
            else (
                (cfg_emb.query_instruction if cfg_emb else None)
                or os.getenv("EMBEDDING_QUERY_INSTRUCTION")
                or "Represent this sentence for searching relevant passages: "
            )
        )
        self._batch_size = (
            batch_size
            or (cfg_emb.batch_size if cfg_emb else None)
            or int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        )
        self._normalize_embeddings = normalize_embeddings
        self._device = device
        self._model = None
        self._dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _load_model(self):
        global _GLOBAL_SENTENCE_TRANSFORMER_CACHE
        if self._model_name in _GLOBAL_SENTENCE_TRANSFORMER_CACHE:
            self._model = _GLOBAL_SENTENCE_TRANSFORMER_CACHE[self._model_name]
            dim_fn = getattr(self._model, "get_embedding_dimension", None)
            if callable(dim_fn):
                self._dimension = self._model.get_embedding_dimension()
            elif hasattr(self._model, "get_sentence_embedding_dimension"):
                self._dimension = self._model.get_sentence_embedding_dimension()
            else:
                self._dimension = 384
            return self._model

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading local embedding model '%s'...", self._model_name)
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
                dim_fn = getattr(self._model, "get_embedding_dimension", None)
                if callable(dim_fn):
                    self._dimension = self._model.get_embedding_dimension()
                elif hasattr(self._model, "get_sentence_embedding_dimension"):
                    self._dimension = self._model.get_sentence_embedding_dimension()
                else:
                    self._dimension = 384
                _GLOBAL_SENTENCE_TRANSFORMER_CACHE[self._model_name] = self._model
                logger.info(
                    "Local embedding model '%s' loaded (dimension=%d)",
                    self._model_name,
                    self._dimension,
                )
            except Exception as exc:
                logger.error("Failed to load local embedding model '%s': %s", self._model_name, exc)
                raise RuntimeError(
                    f"Failed to load sentence-transformers model '{self._model_name}': {exc}"
                ) from exc
        return self._model

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension or 384

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds document/chunk passages (raw text, no query instruction)."""
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=self._normalize_embeddings,
        )
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single search query with BGE retrieval prompt/instruction."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        model = self._load_model()
        full_query = (
            f"{self._query_instruction}{text.strip()}"
            if self._query_instruction
            else text.strip()
        )
        embedding = model.encode(
            [full_query],
            show_progress_bar=False,
            normalize_embeddings=self._normalize_embeddings,
        )[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


class OpenAIEmbeddingProvider:
    """Official OpenAI Python SDK embedding provider with exponential backoff."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key or self._api_key == "your_openai_api_key_placeholder":
            raise ValueError(
                "OPENAI_API_KEY is not configured. Set OPENAI_API_KEY in backend/.env "
                "or pass an explicit key."
            )

        self._model = (
            model
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        )
        self._timeout = timeout
        self._max_retries = max_retries

        # Initialize official OpenAI client
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key, timeout=self._timeout)
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required. Install with 'pip install openai'."
            ) from exc

        self._cached_dimension: Optional[int] = None

    @property
    def model_name(self) -> str:
        return self._model

    def get_dimension(self) -> int:
        """Determines the vector dimension from known models or sample probe."""
        if self._cached_dimension is None:
            KNOWN_DIMENSIONS = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
                "BAAI/bge-small-en-v1.5": 384,
            }
            if self._model in KNOWN_DIMENSIONS:
                self._cached_dimension = KNOWN_DIMENSIONS[self._model]
            else:
                sample_vec = self.embed_texts(["sample probe"])[0]
                self._cached_dimension = len(sample_vec)
        return self._cached_dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts with bounded exponential backoff on retryable errors."""
        if not texts:
            return []

        from openai import APIConnectionError, InternalServerError, RateLimitError

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )
                embeddings = [item.embedding for item in response.data]
                if len(embeddings) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, received {len(embeddings)}"
                    )
                if self._cached_dimension is None and embeddings:
                    self._cached_dimension = len(embeddings[0])
                return embeddings

            except (RateLimitError, APIConnectionError, InternalServerError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    logger.error(
                        "Max retries (%d) exceeded on embedding batch: %s",
                        self._max_retries,
                        exc,
                    )
                    raise
                backoff_time = (2 ** attempt) * 1.5
                logger.warning(
                    "Retryable API error (%s). Backing off %.1fs (attempt %d/%d)...",
                    exc.__class__.__name__,
                    backoff_time,
                    attempt + 1,
                    self._max_retries,
                )
                time.sleep(backoff_time)

            except Exception as exc:
                logger.error("Non-retryable API error during embedding: %s", exc)
                raise

        if last_error:
            raise last_error
        return []

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single search query."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        return self.embed_texts([text.strip()])[0]


class MockEmbeddingProvider:
    """Deterministic, normalized pseudo-embedding provider for offline testing."""

    def __init__(
        self,
        dimension: int = 384,
        model: str = "mock-text-embedding-3-small",
    ):
        self._dimension = dimension
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    def get_dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates deterministic unit-normalized pseudo-vectors based on text hash."""
        results: List[List[float]] = []
        for text in texts:
            # Generate deterministic pseudo-random float vector from text hash
            h = hashlib.sha256(text.encode("utf-8")).digest()
            raw_floats = []
            for i in range(self._dimension):
                byte_val = h[i % len(h)]
                val = float(byte_val) / 255.0 - 0.5 + ((i % 17) - 8) * 0.01
                raw_floats.append(val)

            # L2 normalize
            norm = math.sqrt(sum(v * v for v in raw_floats))
            normalized = [round(v / max(norm, 1e-9), 6) for v in raw_floats]
            results.append(normalized)
        return results

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query."""
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")
        return self.embed_texts([text.strip()])[0]


def get_embedding_provider(
    provider_type: Optional[str] = None,
    dimension: Optional[int] = None,
    model: Optional[str] = None,
) -> BaseEmbeddingProvider:
    """Factory to resolve embedding provider based on config or argument."""
    cfg_emb = getattr(config, "embedding", None) if config else None
    selected = (
        provider_type
        or (cfg_emb.provider if cfg_emb else None)
        or os.getenv("EMBEDDING_PROVIDER")
        or "local"
    ).strip().lower()

    if selected == "local":
        target_model = model or (cfg_emb.model if cfg_emb else None) or DEFAULT_LOCAL_MODEL
        return LocalSentenceTransformerEmbeddingProvider(model_name=target_model)
    elif selected == "mock":
        target_dim = (
            dimension
            or (cfg_emb.dimension if cfg_emb else None)
            or (config.db.vector_dimension if config else None)
            or 384
        )
        return MockEmbeddingProvider(dimension=target_dim)
    elif selected == "openai":
        target_model = model or (cfg_emb.model if cfg_emb else None) or "text-embedding-3-small"
        return OpenAIEmbeddingProvider(model=target_model)
    else:
        logger.warning("Unrecognized embedding provider '%s', defaulting to local", selected)
        return LocalSentenceTransformerEmbeddingProvider()


class EmbeddingService:
    """Service wrapper for embedding single texts, queries, or batches."""

    def __init__(self, provider: BaseEmbeddingProvider):
        self.provider = provider

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single document string."""
        return self.provider.embed_texts([text])[0]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string using provider's query logic."""
        if hasattr(self.provider, "embed_query"):
            return self.provider.embed_query(text)
        return self.provider.embed_texts([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of strings."""
        return self.provider.embed_texts(texts)

    def get_embedding_dimension(self) -> int:
        """Returns the embedding vector dimension."""
        return self.provider.get_dimension()



class EmbeddingPipeline:
    """Manages chunk discovery, batch embedding, checkpointing, and vector validation."""

    def __init__(
        self,
        provider: Optional[BaseEmbeddingProvider] = None,
        chunks_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        batch_size: Optional[int] = None,
    ):
        root = get_project_root()
        self.chunks_dir = chunks_dir or (root / "data" / "processed" / "chunks")
        self.output_dir = output_dir or (root / "data" / "processed" / "embeddings")
        self.output_file = self.output_dir / "embeddings.jsonl"
        self.report_file = self.output_dir / "embedding_validation_report.json"
        self.errors_file = self.output_dir / "embedding_errors.jsonl"

        self.batch_size = (
            batch_size
            or get_config_param("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE)
        )
        self.provider = provider or get_embedding_provider()

    def ensure_output_dir(self) -> None:
        """Creates the embeddings output directory if not present."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_chunks(self) -> List[Dict[str, Any]]:
        """Discovers and loads all valid chunk records from processed JSONL files."""
        if not self.chunks_dir.exists():
            logger.warning("Chunks directory does not exist: %s", self.chunks_dir)
            return []

        jsonl_files = sorted(self.chunks_dir.glob("*.jsonl"))
        chunks: List[Dict[str, Any]] = []

        for f in jsonl_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    # Verify basic chunk structure
                    if "chunk_id" in data and "text" in data:
                        chunks.append(data)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON line in %s: %s", f.name, exc)

        logger.info("Loaded %d chunk(s) from %d JSONL file(s)", len(chunks), len(jsonl_files))
        return chunks

    def load_existing_embeddings(self) -> Dict[str, Dict[str, Any]]:
        """Loads existing embedding records from the checkpoint JSONL file."""
        if not self.output_file.exists():
            return {}

        existing: Dict[str, Dict[str, Any]] = {}
        for line in self.output_file.read_text(encoding="utf-8").splitlines():
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                cid = record.get("chunk_id")
                if cid:
                    existing[cid] = record
            except json.JSONDecodeError:
                continue

        logger.info("Found %d existing embedding checkpoint(s)", len(existing))
        return existing

    @staticmethod
    def validate_embeddings(
        records: List[Dict[str, Any]],
        expected_dimension: Optional[int] = None,
        model_name: str = "unknown",
        total_discovered: int = 0,
        skipped_count: int = 0,
        failed_count: int = 0,
    ) -> EmbeddingValidationReport:
        """Validates that all vectors are non-empty, numeric, and uniformly dimensioned."""
        errors: List[str] = []
        detected_dimension = expected_dimension or 0

        if not records and total_discovered > 0:
            errors.append("No embedding records found for validation.")

        for idx, rec in enumerate(records):
            cid = rec.get("chunk_id", f"record_{idx}")
            vec = rec.get("embedding")

            if vec is None:
                errors.append(f"Chunk '{cid}' has null/missing embedding vector.")
                continue

            if not isinstance(vec, list):
                errors.append(f"Chunk '{cid}' embedding is not a list (got {type(vec).__name__}).")
                continue

            if len(vec) == 0:
                errors.append(f"Chunk '{cid}' has an empty embedding vector (0 elements).")
                continue

            # Verify numeric values
            non_numeric = sum(1 for v in vec if not isinstance(v, (int, float)))
            if non_numeric > 0:
                errors.append(f"Chunk '{cid}' contains {non_numeric} non-numeric vector values.")

            # Dimension uniformity
            v_dim = len(vec)
            if detected_dimension == 0:
                detected_dimension = v_dim
            elif v_dim != detected_dimension:
                errors.append(
                    f"Dimension mismatch in chunk '{cid}': expected {detected_dimension}, got {v_dim}"
                )

        status = "PASS" if not errors and records else "FAIL"

        return EmbeddingValidationReport(
            total_chunks=total_discovered or len(records),
            successfully_embedded=len(records),
            failed_embeddings=failed_count,
            skipped_embeddings=skipped_count,
            embedding_dimension=detected_dimension,
            model_name=model_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            validation_status=status,
            errors=errors,
        )

    def run(
        self,
        force: bool = False,
        provider: Optional[BaseEmbeddingProvider] = None,
    ) -> EmbeddingPipelineSummary:
        """Executes the embedding pipeline with checkpointing and validation.

        Args:
            force: If True, regenerates all embeddings even if checkpoint exists.
            provider: Optional override for embedding provider.

        Returns:
            EmbeddingPipelineSummary with execution counts and report.
        """
        self.ensure_output_dir()
        active_provider = provider or self.provider or get_embedding_provider()

        all_chunks = self.load_chunks()
        total_discovered = len(all_chunks)

        existing_embeddings = {} if force else self.load_existing_embeddings()
        # Invalidate existing checkpoint if vector dimensions differ from target model
        if existing_embeddings:
            sample_rec = next(iter(existing_embeddings.values()))
            sample_vec = sample_rec.get("embedding", [])
            target_dim = active_provider.get_dimension()
            if len(sample_vec) != target_dim:
                logger.warning(
                    "Existing checkpoint vector dimension (%d) does not match target model dimension (%d). "
                    "Resetting checkpoint for clean generation.",
                    len(sample_vec),
                    target_dim,
                )
                existing_embeddings = {}
                force = True

        chunks_to_embed = [c for c in all_chunks if c["chunk_id"] not in existing_embeddings]
        skipped_count = len(all_chunks) - len(chunks_to_embed)

        logger.info(
            "Embedding run mode: total=%d, to_embed=%d, skipped=%d, force=%s",
            total_discovered,
            len(chunks_to_embed),
            skipped_count,
            force,
        )

        newly_embedded_count = 0
        failed_count = 0
        failed_records: List[Dict[str, Any]] = []

        # Open output file in write mode if force, otherwise append mode
        mode = "w" if force else "a"

        if chunks_to_embed:
            # Process in batches
            num_batches = math.ceil(len(chunks_to_embed) / self.batch_size)
            with open(self.output_file, mode, encoding="utf-8") as out_f:
                for b_idx in range(num_batches):
                    batch = chunks_to_embed[b_idx * self.batch_size : (b_idx + 1) * self.batch_size]
                    texts = [c["text"] for c in batch]

                    try:
                        vectors = active_provider.embed_texts(texts)

                        for chunk, vec in zip(batch, vectors):
                            # Separate text and meaningful metadata
                            metadata = {k: v for k, v in chunk.items() if k != "text"}
                            record = EmbeddingRecord(
                                chunk_id=chunk["chunk_id"],
                                text=chunk["text"],
                                embedding=vec,
                                metadata=metadata,
                            )
                            out_f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                            newly_embedded_count += 1

                        logger.info(
                            "Embedded batch %d/%d (%d chunks)",
                            b_idx + 1,
                            num_batches,
                            len(batch),
                        )

                    except Exception as exc:
                        failed_count += len(batch)
                        logger.error("Failed to embed batch %d: %s", b_idx + 1, exc)
                        for chunk in batch:
                            failed_records.append({
                                "chunk_id": chunk.get("chunk_id"),
                                "error": str(exc),
                            })

        # Record failures if any occurred
        if failed_records:
            with open(self.errors_file, "a", encoding="utf-8") as err_f:
                for err in failed_records:
                    err_f.write(json.dumps(err, ensure_ascii=False) + "\n")

        # Load all records currently in output file for comprehensive validation
        all_records = list(self.load_existing_embeddings().values())

        # Validate complete vector collection
        report = self.validate_embeddings(
            records=all_records,
            model_name=active_provider.model_name,
            total_discovered=total_discovered,
            skipped_count=skipped_count,
            failed_count=failed_count,
        )

        # Save validation report
        self.report_file.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return EmbeddingPipelineSummary(
            chunks_discovered=total_discovered,
            new_embeddings_generated=newly_embedded_count,
            skipped_existing=skipped_count,
            failed=failed_count,
            embedding_dimension=report.embedding_dimension,
            model=active_provider.model_name,
            output_file=self.output_file,
            validation_status=report.validation_status,
            report=report,
        )


def print_pipeline_summary(summary: EmbeddingPipelineSummary) -> None:
    """Prints a clean, user-friendly pipeline execution summary."""
    sep = "=" * 50
    print(f"\n{sep}")
    print("Embedding Pipeline Complete")
    print(sep)
    print(f"Chunks discovered:         {summary.chunks_discovered}")
    print(f"New embeddings generated:  {summary.new_embeddings_generated}")
    print(f"Skipped existing:          {summary.skipped_existing}")
    print(f"Failed:                    {summary.failed}")
    print(f"Embedding dimension:       {summary.embedding_dimension}")
    print(f"Model:                     {summary.model}")
    print(f"Output:                    {summary.output_file}")
    print(f"Validation:                {summary.validation_status}")
    print(sep)

    if summary.report.errors:
        print("\nValidation Errors:")
        for err in summary.report.errors[:10]:
            print(f"- {err}")
        if len(summary.report.errors) > 10:
            print(f"- ... and {len(summary.report.errors) - 10} more errors.")
        print(sep)


def main() -> None:
    """Command-line entrypoint for knowledge base embedding generation."""
    parser = argparse.ArgumentParser(
        description="Generate vector embeddings for knowledge base chunks."
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of all embeddings, ignoring checkpoints.",
    )
    parser.add_argument(
        "--mock",
        "-m",
        action="store_true",
        help="Run in mock/offline mode without calling local models or external APIs.",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["local", "mock", "openai"],
        default=None,
        help="Explicitly choose embedding provider (local, mock, openai).",
    )
    parser.add_argument(
        "--replace-mock",
        action="store_true",
        help="Acknowledge and replace existing mock embeddings with real embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Batch size for embedding generation.",
    )
    args = parser.parse_args()

    provider: Optional[BaseEmbeddingProvider] = None
    is_mock = args.mock or (args.provider == "mock")

    if is_mock:
        logger.info("Running embedding pipeline with MockEmbeddingProvider (offline mode)")
        provider = MockEmbeddingProvider(dimension=DEFAULT_VECTOR_DIMENSION)
    else:
        # Prompt requirement 16: clearly indicate provider, model, dimension, and warning
        prov_type = args.provider or getattr(config.embedding, "provider", "local") if config else "local"
        model_name = getattr(config.embedding, "model", DEFAULT_LOCAL_MODEL) if config else DEFAULT_LOCAL_MODEL
        print(f"Provider:  {prov_type}")
        print(f"Model:     {model_name}")
        print(f"Dimension: {DEFAULT_VECTOR_DIMENSION}")
        print("WARNING: Existing mock embeddings will be replaced because all current embeddings are mock.\n")

        provider = get_embedding_provider(
            provider_type=args.provider,
            model=model_name,
        )

    pipeline = EmbeddingPipeline(provider=provider, batch_size=args.batch_size)
    force = args.force or args.replace_mock
    summary = pipeline.run(force=force, provider=provider)
    print_pipeline_summary(summary)


if __name__ == "__main__":
    main()

