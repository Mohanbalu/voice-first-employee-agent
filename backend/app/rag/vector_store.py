"""Vector Store & Similarity Search Module.

Provides tenant-isolated persistence for documents, chunks, and pgvector embeddings,
as well as cosine similarity vector retrieval with strict multi-tenant filtering.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path for direct script execution
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    from backend.app.config import config, get_project_root
    from backend.app.database import get_session_factory
    from backend.app.models.chunk import Chunk
    from backend.app.models.document import Document
    from backend.app.models.embedding import ChunkEmbedding
    from backend.app.models.tenant import Tenant
except ImportError:
    from app.config import config, get_project_root
    from app.database import get_session_factory
    from app.models.chunk import Chunk
    from app.models.document import Document
    from app.models.embedding import ChunkEmbedding
    from app.models.tenant import Tenant

logger = logging.getLogger("knowledge_base.vector_store")


@dataclass
class SearchResult:
    """Represents a vector similarity search result."""

    chunk_id: str
    document_id: str
    document_name: str
    source_file: str
    text: str
    page_start: int
    page_end: int
    section: Optional[str]
    similarity_score: float
    distance: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportSummary:
    """Summary of an embedding import run."""

    tenant_id: str
    tenant_slug: str
    documents_processed: int
    chunks_processed: int
    embeddings_processed: int
    skipped_existing: int
    is_mock: bool
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VectorStore:
    """Tenant-scoped vector store operations for documents, chunks, and embeddings."""

    def __init__(self, target_dimension: Optional[int] = None):
        self.target_dimension = target_dimension or config.db.vector_dimension

    @staticmethod
    def get_or_create_tenant(
        session: Session,
        name: str,
        slug: str,
        tenant_id: Optional[uuid.UUID] = None,
    ) -> Tenant:
        """Retrieves an existing tenant by slug or creates a new one."""
        tenant = session.execute(
            select(Tenant).where(Tenant.slug == slug)
        ).scalar_one_or_none()

        if tenant is None:
            tenant = Tenant(
                id=tenant_id or uuid.uuid4(),
                name=name,
                slug=slug,
            )
            session.add(tenant)
            session.commit()
            session.refresh(tenant)
            logger.info("Created new tenant: %s (slug=%s, id=%s)", name, slug, tenant.id)
        return tenant

    def import_embeddings_from_file(
        self,
        session: Session,
        jsonl_path: Path,
        tenant_id: uuid.UUID,
        allow_mock: bool = False,
    ) -> ImportSummary:
        """Imports documents, chunks, and embeddings idempotently into PostgreSQL/pgvector.

        Safety Rules:
        - Mock embeddings are strictly rejected unless allow_mock=True.
        - Vectors must match configured vector dimension.
        - All operations are scoped to the provided tenant_id.
        """
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Embedding file does not exist: {jsonl_path}")

        # Check tenant exists
        tenant = session.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        ).scalar_one_or_none()
        if tenant is None:
            raise ValueError(f"Tenant with ID {tenant_id} does not exist.")

        summary = ImportSummary(
            tenant_id=str(tenant.id),
            tenant_slug=tenant.slug,
            documents_processed=0,
            chunks_processed=0,
            embeddings_processed=0,
            skipped_existing=0,
            is_mock=False,
        )

        lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            logger.warning("Empty embeddings file: %s", jsonl_path)
            return summary

        # Pre-scan first record to detect mock vs production model
        first_record = json.loads(lines[0])
        # Check validation report if available
        report_file = jsonl_path.parent / "embedding_validation_report.json"
        is_mock_file = False
        detected_model = "unknown"

        if report_file.exists():
            try:
                rep_data = json.loads(report_file.read_text(encoding="utf-8"))
                detected_model = rep_data.get("model_name", "")
                if "mock" in detected_model.lower():
                    is_mock_file = True
            except Exception:
                pass

        if "mock" in first_record.get("chunk_id", "").lower() or is_mock_file:
            is_mock_file = True

        summary.is_mock = is_mock_file

        if is_mock_file and not allow_mock:
            error_msg = (
                "Refusing to import MOCK embeddings into database without explicit authorization. "
                "Pass --allow-mock flag to authorize development/test mock vector import."
            )
            logger.error(error_msg)
            summary.errors.append(error_msg)
            raise ValueError(error_msg)

        # Cache existing documents for this tenant to avoid redundant queries
        existing_docs = {
            d.document_id: d
            for d in session.execute(
                select(Document).where(Document.tenant_id == tenant_id)
            ).scalars().all()
        }

        # Cache existing chunks for this tenant
        existing_chunks = {
            c.chunk_id: c
            for c in session.execute(
                select(Chunk).where(Chunk.tenant_id == tenant_id)
            ).scalars().all()
        }

        # Cache existing chunk embeddings for this tenant
        existing_embeddings = {
            e.chunk_id: e
            for e in session.execute(
                select(ChunkEmbedding).where(ChunkEmbedding.tenant_id == tenant_id)
            ).scalars().all()
        }

        BATCH_FLUSH_SIZE = 100  # flush every N rows to reduce WAN round-trips

        # ── PASS 1: Parse all records; upsert Documents (flush immediately — only ~12) ──
        parsed_records = []
        for line_num, line in enumerate(lines):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as exc:
                summary.errors.append(f"Line {line_num+1}: invalid JSON: {exc}")
                continue

            chunk_id = data.get("chunk_id")
            text_content = data.get("text")
            embedding = data.get("embedding")
            metadata = data.get("metadata", {})

            if not chunk_id or not text_content or embedding is None:
                summary.errors.append(f"Line {line_num+1}: missing required fields.")
                continue

            # Validate embedding vector
            if not isinstance(embedding, list) or len(embedding) != self.target_dimension:
                err = (
                    f"Chunk '{chunk_id}' has vector dimension {len(embedding) if isinstance(embedding, list) else 'invalid'}; "
                    f"expected {self.target_dimension}."
                )
                summary.errors.append(err)
                continue

            # Non-numeric check
            if any(not isinstance(v, (int, float)) for v in embedding):
                summary.errors.append(f"Chunk '{chunk_id}' contains non-numeric vector values.")
                continue

            doc_slug = metadata.get("document_id", "default_doc")
            doc_name = metadata.get("document_name", doc_slug)
            src_file = metadata.get("source_file", f"{doc_slug}.pdf")

            # Upsert Document (few documents — flush immediately to get their IDs)
            if doc_slug not in existing_docs:
                doc_obj = Document(
                    tenant_id=tenant_id,
                    document_id=doc_slug,
                    document_name=doc_name,
                    source_file=src_file,
                )
                session.add(doc_obj)
                session.flush()  # needed: get doc ID before chunks reference it
                existing_docs[doc_slug] = doc_obj
                summary.documents_processed += 1

            parsed_records.append((chunk_id, text_content, embedding, metadata))

        # ── PASS 2: Upsert Chunks in batches ──────────────────────────────────────────
        pending = 0
        for chunk_id, text_content, embedding, metadata in parsed_records:
            doc_slug = metadata.get("document_id", "default_doc")
            doc_obj = existing_docs[doc_slug]
            c_index = metadata.get("chunk_index", 0)
            p_start = metadata.get("page_start", 1)
            p_end = metadata.get("page_end", 1)
            section = metadata.get("section")

            if chunk_id not in existing_chunks:
                chunk_obj = Chunk(
                    tenant_id=tenant_id,
                    document_ref_id=doc_obj.id,
                    document_id=doc_slug,
                    chunk_id=chunk_id,
                    chunk_index=c_index,
                    text=text_content,
                    page_start=p_start,
                    page_end=p_end,
                    section=section,
                    metadata_json=metadata,
                )
                session.add(chunk_obj)
                existing_chunks[chunk_id] = chunk_obj
                summary.chunks_processed += 1
                pending += 1
                if pending % BATCH_FLUSH_SIZE == 0:
                    session.flush()
                    logger.info("Chunks flushed: %d so far...", summary.chunks_processed)

        if pending % BATCH_FLUSH_SIZE != 0:
            session.flush()  # flush remaining chunks to get their IDs

        # ── PASS 3: Upsert ChunkEmbeddings in batches ────────────────────────────────
        pending = 0
        for chunk_id, text_content, embedding, metadata in parsed_records:
            if chunk_id not in existing_embeddings:
                chunk_obj = existing_chunks[chunk_id]
                emb_obj = ChunkEmbedding(
                    tenant_id=tenant_id,
                    chunk_ref_id=chunk_obj.id,
                    chunk_id=chunk_id,
                    embedding=embedding,
                    embedding_model=detected_model,
                    embedding_dimension=self.target_dimension,
                    is_mock=is_mock_file,
                )
                session.add(emb_obj)
                existing_embeddings[chunk_id] = emb_obj
                summary.embeddings_processed += 1
                pending += 1
                if pending % BATCH_FLUSH_SIZE == 0:
                    session.flush()
                    logger.info("Embeddings flushed: %d so far...", summary.embeddings_processed)
            else:
                summary.skipped_existing += 1

        session.commit()
        logger.info(
            "Import finished for tenant '%s': %d docs, %d chunks, %d embeddings, %d skipped",
            tenant.slug,
            summary.documents_processed,
            summary.chunks_processed,
            summary.embeddings_processed,
            summary.skipped_existing,
        )
        return summary

    def search_similar_chunks(
        self,
        session: Session,
        tenant_id: uuid.UUID,
        query_embedding: List[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[SearchResult]:
        """Low-level vector similarity search foundation with strict tenant isolation.

        Enforces WHERE chunk_embeddings.tenant_id = :tenant_id BEFORE similarity ranking.
        Calculates cosine similarity as: 1 - cosine_distance.
        """
        # Validate query dimension
        if not isinstance(query_embedding, list) or len(query_embedding) != self.target_dimension:
            raise ValueError(
                f"Query vector dimension is {len(query_embedding) if isinstance(query_embedding, list) else 'invalid'}; "
                f"expected {self.target_dimension}."
            )

        if any(not isinstance(v, (int, float)) for v in query_embedding):
            raise ValueError("Query vector contains non-numeric values.")

        if top_k <= 0:
            top_k = 5

        # Compute cosine distance using pgvector operator <=>
        # cosine_similarity = 1 - cosine_distance
        cosine_dist = ChunkEmbedding.embedding.cosine_distance(query_embedding)

        stmt = (
            select(
                Chunk.chunk_id,
                Chunk.document_id,
                Document.document_name,
                Document.source_file,
                Chunk.text,
                Chunk.page_start,
                Chunk.page_end,
                Chunk.section,
                Chunk.metadata_json,
                cosine_dist.label("distance"),
            )
            .join(Chunk, ChunkEmbedding.chunk_ref_id == Chunk.id)
            .join(Document, Chunk.document_ref_id == Document.id)
            .where(ChunkEmbedding.tenant_id == tenant_id)  # MANDATORY TENANT ISOLATION
            .order_by("distance")
            .limit(top_k)
        )

        rows = session.execute(stmt).fetchall()
        results: List[SearchResult] = []

        for row in rows:
            dist = float(row.distance)
            sim = round(1.0 - dist, 4)
            if sim < min_similarity:
                continue

            results.append(
                SearchResult(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    document_name=row.document_name,
                    source_file=row.source_file,
                    text=row.text,
                    page_start=row.page_start,
                    page_end=row.page_end,
                    section=row.section,
                    similarity_score=sim,
                    distance=round(dist, 4),
                    metadata=row.metadata_json or {},
                )
            )

        return results

    @staticmethod
    def get_tenant_stats(session: Session, tenant_id: uuid.UUID) -> Dict[str, int]:
        """Returns entity counts for a specific tenant."""
        doc_count = session.execute(
            select(Document).where(Document.tenant_id == tenant_id)
        ).scalars().all()
        chunk_count = session.execute(
            select(Chunk).where(Chunk.tenant_id == tenant_id)
        ).scalars().all()
        emb_count = session.execute(
            select(ChunkEmbedding).where(ChunkEmbedding.tenant_id == tenant_id)
        ).scalars().all()

        return {
            "documents": len(doc_count),
            "chunks": len(chunk_count),
            "embeddings": len(emb_count),
        }


def print_import_summary(summary: ImportSummary) -> None:
    """Prints formatted summary of import execution."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("Vector Store Import Complete")
    print(sep)
    print(f"Tenant ID:              {summary.tenant_id}")
    print(f"Tenant Slug:            {summary.tenant_slug}")
    print(f"Documents Ingested:     {summary.documents_processed}")
    print(f"Chunks Ingested:        {summary.chunks_processed}")
    print(f"Embeddings Ingested:    {summary.embeddings_processed}")
    print(f"Skipped (Existing):     {summary.skipped_existing}")
    print(f"Mock Vectors Allowed:   {summary.is_mock}")
    print(sep)

    if summary.errors:
        print("\nErrors / Warnings:")
        for err in summary.errors[:10]:
            print(f"- {err}")
        print(sep)


def main() -> None:
    """CLI entrypoint for vector store operations."""
    parser = argparse.ArgumentParser(
        description="Vector store management & embedding import tool."
    )
    parser.add_argument(
        "--import-file",
        "-i",
        type=str,
        default=None,
        help="Path to embeddings.jsonl file to import.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Explicitly permit importing mock/test vector embeddings into the database.",
    )
    parser.add_argument(
        "--tenant-slug",
        type=str,
        default=config.tenant.default_slug,
        help="Tenant slug to import records under (default: dev-org).",
    )
    parser.add_argument(
        "--tenant-name",
        type=str,
        default=config.tenant.default_name,
        help="Tenant name (default: Development Organization).",
    )
    args = parser.parse_args()

    session_factory = get_session_factory()
    store = VectorStore()

    with session_factory() as session:
        tenant = store.get_or_create_tenant(
            session=session,
            name=args.tenant_name,
            slug=args.tenant_slug,
            tenant_id=uuid.UUID(config.tenant.default_id),
        )

        root = get_project_root()
        import_path = (
            Path(args.import_file)
            if args.import_file
            else (root / "data" / "processed" / "embeddings" / "embeddings.jsonl")
        )

        try:
            summary = store.import_embeddings_from_file(
                session=session,
                jsonl_path=import_path,
                tenant_id=tenant.id,
                allow_mock=args.allow_mock,
            )
            print_import_summary(summary)
        except Exception as exc:
            logger.error("Import failed: %s", exc)
            print(f"\n[ERROR] Import failed: {exc}")


if __name__ == "__main__":
    main()
