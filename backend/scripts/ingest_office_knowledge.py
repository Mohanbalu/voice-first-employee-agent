"""HCL Office Locations & Facilities Knowledge Ingestion Script.

Ingests the authoritative HCL office locations document into the RAG knowledge base:
1. Chunks `data/raw/office/hcl_office_locations.md` into granular, section-aware chunks.
2. Generates 384-dimensional dense embeddings using local model BAAI/bge-small-en-v1.5.
3. Writes chunk records to `data/processed/chunks/hcl_office_locations.jsonl`.
4. Writes embedding records to `data/processed/embeddings/hcl_office_locations.jsonl`.
5. Imports document, chunks, and embeddings into AWS RDS PostgreSQL pgvector under tenant
   `00000000-0000-0000-0000-000000000001` without deleting or modifying any existing embeddings.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from backend.app.config import config, get_project_root
from backend.app.database import get_session_factory
from backend.app.models.chunk import Chunk
from backend.app.models.document import Document
from backend.app.models.embedding import ChunkEmbedding
from backend.app.models.tenant import Tenant
from backend.app.rag.embeddings import LocalSentenceTransformerEmbeddingProvider
from backend.app.rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("ingest_office_knowledge")

DOCUMENT_ID = "hcl_office_locations"
DOCUMENT_NAME = "HCL Office Locations, Floors, Teams and Facilities"
SOURCE_FILE = "hcl_office_locations.md"
TARGET_TENANT_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def get_office_knowledge_chunks() -> List[Dict[str, Any]]:
    """Constructs authoritative, self-contained chunks for HCL office locations.

    Each chunk contains direct factual statements to ensure optimal dense semantic retrieval
    for building, floor, team, and facility questions.
    """
    raw_sections: List[Tuple[str, str, str]] = [
        (
            "SDC — Ground Floor",
            "SDC Ground Floor: Reception and Cafeteria",
            (
                "SDC contains Techbees facilities, classrooms, IT teams, seminar halls, and office/support areas.\n\n"
                "SDC Ground Floor Facilities:\n"
                "• Reception: SDC has a reception on the ground floor.\n"
                "• Cafeteria: SDC has a cafeteria on the ground floor. The cafeteria is located on SDC Ground Floor. "
                "Employees can visit the ground floor of SDC for cafeteria dining and food services.\n"
                "Note: The ground floor of SDC is the only cafeteria location identified in the office information."
            ),
        ),
        (
            "SDC — 2nd Floor",
            "SDC 2nd Floor: Techbees Classrooms, IT Team, and Seminar Halls",
            (
                "SDC 2nd Floor Facilities and Teams:\n"
                "• Techbees Classrooms: Techbees has classrooms in SDC located on the 2nd floor.\n"
                "• IT Team: SDC has an IT team located on the 2nd floor.\n"
                "• Seminar Halls: SDC has seminar halls located on the 2nd floor for sessions and presentations.\n\n"
                "Employees looking for Techbees training classes, seminar halls, or the SDC IT team should go to SDC 2nd floor."
            ),
        ),
        (
            "SDC — 3rd Floor",
            "SDC 3rd Floor: Laptop and Technical Issue Support Teams",
            (
                "SDC 3rd Floor Support Teams:\n"
                "• Laptop & Technical Support: SDC has teams on the 3rd floor responsible for issues related to laptops "
                "and similar technical queries.\n"
                "• Employee Support: Employees can approach these teams on the 3rd floor of SDC for laptop-related issues, "
                "hardware problems, and technical queries.\n\n"
                "Important Distinction: Do not assume that every IT-related issue must go to the 3rd floor. "
                "The SDC IT team is on the 2nd floor, while the teams specifically handling laptop-related issues "
                "and technical queries are on the 3rd floor of SDC."
            ),
        ),
        (
            "Tower 1 — Overview & Trainees",
            "Tower 1: Trainees, Shared ODCs, and Project ODCs",
            (
                "Tower 1 Overview and Teams:\n"
                "• Trainees: Tower 1 includes trainees. Trainees are located in Tower 1 across its project areas.\n"
                "• ODCs: Tower 1 includes shared ODCs and separate ODCs for different projects.\n"
                "Note: Do not assume that all trainees are located on one specific floor, as the source does not restrict "
                "trainees to a single floor."
            ),
        ),
        (
            "Tower 1 — Basement",
            "Tower 1 Basement: Parking Facility",
            (
                "Tower 1 Basement:\n"
                "• Parking: Tower 1 has parking in the basement.\n"
                "Employees and visitors can park their vehicles in the basement of Tower 1."
            ),
        ),
        (
            "Tower 1 — Ground Floor",
            "Tower 1 Ground Floor: Reception, Play Area, Indoor Games, and Breakout Rooms",
            (
                "Tower 1 Ground Floor Facilities:\n"
                "• Reception: Tower 1 has a reception on the ground floor.\n"
                "• Play Area & Indoor Games: Tower 1 has a play area on the ground floor. The play area includes carrom, "
                "chess, and table tennis. Employees can play table tennis, carrom, and chess on the ground floor of Tower 1.\n"
                "• Breakout Rooms: Tower 1 has breakout rooms located on the ground floor for informal discussions and relaxation."
            ),
        ),
        (
            "Tower 1 — 1st Floor",
            "Tower 1 1st Floor: ODCs and IT Team",
            (
                "Tower 1 1st Floor Facilities and Teams:\n"
                "• ODCs: Tower 1 has ODCs (Offshore Development Centers) on the 1st floor.\n"
                "• IT Team: Tower 1 has an IT team located on the 1st floor.\n"
                "Tower 1 1st floor houses project ODC working areas and the Tower 1 IT team."
            ),
        ),
        (
            "Tower 2 — Overview & Professionals",
            "Tower 2: Professionals, Experienced Teams, and Project ODCs",
            (
                "Tower 2 Overview and Teams:\n"
                "• Professionals & Experienced Teams: Tower 2 includes professionals and experienced teams.\n"
                "• Project ODCs: Professionals and experienced teams have separate ODCs for their respective projects in Tower 2.\n"
                "Floor Specification Notice: The provided office information DOES NOT specify the floor number of the "
                "professional and experienced-team ODCs in Tower 2. If asked which floor experienced teams are on in Tower 2, "
                "state that they have separate project ODCs in Tower 2, but the available records do not specify the floor number."
            ),
        ),
        (
            "Tower 2 — Basement",
            "Tower 2 Basement: Parking Facility",
            (
                "Tower 2 Basement:\n"
                "• Parking: Tower 2 has parking in the basement.\n"
                "Employees and visitors can park their vehicles in the basement of Tower 2."
            ),
        ),
        (
            "Tower 2 — Ground Floor",
            "Tower 2 Ground Floor: Reception, Play Area, Indoor Games, and Breakout Rooms",
            (
                "Tower 2 Ground Floor Facilities:\n"
                "• Reception: Tower 2 has a reception on the ground floor.\n"
                "• Play Area & Indoor Games: Tower 2 has a play area on the ground floor. The play area includes carrom, "
                "chess, and table tennis. Employees can play table tennis, carrom, and chess on the ground floor of Tower 2.\n"
                "• Breakout Rooms: Tower 2 has breakout rooms located on the ground floor for team discussions."
            ),
        ),
        (
            "Campus Facilities Summary",
            "HCL Campus Comprehensive Facilities Summary",
            (
                "HCL Campus Facilities Summary:\n"
                "• Cafeteria: SDC Ground Floor. There is only one cafeteria identified in the office information.\n"
                "• Parking: Available in the Basement of Tower 1 and the Basement of Tower 2.\n"
                "• Play Areas & Indoor Games: Located on the Ground Floor of Tower 1 and the Ground Floor of Tower 2. "
                "Both towers feature table tennis, carrom, and chess in their ground floor play areas.\n"
                "• Breakout Rooms: Available on the Ground Floor of Tower 1 and the Ground Floor of Tower 2.\n"
                "• Techbees Classrooms: SDC 2nd Floor.\n"
                "• Seminar Halls: SDC 2nd Floor.\n"
                "• Laptop & Technical Query Support Teams: SDC 3rd Floor.\n"
                "• IT Teams: SDC 2nd Floor and Tower 1 1st Floor.\n"
                "• Trainees: Tower 1 (floors not restricted to a single floor).\n"
                "• Experienced Teams / Professionals: Tower 2 in separate project ODCs (floor number unspecified in records)."
            ),
        ),
    ]

    chunks = []
    for idx, (section_name, heading, text_body) in enumerate(raw_sections, start=1):
        chunk_id = f"hcl_office_locations_chunk_{idx:03d}"
        combined_text = f"## {heading}\n\n{text_body}"
        chunk = {
            "chunk_id": chunk_id,
            "text": combined_text,
            "metadata": {
                "document_id": DOCUMENT_ID,
                "document_name": DOCUMENT_NAME,
                "source_file": SOURCE_FILE,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "page_start": 1,
                "page_end": 1,
                "section": section_name,
                "knowledge_type": "office_location",
                "organization": "HCL",
                "source_type": "internal_office_knowledge",
                "topic": "location",
                "tenant_id": str(TARGET_TENANT_ID),
            },
        }
        chunks.append(chunk)

    return chunks


def ingest_office_knowledge() -> Dict[str, Any]:
    """Runs end-to-end embedding generation and AWS RDS pgvector insertion."""
    root = get_project_root()
    chunks_file = root / "data" / "processed" / "chunks" / "hcl_office_locations.jsonl"
    embeddings_file = root / "data" / "processed" / "embeddings" / "hcl_office_locations.jsonl"
    master_embeddings_file = root / "data" / "processed" / "embeddings" / "embeddings.jsonl"

    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build Chunks
    logger.info("Building office knowledge chunks...")
    chunk_records = get_office_knowledge_chunks()
    logger.info("Generated %d chunks for '%s'", len(chunk_records), DOCUMENT_NAME)

    # Write chunks JSONL
    with open(chunks_file, "w", encoding="utf-8") as f:
        for c in chunk_records:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Saved chunks to: %s", chunks_file)

    # 2. Generate 384-dimensional Embeddings via local BGE model
    logger.info("Loading local embedding provider (BAAI/bge-small-en-v1.5, dimension=384)...")
    start_time = time.time()
    provider = LocalSentenceTransformerEmbeddingProvider(
        model_name="BAAI/bge-small-en-v1.5",
    )

    texts = [c["text"] for c in chunk_records]
    logger.info("Generating dense vector embeddings for %d chunks...", len(texts))
    embeddings = provider.embed_texts(texts)
    embedding_duration = time.time() - start_time
    logger.info("Generated %d embeddings in %.2f seconds", len(embeddings), embedding_duration)

    # 3. Create embedded records
    embedded_records = []
    for chunk, emb in zip(chunk_records, embeddings):
        record = {
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": emb,
            "metadata": chunk["metadata"],
        }
        embedded_records.append(record)

    # Write office embeddings JSONL
    with open(embeddings_file, "w", encoding="utf-8") as f:
        for r in embedded_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Saved embeddings to: %s", embeddings_file)

    # Append to master embeddings.jsonl if not already present
    if master_embeddings_file.exists():
        existing_lines = master_embeddings_file.read_text(encoding="utf-8").splitlines()
        existing_ids = set()
        for line in existing_lines:
            if line.strip():
                try:
                    data = json.loads(line)
                    existing_ids.add(data.get("chunk_id"))
                except Exception:
                    pass

        new_to_append = [r for r in embedded_records if r["chunk_id"] not in existing_ids]
        if new_to_append:
            with open(master_embeddings_file, "a", encoding="utf-8") as f:
                for r in new_to_append:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info("Appended %d new office embeddings to master: %s", len(new_to_append), master_embeddings_file)

    # 4. Import into AWS RDS pgvector
    logger.info("Importing into AWS RDS PostgreSQL pgvector...")
    session_factory = get_session_factory()
    store = VectorStore(target_dimension=384)

    with session_factory() as session:
        # Pre-import verification of existing records
        initial_chunks = session.execute(text("SELECT COUNT(*) FROM chunks;")).scalar() or 0
        initial_embeddings = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings;")).scalar() or 0
        logger.info("Pre-import counts in DB: chunks=%d, embeddings=%d", initial_chunks, initial_embeddings)

        summary = store.import_embeddings_from_file(
            session=session,
            jsonl_path=embeddings_file,
            tenant_id=TARGET_TENANT_ID,
            allow_mock=False,
        )
        session.commit()

        # Post-import verification
        post_chunks = session.execute(text("SELECT COUNT(*) FROM chunks;")).scalar() or 0
        post_embeddings = session.execute(text("SELECT COUNT(*) FROM chunk_embeddings;")).scalar() or 0
        new_office_chunks_in_db = session.execute(
            text("SELECT COUNT(*) FROM chunks WHERE document_id = :doc_id;"),
            {"doc_id": DOCUMENT_ID},
        ).scalar() or 0

        logger.info("Import Summary:")
        logger.info("  Documents processed:  %d", summary.documents_processed)
        logger.info("  Chunks processed:     %d", summary.chunks_processed)
        logger.info("  Embeddings processed: %d", summary.embeddings_processed)
        logger.info("  Total chunks in DB:   %d (+%d)", post_chunks, post_chunks - initial_chunks)
        logger.info("  Total embeddings:     %d (+%d)", post_embeddings, post_embeddings - initial_embeddings)
        logger.info("  Office chunks in DB:  %d", new_office_chunks_in_db)

    return {
        "success": True,
        "document_id": DOCUMENT_ID,
        "new_chunks_count": len(chunk_records),
        "new_embeddings_count": len(embeddings),
        "embedding_duration": embedding_duration,
        "initial_chunks": initial_chunks,
        "post_chunks": post_chunks,
        "initial_embeddings": initial_embeddings,
        "post_embeddings": post_embeddings,
        "office_chunks_in_db": new_office_chunks_in_db,
    }


if __name__ == "__main__":
    result = ingest_office_knowledge()
    print("\n" + "=" * 60)
    print("      HCL OFFICE KNOWLEDGE INGESTION COMPLETE")
    print("=" * 60)
    print(f"Document ID:           {result['document_id']}")
    print(f"New Chunks Count:      {result['new_chunks_count']}")
    print(f"New Embeddings Count:  {result['new_embeddings_count']}")
    print(f"Embedding Duration:    {result['embedding_duration']:.2f} seconds")
    print(f"Total Chunks in DB:    {result['post_chunks']} (was {result['initial_chunks']})")
    print(f"Total Embeddings in DB:{result['post_embeddings']} (was {result['initial_embeddings']})")
    print(f"Office Chunks in DB:   {result['office_chunks_in_db']}")
    print("=" * 60)
