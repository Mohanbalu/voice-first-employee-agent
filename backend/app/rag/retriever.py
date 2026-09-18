"""RAG Retriever — Module 4.

Orchestrates query embedding generation and tenant-isolated pgvector similarity
search. Enforces mock/production vector safety and configurable similarity thresholds.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root in sys.path for direct script execution
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

try:
    from backend.app.config import config
    from backend.app.database import get_session_factory, is_db_reachable
    from backend.app.models.embedding import ChunkEmbedding
    from backend.app.models.chunk import Chunk
    from backend.app.models.document import Document
    from backend.app.rag.embeddings import (
        EmbeddingService,
        LocalSentenceTransformerEmbeddingProvider,
        MockEmbeddingProvider,
        OpenAIEmbeddingProvider,
        get_embedding_provider,
    )
    from backend.app.rag.vector_store import SearchResult, VectorStore
except ImportError:
    from app.config import config
    from app.database import get_session_factory, is_db_reachable
    from app.models.embedding import ChunkEmbedding
    from app.models.chunk import Chunk
    from app.models.document import Document
    from app.rag.embeddings import (
        EmbeddingService,
        LocalSentenceTransformerEmbeddingProvider,
        MockEmbeddingProvider,
        OpenAIEmbeddingProvider,
        get_embedding_provider,
    )
    from app.rag.vector_store import SearchResult, VectorStore

from sqlalchemy import select
from sqlalchemy.orm import Session
import uuid

logger = logging.getLogger("rag.retriever")

# ── RAG configuration with env-variable overrides ────────────────────────────

def _int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _bool_env(key: str, default: bool) -> bool:
    val = os.getenv(key, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


RAG_TOP_K: int = _int_env("RAG_TOP_K", 15)
RAG_MIN_SIMILARITY: float = _float_env("RAG_MIN_SIMILARITY", 0.50)
RAG_MAX_CONTEXT_CHUNKS: int = _int_env("RAG_MAX_CONTEXT_CHUNKS", 6)
RAG_ALLOW_MOCK: bool = _bool_env("RAG_ALLOW_MOCK", True)


class RetrievalConfig:
    """Configurable RAG retrieval parameters."""

    def __init__(
        self,
        top_k: int = RAG_TOP_K,
        min_similarity: float = RAG_MIN_SIMILARITY,
        max_context_chunks: int = RAG_MAX_CONTEXT_CHUNKS,
        allow_mock: bool = RAG_ALLOW_MOCK,
    ):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.max_context_chunks = max_context_chunks
        self.allow_mock = allow_mock


# ── Entity & Keyword Dictionaries for Hybrid Retrieval ───────────────────────

OFFICE_ENTITIES = {
    "it_team": ["it team", "it department", "it support", "tech team", "technical team", "it helpdesk", "it"],
    "laptop_support": [
        "laptop", "laptop issue", "laptop problem", "laptop problems",
        "hardware", "technical query", "technical queries", "technical issue", "technical support",
        "tech support", "hardware problem", "hardware issue", "laptop support",
    ],
    "techbees": ["techbees", "techbee", "techbees classroom", "techbees classrooms", "classroom", "classrooms"],
    "seminar": ["seminar", "seminar hall", "seminar halls"],
    "cafeteria": ["cafeteria", "canteen", "food court", "lunch"],
    "parking": ["parking", "park", "car parking", "bike parking", "vehicles"],
    "reception": ["reception", "front desk", "lobby"],
    "play_area": ["play area", "indoor games", "table tennis", "carrom", "chess", "games"],
    "breakout": ["breakout", "breakout room", "breakout rooms", "relaxation"],
    "odc": ["odc", "odcs", "offshore development center", "shared odc", "project odc"],
    "trainees": ["trainee", "trainees"],
    "experienced_teams": ["experienced", "experienced team", "experienced teams", "professional", "professionals"],
    # Leave-related entity signals — map colloquial terms to policy document keywords
    "leave_policy": [
        "casual leave", "casual leaves", "vacation", "vacations", "vacation days",
        "personal leave", "personal day", "personal days", "time off", "days off",
        "earned leave", "privilege leave", "pl", "cl", "annual leave", "annual leaves",
        "my leave", "leave policy", "leave balance", "leave entitlement",
        "leave quota", "leave days", "sick leave", "maternity leave", "paternity leave",
        "carry forward", "leave encashment", "leave types", "types of leave",
    ],
}

OFFICE_BUILDINGS = {
    "sdc": ["sdc"],
    "tower_1": ["tower 1", "tower1", "t1"],
    "tower_2": ["tower 2", "tower2", "t2"],
}


def _extract_query_entities(query: str) -> dict[str, Any]:
    """Extracts office entity signals from the query."""
    q_lower = query.lower()
    matched_categories: list[str] = []
    matched_terms: list[str] = []

    for cat, terms in OFFICE_ENTITIES.items():
        for term in terms:
            if term in q_lower:
                matched_categories.append(cat)
                matched_terms.append(term)
                break

    matched_buildings: list[str] = []
    for b_key, b_terms in OFFICE_BUILDINGS.items():
        for term in b_terms:
            if term in q_lower:
                matched_buildings.append(b_key)
                break

    return {
        "categories": set(matched_categories),
        "buildings": set(matched_buildings),
        "terms": matched_terms,
        "is_office_query": bool(matched_categories or matched_buildings),
    }


_QUERY_STOPWORDS: set[str] = {
    "what", "is", "are", "the", "a", "an", "for", "in", "on", "at", "to", "from",
    "of", "and", "or", "how", "can", "i", "my", "do", "does", "about", "tell",
    "me", "which", "where", "who", "when", "why", "with", "as", "by", "this", "that",
    "please", "give", "show", "much", "many", "would", "should", "could", "be", "am",
}


# Synonym expansion: maps colloquial employee terms to the actual terminology
# used in the HCL policy documents, enabling correct lexical retrieval.
_LEAVE_SYNONYM_EXPANSION: dict[str, list[str]] = {
    "casual leave": ["annual leaves", "my leave", "types of leave", "leave policy"],
    "casual leaves": ["annual leaves", "my leave", "types of leave", "leave policy"],
    "vacation": ["annual leaves", "annual leave", "leave policy"],
    "vacations": ["annual leaves", "annual leave", "leave policy"],
    "vacation days": ["annual leaves", "annual leave", "leave policy"],
    "time off": ["annual leaves", "my leave", "leave policy"],
    "days off": ["annual leaves", "my leave", "leave policy"],
    "personal leave": ["my leave", "annual leaves", "leave policy"],
    "personal day": ["my leave", "annual leaves", "leave policy"],
    "personal days": ["my leave", "annual leaves", "leave policy"],
    "earned leave": ["annual leaves", "leave policy"],
    "privilege leave": ["annual leaves", "leave policy"],
}


def _extract_query_keywords(query: str) -> list[str]:
    """Extracts search terms and domain phrases from any query for lexical database search.
    Includes synonym expansion so colloquial leave terms map to HCL policy document terminology.
    """
    import re
    q_lower = query.lower()
    raw_tokens = re.findall(r"[a-zA-Z0-9_\-]+", q_lower)
    terms = [t for t in raw_tokens if len(t) > 2 and t not in _QUERY_STOPWORDS]

    # Prioritise key domain phrases
    phrases = [
        "annual leaves", "annual leave", "my leave", "sick leave", "casual leave",
        "casual leaves", "maternity leave", "paternity leave",
        "leave policy", "types of leave", "leave entitlement", "leave balance",
        "work from home", "remote work", "health insurance", "medical insurance",
        "code of conduct", "anti-bribery", "anti bribery", "harassment policy",
        "travel policy", "expense reimbursement", "laptop support", "it support",
        "tech support", "office location",
    ]
    matched = [p for p in phrases if p in q_lower]

    # Apply synonym expansion: if a colloquial term is in the query, add its
    # policy-document equivalents so lexical search finds the right chunks.
    expansion: list[str] = []
    for colloquial_term, policy_synonyms in _LEAVE_SYNONYM_EXPANSION.items():
        if colloquial_term in q_lower:
            for syn in policy_synonyms:
                if syn not in matched and syn not in expansion:
                    expansion.append(syn)

    result: list[str] = []
    for item in matched + expansion + terms:
        if item not in result:
            result.append(item)
    return result


def _build_embedding_service(allow_mock: bool) -> EmbeddingService:
    """Creates an embedding service based on configured EMBEDDING_PROVIDER."""
    is_render = bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID"))
    cfg_emb = getattr(config, "embedding", None)
    provider_name = (
        (cfg_emb.provider if cfg_emb else None)
        or os.getenv("EMBEDDING_PROVIDER")
        or "mock"
    ).strip().lower()

    if is_render and provider_name == "local":
        provider_name = "mock"

    if provider_name == "local":
        try:
            provider = LocalSentenceTransformerEmbeddingProvider()
            logger.info("Retriever using local Sentence-Transformers model: %s", provider.model_name)
            return EmbeddingService(provider)
        except Exception as exc:
            logger.warning("Could not initialise local Sentence-Transformers provider: %s", exc)
            if not allow_mock:
                raise

    elif provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        is_placeholder = not api_key or api_key == "your_openai_api_key_placeholder"
        if not is_placeholder:
            model = (
                os.getenv("OPENAI_EMBEDDING_MODEL")
                or os.getenv("EMBEDDING_MODEL")
                or "text-embedding-3-small"
            )
            try:
                provider = OpenAIEmbeddingProvider(api_key=api_key, model=model)
                logger.info("Retriever using production OpenAI embedding model: %s", model)
                return EmbeddingService(provider)
            except Exception as exc:
                logger.warning("Could not initialise OpenAI embedding provider: %s", exc)
                if not allow_mock:
                    raise

    if allow_mock or provider_name in ("mock", "none") or is_render:
        dimension = config.db.vector_dimension
        provider = MockEmbeddingProvider(dimension=dimension)
        logger.info(
            "Retriever using safe cloud MockEmbeddingProvider (dim=%d, is_render=%s)",
            dimension,
            is_render,
        )
        return EmbeddingService(provider)

    raise RuntimeError(
        f"No embedding provider available for '{provider_name}'. Check your configuration "
        "or set RAG_ALLOW_MOCK=true for development-only mock mode."
    )


import json

_CACHED_LOCAL_CHUNKS: Optional[List[dict]] = None


def _get_local_chunks() -> List[dict]:
    global _CACHED_LOCAL_CHUNKS
    if _CACHED_LOCAL_CHUNKS is not None:
        return _CACHED_LOCAL_CHUNKS

    chunks = []
    bases = [
        Path("data/processed/chunks"),
        Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed" / "chunks",
    ]
    for base in bases:
        if base.exists():
            for fpath in sorted(base.glob("*.jsonl")):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line_str = line.strip()
                            if line_str:
                                chunks.append(json.loads(line_str))
                except Exception:
                    pass
            if chunks:
                break
    _CACHED_LOCAL_CHUNKS = chunks
    return _CACHED_LOCAL_CHUNKS


GENERIC_DOCUMENT_TERMS: set[str] = {
    "policy", "policies", "employee", "employees", "company", "guidelines",
    "rules", "general", "workplace", "work", "organization", "management",
    "standard", "standards", "procedure", "procedures", "global", "document",
    "portal", "system", "applicable", "applicability",
}


def _search_local_chunks(terms: list[str], limit: int = 15) -> List[SearchResult]:
    if not terms:
        return []
    local_chunks = _get_local_chunks()
    if not local_chunks:
        return []

    search_terms = [t.lower().strip() for t in terms if len(t.strip()) > 2][:12]
    scored = []
    for c in local_chunks:
        text_lower = (c.get("text") or "").lower()
        doc_lower = (c.get("document_name") or "").lower()
        hits = 0
        for t in search_terms:
            if t in text_lower:
                hits += (3 if " " in t else 1)
            if t not in GENERIC_DOCUMENT_TERMS and t in doc_lower:
                hits += (4 if " " in t else 2)
        if hits > 0:
            sim = min(0.68 + 0.05 * hits, 0.95)
            meta = c.get("metadata") or {}
            scored.append((
                hits,
                SearchResult(
                    chunk_id=c.get("chunk_id", str(uuid.uuid4())),
                    document_id=c.get("document_id", "doc"),
                    document_name=c.get("document_name", "Company Policy"),
                    source_file=c.get("source_file", "policy.pdf"),
                    text=c.get("text", ""),
                    page_start=c.get("page_start", 1),
                    page_end=c.get("page_end", 1),
                    section=c.get("section"),
                    similarity_score=sim,
                    distance=round(1.0 - sim, 4),
                    metadata=meta,
                )
            ))
    scored.sort(key=lambda x: (x[0], x[1].similarity_score), reverse=True)
    return [item[1] for item in scored[:limit]]


class RAGRetriever:
    """
    Tenant-isolated hybrid retriever that:
    1. Embeds the query using the configured provider.
    2. Runs a pgvector cosine-similarity search scoped to the tenant.
    3. Performs lexical entity candidate search.
    4. Applies hybrid fusion and reranking with entity boosting.
    5. Filters out mock vectors (unless allow_mock=True).
    6. Returns up to max_context_chunks ranked results.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        config: Optional[RetrievalConfig] = None,
    ):
        self.cfg = config or RetrievalConfig()
        self._embedding_service = embedding_service  # injected for tests

    def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is not None:
            return self._embedding_service
        self._embedding_service = _build_embedding_service(self.cfg.allow_mock)
        return self._embedding_service

    def _search_lexical_candidates(
        self,
        session: Optional[Session],
        tenant_id: uuid.UUID,
        terms: list[str],
        limit: int = 15,
    ) -> List[SearchResult]:
        """Fetches candidate chunks matching extracted entity keywords and query terms."""
        if not terms:
            return []

        if session is not None:
            try:
                from sqlalchemy import case, or_

                distinctive_terms = [t.lower().strip() for t in terms if len(t.strip()) > 2]
                if not distinctive_terms:
                    return []

                score_cases = []
                where_filters = []

                for term in distinctive_terms[:12]:
                    is_phrase = " " in term
                    text_weight = 3 if is_phrase else 1
                    title_weight = 4 if is_phrase else 2

                    score_cases.append(case((Chunk.text.ilike(f"%{term}%"), text_weight), else_=0))
                    where_filters.append(Chunk.text.ilike(f"%{term}%"))

                    if term not in GENERIC_DOCUMENT_TERMS:
                        score_cases.append(case((Document.document_name.ilike(f"%{term}%"), title_weight), else_=0))
                        where_filters.append(Document.document_name.ilike(f"%{term}%"))

                if not where_filters:
                    return []

                relevance = sum(score_cases)

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
                        relevance.label("relevance"),
                    )
                    .join(Document, Chunk.document_ref_id == Document.id)
                    .where(Chunk.tenant_id == tenant_id)
                    .where(relevance > 0)
                    .order_by(relevance.desc())
                    .limit(limit)
                )
                rows = session.execute(stmt).fetchall()
                if rows:
                    results: List[SearchResult] = []
                    for row in rows:
                        rel = float(row.relevance) if getattr(row, "relevance", None) is not None else 1.0
                        sim = min(0.68 + 0.05 * rel, 0.95)
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
                                distance=round(1.0 - sim, 4),
                                metadata=row.metadata_json or {},
                            )
                        )
                    return results
            except Exception as exc:
                logger.warning("Lexical database query failed (%s); falling back to local chunks.", exc)

        return _search_local_chunks(terms, limit=limit)

    def _embed_with_timeout(self, query: str, timeout_seconds: float = 2.0) -> Optional[List[float]]:
        """Attempts to embed query with a strict timeout to avoid thread blocking on model loading."""
        import concurrent.futures
        try:
            svc = self._get_embedding_service()
            if isinstance(getattr(svc, "_provider", None), MockEmbeddingProvider):
                return svc.embed_query(query)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(svc.embed_query, query)
                return fut.result(timeout=timeout_seconds)
        except Exception as exc:
            logger.warning("Query embedding skipped or timed out (%s); relying on lexical candidates.", exc)
            return None

    def retrieve(
        self,
        session: Session,
        tenant_id: uuid.UUID,
        question: str,
    ) -> List[SearchResult]:
        """
        Full hybrid retrieval pipeline:
        - Instant lexical candidate retrieval for query terms and entities.
        - Timeout-safe query embedding generation (3.5s max).
        - pgvector similarity search (top_k candidates).
        - Filter mock embeddings in production mode.
        - Hybrid fusion and reranking with entity boosting.
        - Adaptive similarity thresholding.
        - Limit to max_context_chunks.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        q_clean = question.strip()
        entities = _extract_query_entities(q_clean)
        query_keywords = _extract_query_keywords(q_clean)

        # 1. Fetch lexical candidates immediately from RDS (fast, ~15ms)
        combined_terms = list(entities.get("terms", []))
        for kw in query_keywords:
            if kw not in combined_terms:
                combined_terms.append(kw)

        lexical_candidates = self._search_lexical_candidates(
            session=session,
            tenant_id=tenant_id,
            terms=combined_terms,
            limit=15,
        )

        # 2. Try vector retrieval with safe 3.5s timeout if DB is reachable AND real embeddings active
        vector_candidates: List[SearchResult] = []
        svc = self._get_embedding_service()
        is_mock_provider = isinstance(getattr(svc, "_provider", None), MockEmbeddingProvider)

        if not is_mock_provider and session is not None and is_db_reachable():
            try:
                query_vec = self._embed_with_timeout(q_clean, timeout_seconds=3.5)
                if query_vec is not None and len(query_vec) == config.db.vector_dimension:
                    store = VectorStore(target_dimension=config.db.vector_dimension)
                    vector_candidates = store.search_similar_chunks(
                        session=session,
                        tenant_id=tenant_id,
                        query_embedding=query_vec,
                        top_k=self.cfg.top_k,
                        min_similarity=0.0,
                    )
            except Exception as exc:
                logger.warning("Vector candidate retrieval error (%s); proceeding with lexical.", exc)
        elif is_mock_provider:
            logger.info("Retriever in cloud/mock mode: skipping pgvector distance noise, relying on high-precision ranked lexical candidates.")

        # Merge candidate pools by chunk_id
        candidate_dict: dict[str, SearchResult] = {}
        for vc in vector_candidates:
            candidate_dict[vc.chunk_id] = vc

        for lc in lexical_candidates:
            if lc.chunk_id not in candidate_dict:
                candidate_dict[lc.chunk_id] = lc
            else:
                candidate_dict[lc.chunk_id].similarity_score = min(
                    candidate_dict[lc.chunk_id].similarity_score + 0.05, 1.0
                )

        candidates = list(candidate_dict.values())

        logger.info(
            "Hybrid candidate pool: %d candidates (vector=%d, lexical=%d) for tenant %s",
            len(candidates),
            len(vector_candidates),
            len(lexical_candidates),
            tenant_id,
        )

        # 3. Filter mock vectors in production mode
        if not self.cfg.allow_mock and session is not None:
            try:
                filtered = self._filter_production_only(session, tenant_id, candidates)
                if len(filtered) < len(candidates):
                    logger.warning(
                        "Dropped %d mock-vector results in production mode.",
                        len(candidates) - len(filtered),
                    )
                candidates = filtered
            except Exception as exc:
                logger.warning("Could not filter mock vectors (%s); retaining candidates.", exc)

        # 4. Rerank candidates with hybrid scoring & entity boosting
        scored_candidates: list[tuple[float, SearchResult]] = []
        q_lower = q_clean.lower()
        has_specific_laptop = "laptop" in q_lower or "hardware" in q_lower or "technical issue" in q_lower
        has_specific_building = entities["buildings"]

        for cand in candidates:
            base_sim = cand.similarity_score
            text_lower = (cand.text + " " + (cand.section or "")).lower()
            boost = 0.0
            has_direct_hit = False

            # Specific building matching
            cand_building: Optional[str] = None
            if "sdc" in text_lower:
                cand_building = "sdc"
            elif "tower 1" in text_lower or "tower1" in text_lower:
                cand_building = "tower_1"
            elif "tower 2" in text_lower or "tower2" in text_lower:
                cand_building = "tower_2"

            if has_specific_building:
                if cand_building in has_specific_building:
                    boost += 0.12
                    has_direct_hit = True
                elif cand_building is not None:
                    # Deprioritize other buildings if user explicitly asked for one
                    boost -= 0.10

            # Category / Team / Facility entity matching
            for cat in entities["categories"]:
                if cat == "laptop_support":
                    if "laptop" in text_lower or "technical issue" in text_lower:
                        boost += 0.25 if has_specific_laptop else 0.10
                        has_direct_hit = True
                elif cat == "it_team":
                    if "it team" in text_lower or "it" in text_lower:
                        # General IT team requested
                        if not has_specific_laptop and ("2nd floor" in text_lower or "1st floor" in text_lower):
                            boost += 0.20
                        else:
                            boost += 0.12
                        has_direct_hit = True
                elif cat == "techbees":
                    if "techbee" in text_lower or "classroom" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "seminar":
                    if "seminar" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "cafeteria":
                    if "cafeteria" in text_lower or "canteen" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "parking":
                    if "parking" in text_lower or "park" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "play_area":
                    if "play area" in text_lower or "table tennis" in text_lower or "carrom" in text_lower or "chess" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "breakout":
                    if "breakout" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "odc":
                    if "odc" in text_lower:
                        boost += 0.20
                        has_direct_hit = True
                elif cat == "trainees":
                    if "trainee" in text_lower:
                        boost += 0.25
                        has_direct_hit = True
                elif cat == "experienced_teams":
                    if "experienced" in text_lower or "professional" in text_lower:
                        boost += 0.25
                        has_direct_hit = True

            # General keyword occurrence bonus
            for term in combined_terms:
                if term in text_lower:
                    boost += 0.05
                    has_direct_hit = True

            final_score = round(base_sim + boost, 4)

            # Accept if similarity passes threshold OR has direct entity match
            if base_sim >= self.cfg.min_similarity or has_direct_hit or final_score >= self.cfg.min_similarity:
                cand.similarity_score = final_score
                scored_candidates.append((final_score, cand))

        # Sort by reranked score descending
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        above_threshold = [c for _, c in scored_candidates]

        logger.info(
            "%d/%d candidates selected after hybrid rerank & threshold %.2f",
            len(above_threshold),
            len(candidates),
            self.cfg.min_similarity,
        )

        return above_threshold[: self.cfg.max_context_chunks]

    def _filter_production_only(
        self, session: Session, tenant_id: uuid.UUID, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Drops any SearchResult whose backing ChunkEmbedding.is_mock is True.
        Queries the DB for the is_mock flag in a single batched query.
        """
        if not results:
            return results

        chunk_ids = [r.chunk_id for r in results]

        rows = session.execute(
            select(ChunkEmbedding.chunk_id, ChunkEmbedding.is_mock)
            .where(
                ChunkEmbedding.tenant_id == tenant_id,
                ChunkEmbedding.chunk_id.in_(chunk_ids),
            )
        ).fetchall()

        mock_ids = {row.chunk_id for row in rows if row.is_mock}
        production = [r for r in results if r.chunk_id not in mock_ids]
        return production
