
---

## Module 4 — RAG Answer Engine

> **Scope**: Retrieval-Augmented Generation pipeline — question embedding, pgvector retrieval, relevance filtering, context construction, LLM answer generation, source attribution. Does **NOT** include LangGraph agents, voice, authentication, calendar, or notifications.

> **IMPORTANT**: The current RDS database contains **886 mock embeddings** (`is_mock=True`). These are deterministic test vectors and **NOT production semantic embeddings**. RAG retrieval with mock vectors will not produce meaningful semantic results. See "Production Embedding Transition" below.

### Architecture

```
Employee Question
       |
       v
Query Embedding (OpenAI text-embedding-3-small)
       |
       v
pgvector Cosine Similarity Search (tenant_id mandatory)
       |
       v
Mock/Production Filter (is_mock check)
       |
       v
Similarity Threshold Filter (RAG_MIN_SIMILARITY=0.70)
       |
       v
Context Builder (top-K chunks -> LLM prompt block)
       |
       v
OpenAI LLM (strict grounding system prompt)
       |
       v
RAGResponse {answer, sources[], confidence, mode, latency_ms}
```

### Files Created

| File | Purpose |
|---|---|
| `backend/app/rag/retriever.py` | Query embedding + tenant-isolated pgvector search + mock/production filter |
| `backend/app/rag/context_builder.py` | Formats retrieved chunks into LLM prompt context + SourceReference list |
| `backend/app/rag/answer_generator.py` | OpenAI Chat Completions with strict grounding prompt + retry |
| `backend/app/rag/rag_service.py` | High-level pipeline orchestrator (`RAGService.answer_question()`) |
| `backend/app/rag/evaluate_rag.py` | Manual evaluation script for policy questions |
| `backend/app/schemas/chat.py` | Pydantic v2 request/response schemas |
| `backend/app/routes/chat.py` | `POST /api/chat` FastAPI endpoint |
| `backend/app/main.py` | FastAPI application entry point |
| `tests/test_rag_retriever.py` | 9 tests: query embedding, retrieval, mock rejection, isolation |
| `tests/test_context_builder.py` | 16 tests: context structure, chunk limit, source preservation |
| `tests/test_answer_generator.py` | 11 tests: empty context, grounding prompt, retries, API key safety |
| `tests/test_rag_service.py` | 20 tests: full pipeline, cross-tenant, confidence, error handling |
| `tests/test_chat_route.py` | 16 tests: response structure, validation, no-match, secrets |

### Mock vs Production Safety

```
RAG_ALLOW_MOCK=false  (default = production mode)
```

| Mode | Behavior |
|---|---|
| `RAG_ALLOW_MOCK=false` | Only `is_mock=False` embeddings used in retrieval |
| `RAG_ALLOW_MOCK=true` | Mock vectors allowed — DEVELOPMENT ONLY — NOT semantically valid |

The system **never silently mixes** mock and production vectors in the same result.

### Confidence Label

The `confidence` field is a **RETRIEVAL RELEVANCE INDICATOR** — NOT a calibrated probability.

| Label | Top Score Threshold (configurable) |
|---|---|
| `HIGH` | >= RAG_CONFIDENCE_HIGH (default: 0.85) |
| `MEDIUM` | >= RAG_CONFIDENCE_MEDIUM (default: 0.70) |
| `LOW` | >= RAG_CONFIDENCE_LOW (default: 0.50) |
| `NO_MATCH` | No sources retrieved or all below threshold |

### Configuration (.env)

| Variable | Default | Purpose |
|---|---|---|
| `RAG_TOP_K` | `8` | pgvector candidates to retrieve |
| `RAG_MIN_SIMILARITY` | `0.70` | Minimum similarity threshold |
| `RAG_MAX_CONTEXT_CHUNKS` | `5` | Max chunks sent to LLM |
| `RAG_ALLOW_MOCK` | `false` | Allow mock vectors in retrieval |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `LLM_TIMEOUT_SECONDS` | `30` | LLM request timeout |
| `LLM_MAX_RETRIES` | `2` | Retries on transient errors |

### API Endpoint

**`POST /api/chat`** — RAG-powered company policy Q&A.

Request:
```json
{
  "question": "How many days of annual leave can I take?",
  "tenant_id": "00000000-0000-0000-0000-000000000001"
}
```

> **Development note**: `tenant_id` is accepted in the request body until Module 8 (Authentication) replaces it with JWT identity.

Response:
```json
{
  "answer": "Employees are entitled to 20 days of annual leave per year...",
  "sources": [{"document_name": "Leave and Holiday Policy", "page_start": 12, "page_end": 13, "section": "Annual Leave", "chunk_id": "leave_policy_annual_0001", "similarity_score": 0.92}],
  "retrieved_count": 8,
  "used_context_count": 2,
  "confidence": "HIGH",
  "mode": "production",
  "model": "gpt-4o-mini",
  "latency_ms": 1240.5,
  "error": null
}
```

### Running the API

```powershell
# From project root
py -m uvicorn backend.app.main:app --reload --port 8000

# API docs: http://localhost:8000/docs
```

### Testing

```powershell
# All 141 tests (69 previous + 72 new Module 4)
py -m pytest tests/ -v

# Module 4 tests only
py -m pytest tests/test_rag_retriever.py tests/test_context_builder.py tests/test_answer_generator.py tests/test_rag_service.py tests/test_chat_route.py -v
```

### Evaluation

```powershell
# Mock mode (no OpenAI credits spent)
set RAG_ALLOW_MOCK=true && py backend/app/rag/evaluate_rag.py

# Production mode (requires real embeddings + OPENAI_API_KEY)
py backend/app/rag/evaluate_rag.py
```

### Production Embedding Transition

When ready to use real semantic retrieval:

1. Set `OPENAI_API_KEY` in `backend/.env`
2. Generate real embeddings: `py backend/app/rag/embeddings.py`
3. Import to RDS (no `--allow-mock`): `py backend/app/rag/vector_store.py`
4. Keep `RAG_ALLOW_MOCK=false` (default) in production
5. Run evaluation: `py backend/app/rag/evaluate_rag.py`
