# Voice-First Agentic AI Employee Workplace Assistant

An enterprise-grade, voice-first intelligent workplace assistant designed to automate daily employee workflows, streamline HR/IT escalations, provide indoor office navigation, and answer company policy inquiries via RAG.

## Project Features
1. **Voice-first AI employee assistant**: Natural voice interaction (STT + TTS) for hands-free queries and commands.
2. **Company policy/document Q&A using RAG**: Retrieval-augmented generation over uploaded employee handbooks and manuals.
3. **Agentic HR/IT ticket escalation**: Intelligent diagnosis, routing, and escalation to HR/IT systems.
4. **Automatic HR/IT email communication**: Automated follow-up drafts and notifications.
5. **Ticket follow-up and employee notification**: Proactive status updates on unresolved requests.
6. **Location-aware office navigation**: Indoor navigation graph across buildings, floors, and rooms.
7. **Meeting/calendar scheduling**: Voice-driven calendar booking and room reservations.
8. **Tasks and reminders**: Priority task tracking and schedule reminders.
9. **Timesheet assistant**: Guided weekly timesheet logging and validation.
10. **Important message detection and notifications**: Triage critical announcements and urgent emails.
11. **Employee dashboard**: Clean interface for viewing personal tasks, meetings, tickets, and assistant history.
12. **HR/Admin dashboard**: Centralized administration for employee records, documents, tickets, and floor plans.

## Tech Stack
- **Frontend**: React.js + Vite
- **Backend**: Python + FastAPI
- **Database**: PostgreSQL
- **Vector Database**: PostgreSQL + pgvector
- **AI/LLM**: Groq API (Primary), OpenAI & Puter AI (Configurable Options)
- **Agent Orchestration**: LangGraph
- **RAG Pipeline**: Ingestion + chunking + embeddings + vector retrieval
- **Voice Pipeline**: Speech-to-Text (STT) + Text-to-Speech (TTS)
- **Authentication**: JWT (JSON Web Tokens)
- **Indoor Navigation**: Building/floor/room graph with QR/BLE/GPS support

## Project Structure
```
voice-first-employee-agent/
├── backend/          # FastAPI application, LangGraph agents, RAG, and database
├── frontend/         # React.js + Vite web application
├── data/             # Policies, office maps, and database seed scripts
├── tests/            # Automated test suite
└── docs/             # Architecture, API specifications, and research notes
```

## Knowledge Base & RAG Pipeline

### Module 3.2 — Local Semantic Embeddings (Sentence-Transformers)

- **Purpose**: Generates local, high-performance dense semantic vector embeddings for all 886 validated knowledge chunks in `data/processed/chunks/`, fully offline on CPU without spending third-party API credits.
- **Model Configuration**: Default model is `BAAI/bge-small-en-v1.5` producing 384-dimensional normalized vectors, loaded via `sentence-transformers`.
- **Query Instruction**: Following BGE specifications, queries are automatically prefixed with `"Represent this sentence for searching relevant passages: "`, while document chunks are embedded as raw text.
- **Environment Variables**:
  - `EMBEDDING_PROVIDER`: `local` (default), `openai`, or `mock`.
  - `EMBEDDING_MODEL`: Target model (default: `BAAI/bge-small-en-v1.5`).
  - `EMBEDDING_DIMENSION`: Target vector dimension (default: `384`).
  - `EMBEDDING_BATCH_SIZE`: Number of chunks encoded per CPU batch (default: `64`).
  - `VECTOR_DIMENSION`: PostgreSQL/pgvector column dimension (default: `384`).
- **How to Run**:
  - **Local Model (Default, 886 real embeddings)**:
    ```powershell
    py backend/app/rag/embeddings.py --replace-mock
    ```
  - **Offline / Mock Mode (Testing deterministic vectors)**:
    ```powershell
    py backend/app/rag/embeddings.py --mock
    ```
  - **Force Rebuild (Ignoring existing checkpoints)**:
    ```powershell
    py backend/app/rag/embeddings.py --force
    ```
- **Output Files**:
  - `data/processed/embeddings/embeddings.jsonl`: Line-delimited JSON records with `chunk_id`, `text`, `embedding` vector (384 floats), and preserved `metadata`.
  - `data/processed/embeddings/embedding_validation_report.json`: Verification report detailing vector counts, dimensions, model, and integrity.
- **Checkpoint & Resume**:
  - Automatically scans `embeddings.jsonl` and validates vector dimension against `EMBEDDING_DIMENSION`. Automatically invalidates checkpoints if dimension differs.
- **Validation**:
  - Checks that all 886 vectors are non-empty arrays of numeric floats matching dimension 384 with 0 missing fields.
- **Testing Instructions**:
  ```powershell
  py -m pytest tests/test_local_embeddings.py tests/test_embeddings.py -v
  ```
> **Note**: Vector storage in AWS RDS PostgreSQL pgvector uses dimension 384 with HNSW cosine indexing.

---

## Module 3.3 — PostgreSQL + pgvector Vector Storage Foundation

> **Scope**: This module establishes the production database layer. It does **NOT** implement the RAG answer engine, LLM calls, LangGraph agents, voice, or frontend. RAG retrieval/answer generation will be implemented in **Module 4**.

### Architecture

```
FastAPI Backend
      │
      ▼
SQLAlchemy 2.0 Engine (psycopg3 sync driver)
      │
      ▼
AWS RDS PostgreSQL 18.x
      │
      ├── pgvector extension
      │
      ├── tenants           ← SaaS multi-tenant root
      ├── documents         ← Policy documents scoped to tenant
      ├── chunks            ← Text segments with page/section metadata
      └── chunk_embeddings  ← vector(1536) embeddings per chunk
```

### AWS RDS Setup Requirements

1. Launch an AWS RDS **PostgreSQL 15+** instance (pgvector requires PG 11+).
2. Enable the `pgvector` extension (run once, or let the migration handle it):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Ensure the RDS security group allows inbound TCP on port **5432** from your application server.
4. Use SSL (`sslmode=require`) for all connections.

### Environment Variables

| Variable | Description | Example |
|---|---|---|
| `DATABASE_URL` | Full psycopg connection string | `postgresql+psycopg://user:pass@host:5432/db` |
| `DB_SSL_MODE` | SSL mode for RDS | `require` |
| `DB_POOL_SIZE` | Connection pool size | `10` |
| `DB_MAX_OVERFLOW` | Extra connections above pool | `20` |
| `VECTOR_DIMENSION` | Must match embedding model | `1536` |
| `DEV_TENANT_ID` | UUID for dev/seed tenant | `00000000-0000-0000-0000-000000000001` |
| `DEV_TENANT_SLUG` | Slug for dev tenant | `dev-org` |

Copy `backend/.env.example` → `backend/.env` and fill in real values. Never commit `.env`.

### Database Schema

```
tenants
  id UUID PK
  name VARCHAR(255)
  slug VARCHAR(100) UNIQUE INDEX
  created_at, updated_at TIMESTAMPTZ

documents  (FK → tenants)
  id UUID PK
  tenant_id UUID FK(tenants.id CASCADE)
  document_id VARCHAR(200)     ← slug e.g. travel_policy
  document_name VARCHAR(255)
  source_file VARCHAR(255)
  created_at, updated_at TIMESTAMPTZ
  UNIQUE(tenant_id, document_id)

chunks  (FK → tenants, documents)
  id UUID PK
  tenant_id UUID FK(tenants.id CASCADE)
  document_ref_id UUID FK(documents.id CASCADE)
  document_id VARCHAR(200)
  chunk_id VARCHAR(255)        ← e.g. travel_policy_0001
  chunk_index INTEGER
  text TEXT
  page_start INTEGER
  page_end INTEGER
  section VARCHAR(255)
  metadata_json JSON
  created_at TIMESTAMPTZ
  UNIQUE(tenant_id, chunk_id)

chunk_embeddings  (FK → tenants, chunks)
  id UUID PK
  tenant_id UUID FK(tenants.id CASCADE)
  chunk_ref_id UUID FK(chunks.id CASCADE)
  chunk_id VARCHAR(255)
  embedding vector(1536)       ← pgvector column
  embedding_model VARCHAR(100)
  embedding_dimension INTEGER
  is_mock BOOLEAN              ← true = test vectors, false = production
  created_at, updated_at TIMESTAMPTZ
  UNIQUE(tenant_id, chunk_id)
  INDEX: HNSW (embedding vector_cosine_ops, m=16, ef_construction=64)
```

### Multi-Tenant Isolation

Every record is scoped by `tenant_id`. The similarity search enforces:
```sql
WHERE chunk_embeddings.tenant_id = :tenant_id
ORDER BY embedding <=> :query_vector
LIMIT :top_k
```
Company A can **never** retrieve Company B's chunks. See the mandatory cross-tenant security test: `tests/test_vector_store.py::TestCrossTenantSecurityIsolation`.

### pgvector Index Strategy

- **Index type**: HNSW (Hierarchical Navigable Small World)
- **Distance metric**: Cosine (`vector_cosine_ops`)
- **Parameters**: `m=16, ef_construction=64` — balanced for datasets up to ~10k vectors
- **Why HNSW**: Logarithmic search time, better recall than IVFFlat for small-to-medium datasets
- **Scaling**: Switch to IVFFlat with `lists=100` when dataset exceeds ~100k vectors

### Migration Commands

```powershell
# Apply all pending migrations to RDS
py -m alembic -c backend/alembic.ini upgrade head

# View current migration status
py -m alembic -c backend/alembic.ini current

# Generate a new migration after model changes
py -m alembic -c backend/alembic.ini revision --autogenerate -m "description"

# Downgrade (rollback)
py -m alembic -c backend/alembic.ini downgrade -1
```

### Database Health Check

```powershell
py backend/app/database.py
```

Expected output (no credentials exposed):
```
=======================================================
Database Health & Connectivity Check
=======================================================
Target URL:         postgresql+psycopg://postgres:****@<host>:5432/postgres
Connection Status:  PASS
PostgreSQL Version: PostgreSQL 18.x on ...
pgvector Extension: DETECTED
Tables Detected:    alembic_version, chunk_embeddings, chunks, documents, tenants
=======================================================
```

### Embedding Import

> **⚠ Important**: The current 886 embeddings were generated with `--mock` and are deterministic test vectors. They must NOT be treated as production semantic embeddings.

```powershell
# Development import (mock vectors — requires explicit flag)
py backend/app/rag/vector_store.py --allow-mock

# Production import (real OpenAI embeddings, no flag needed)
py backend/app/rag/vector_store.py

# Import from a specific file
py backend/app/rag/vector_store.py --import-file path/to/embeddings.jsonl --allow-mock

# Import under a specific tenant
py backend/app/rag/vector_store.py --allow-mock --tenant-slug my-company --tenant-name "My Company"
```

### Production Embedding Sequence (Step-by-Step)

1. Set `OPENAI_API_KEY` in `backend/.env`
2. Set `EMBEDDING_MODEL=text-embedding-3-small` (or your chosen model)
3. Generate real embeddings:
   ```powershell
   py backend/app/rag/embeddings.py
   ```
4. Verify dimension in `data/processed/embeddings/embedding_validation_report.json`
5. Connect to RDS (health check must show PASS)
6. Import without `--allow-mock`:
   ```powershell
   py backend/app/rag/vector_store.py
   ```
7. Verify expected counts in RDS (1 tenant, 12 documents, 886 chunks, 886 embeddings)

### Tests

```powershell
# All tests (unit + integration)
py -m pytest tests/ -v

# Vector store tests only
py -m pytest tests/test_vector_store.py -v

# Database config/health tests
py -m pytest tests/test_database.py -v
```

| Test Class | Coverage |
|---|---|
| `TestSchemaDefinitions` | Model table names, fields, constraints |
| `TestVectorStoreDimensionValidation` | Dimension mismatch rejection, non-numeric rejection |
| `TestMockEmbeddingRejection` | Refuses mock import without `--allow-mock` |
| `TestCrossTenantSecurityIsolation` | **Mandatory**: Tenant A never sees Tenant B |
| `TestIdempotencyAndUpsert` | Re-import skips duplicates |
| `TestDatabaseConfig` | Credential masking, URL sanitization |
| `TestDatabaseHealth` | Unreachable DB returns clean FAIL, no secret leakage |

### Troubleshooting

| Problem | Solution |
|---|---|
| `pgvector Extension: NOT DETECTED` | Run `CREATE EXTENSION IF NOT EXISTS vector;` as a superuser on RDS, or let `enable_pgvector_extension()` do it |
| `Connection Status: FAIL` | Check RDS security group allows port 5432 from your IP; check `DATABASE_URL` in `.env` |
| `Refusing to import MOCK embeddings` | Add `--allow-mock` flag or generate real embeddings first |
| `vector dimension mismatch` | Ensure `VECTOR_DIMENSION` in `.env` matches your embedding model output |
| `alembic.util.CommandError: Can't locate revision` | Run `alembic upgrade head` from project root |


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

---

## Module 5 — Agent Orchestrator (LangGraph)

> **Scope**: Implements the enterprise AI agent orchestrator using LangGraph. Routes user requests to RAG policy Q&A, clarification questions, safe tool intent stubs, or polite declines. Does **NOT** execute real external tool APIs yet (HR/IT/Calendar execution will be implemented in Module 6+).

### Architecture

```
                 User Request
                      │
                      ▼
             AgentOrchestrator
             (LangGraph StateGraph)
                      │
                      ▼
               classify_intent
         (LLM / Heuristic Fallback)
                      │
        ┌─────────────┼───────────────┬────────────────┐
        ▼             ▼               ▼                ▼
 knowledge_query  leave_request  clarify_needed   out_of_scope
 (or policy Q&A) (or tool intent) (vague request) (non-work query)
        │             │               │                │
        ▼             ▼               ▼                ▼
 KnowledgeAgent  ToolRouterNode   ClarifyNode     DeclineNode
  (RAGService)   (Safe Stubs)    (Follow-up Q)   (Safe Refusal)
        │             │               │                │
        └─────────────┴───────┬───────┴────────────────┘
                              ▼
                         AgentResponse
```

### Components

| Component | Path | Description |
|-----------|------|-------------|
| **AgentState** | `backend/app/agents/agent_state.py` | `TypedDict` passed between nodes + `IntentType` enum |
| **IntentClassifier** | `backend/app/agents/intent_classifier.py` | LLM-based classification with keyword heuristic fallback & test provider injection |
| **KnowledgeAgentNode** | `backend/app/agents/knowledge_agent.py` | Graph node executing RAG answer generation via `RAGService` |
| **ToolRouterNode** | `backend/app/agents/tool_router.py` | Dispatches tool intents to safe pending stubs |
| **HRAgentNode** | `backend/app/agents/hr_agent.py` | Safe stub for leave requests (`hr_leave_system`) |
| **ClarifyNode** | `backend/app/agents/clarify_node.py` | Generates context-aware clarification questions |
| **DeclineNode** | `backend/app/agents/decline_node.py` | Polite safe refusal for non-work queries |
| **AgentOrchestrator** | `backend/app/agents/orchestrator.py` | Compiles and executes the LangGraph `StateGraph` |

### API Endpoint

**`POST /api/agent`** — Unified Agent Orchestrator Endpoint.

Request:
```json
{
  "request": "How do I apply for annual leave?",
  "tenant_id": "00000000-0000-0000-0000-000000000001"
}
```

Response:
```json
{
  "response": "I can help with your leave request. In Module 6, I will connect to the HR system...",
  "agent_mode": "tool_intent",
  "intent": "leave_request",
  "intent_confidence": 0.95,
  "requires_clarification": false,
  "clarification_question": null,
  "rag_sources": [],
  "tool_intents": [
    {
      "tool": "hr_leave_system",
      "intent": "leave_request",
      "status": "stub_pending",
      "message": "...",
      "requires_action": true,
      "action_description": "Submit leave request",
      "module_planned": "Module 6 — HR Integration"
    }
  ],
  "error": null
}
```

### Testing Instructions

```powershell
# Run Module 5 agent orchestrator tests only (30 tests)
py -m pytest tests/test_agents.py -v

# Run full project test suite (190 tests across all modules)
py -m pytest tests/ -v
```

---

## AI Provider Architecture (Puter AI & OpenAI)

## AI Provider Architecture (Groq, Puter AI, OpenAI)

The application features a clean, decoupled AI provider abstraction layer under `backend/app/ai/`, allowing the backend to route LLM queries dynamically across **Groq** (Primary), **Puter AI**, or **OpenAI** without modifying business or agent logic.

### Layer Architecture

```
       RAG AnswerGenerator             IntentClassifier
        (Knowledge Q&A)              (LangGraph Routing)
               │                              │
               └──────────────┬───────────────┘
                              ▼
                         AIProvider
                    (Common Abstraction)
                              │
               ┌──────────────┼──────────────┬──────────────┐
               ▼              ▼              ▼              ▼
         GroqAIProvider PuterAIProvider OpenAIProvider MockAIProvider
       (Primary/Fast) (OpenAI-compat) (Direct API)   (Offline Tests)
```

### Supported Providers

1. **Groq AI (`GroqAIProvider`) — PRIMARY**:
   - High-speed, low-latency LLM inference using Groq's official OpenAI-compatible REST API (`https://api.groq.com/openai/v1`).
   - Default model: `openai/gpt-oss-120b` (also supports `openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, etc.).
   - Standard JSON object completion mode for structured intent classification.
   - Built-in exponential backoff for rate limits (`429`) and server errors (`5xx`).

2. **Puter AI (`PuterAIProvider`)**:
   - Uses Puter's OpenAI-compatible gateway (`https://api.puter.com/puterai/openai/v1/`).
   - Authenticated with personal Puter Auth Token.
   - Retained for backward compatibility. Note: Free Puter personal accounts may encounter `subscription_required` (HTTP 402) on certain endpoints depending on platform quota policies.

3. **OpenAI (`OpenAIProvider`)**:
   - Direct integration with OpenAI's Chat Completions API (`gpt-4o-mini`, `gpt-4o`).
   - Retained as a production option and fallback.

4. **Mock Provider (`MockAIProvider`)**:
   - In-memory deterministic responses for unit and integration testing.
   - Ensures all test suites run offline with zero token costs and zero external network calls.

### Embeddings Note

> [!IMPORTANT]
> **Embedding Provider Separated**: Embeddings are deliberately decoupled from the chat LLM provider. The 886 knowledge chunks currently in PostgreSQL pgvector continue to use their calibrated vector store setup (`vector(1536)`). Chat completions and RAG answer synthesis use the configured `AI_PROVIDER` (Groq).

### Groq AI Setup Instructions (Current Configuration)

1. Obtain a Groq API Key from the [Groq Console](https://console.groq.com/).
2. In `backend/.env`, configure:
   ```bash
   AI_PROVIDER=groq
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   GROQ_BASE_URL=https://api.groq.com/openai/v1
   ```
3. To test your connection to Groq AI live, run the standalone smoke test:
   ```powershell
   py backend/scripts/test_groq_ai.py
   ```

### Puter AI Setup Instructions (Optional)

1. If using Puter AI, obtain a Puter Auth Token from [Puter.com](https://puter.com).
2. In `backend/.env`, set:
   ```bash
   AI_PROVIDER=puter
   PUTER_AUTH_TOKEN=your_actual_puter_auth_token_here
   PUTER_MODEL=gpt-4o-mini
   PUTER_BASE_URL=https://api.puter.com/puterai/openai/v1/
   ```
3. To test connection to Puter AI:
   ```powershell
   py backend/scripts/test_puter_ai.py
   ```

---

## Voice Processing Subsystem

### Module 6.1 — Speech-to-Text (Groq Whisper Large V3 Turbo)

- **Purpose**: Converts voice audio recordings into accurate text transcripts using Groq's high-speed Whisper Large V3 Turbo API (`whisper-large-v3-turbo`).
- **Current Scope**: Implements pure **Audio File $\rightarrow$ Groq STT $\rightarrow$ Transcript**. Voice-to-Agent and Voice-to-RAG routing will be integrated in subsequent modules.
- **Model**: `whisper-large-v3-turbo` (sub-second latency, high multilingual accuracy).
- **Supported Audio Formats**: `wav`, `mp3`, `m4a`, `ogg`, `webm`, `mp4`, `flac`, `mpeg`, `mpga`.
- **Maximum File Size**: 25 MB (configurable via `STT_MAX_UPLOAD_SIZE_MB`).

#### API Endpoint

`POST /api/voice/transcribe`

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `audio` (required): Binary audio file upload.
  - `language` (optional): ISO-639-1 language code (e.g., `en`, `es`, `fr`). Defaults to auto-detection.
  - `prompt` (optional): Context prompt to guide transcription of domain-specific terminology or acronyms.

#### Request Example (PowerShell / curl)

```powershell
curl.exe -X POST "http://localhost:8000/api/voice/transcribe" `
  -F "audio=@data/processed/sample_test.wav;type=audio/wav" `
  -F "language=en" `
  -F "prompt=Company policy questions"
```

#### Response Example

```json
{
  "success": true,
  "text": "What is the annual leave and sick leave policy?",
  "language": "en",
  "duration": 3.45,
  "segments": []
}
```

#### Error Status Codes

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Missing audio file or empty (0 bytes) upload |
| `413 Payload Too Large` | Audio file exceeds `STT_MAX_UPLOAD_SIZE_MB` (25 MB) |
| `415 Unsupported Media Type` | Unsupported file format/extension |
| `502 Bad Gateway` | Groq STT upstream API error or rate limit |
| `500 Internal Server Error` | Unconfigured API key or server-side failure |

#### Configuration Parameters (`backend/.env`)

```env
# Speech-to-Text Settings
STT_PROVIDER=groq
STT_MODEL=whisper-large-v3-turbo
STT_LANGUAGE=en
STT_MAX_UPLOAD_SIZE_MB=25
STT_TIMEOUT_SECONDS=60
```

> **Note**: Uses the existing `GROQ_API_KEY` configuration. No new API keys are required.

#### Manual Smoke Test

Run the live Groq STT test with an audio file or generate a synthetic test audio file:

```powershell
# With your own audio file
py backend/scripts/test_speech_to_text.py path/to/recording.wav

# Or generate a synthetic test audio file automatically
py backend/scripts/test_speech_to_text.py --generate-sample
```

#### Security & Privacy Notes

- **Zero Audio Persistence**: Audio files are streamed to temporary files during processing and **guaranteed to be deleted immediately** after transcription in a `finally` block, even on provider failure.
- **No Secret Leakage**: API keys and authorization headers are never logged, exposed in error responses, or printed in representations.
- **Strict Validation**: Executable files and unsupported formats are rejected immediately prior to reading content.

---

### Module 6.2 — STT → LangGraph Agent Orchestrator

- **Purpose**: Connects the voice Speech-to-Text subsystem (Module 6.1) to the LangGraph Agent Orchestrator (Module 5), providing an end-to-end voice-to-reasoning pipeline.
- **End-to-End Pipeline Flow**:
  ```
  User Audio File (.wav, .mp3, etc.)
        │
        ▼
  POST /api/voice/agent (multipart/form-data)
        │
        ▼
  Groq Whisper Large V3 Turbo STT
        │
        ▼
  Transcribed Text (e.g. "What is the annual leave policy for employees?")
        │
        ├── [Empty / Whitespace] ──> Short-circuit (status="empty_transcript", 0 LLM tokens spent)
        │
        ▼
  LangGraph Agent Orchestrator
        │
        ▼
  Intent Classification (Groq LLM / Prompt-Engineered State Graph)
        ├── knowledge_query ──> RAG Pipeline (AWS RDS pgvector 384-dim BGE) ──> Groq LLM Answer + Sources
        ├── general_chat    ──> Direct Conversational LLM Response
        └── action_request  ──> Structured Tool Routing (HR/IT/Calendar/Timesheet/Navigation)
        │
        ▼
  VoiceAgentResponse JSON
  ```

#### API Endpoint

`POST /api/voice/agent`

- **Content-Type**: `multipart/form-data`
- **Form Parameters**:
  - `audio` (required): Binary audio file upload (`.wav`, `.mp3`, `.m4a`, `.ogg`, `.webm`, `.flac`, etc.).
  - `tenant_id` (optional): Multi-tenant organization UUID. Defaults to `config.tenant.default_id` (`00000000-0000-0000-0000-000000000001`).
  - `language` (optional): ISO-639-1 language code (e.g., `en`, `es`, `fr`). Defaults to auto-detection or `STT_LANGUAGE`.
  - `prompt` (optional): Context prompt to guide Whisper transcription on domain terminology.
  - `conversation_id` (optional): Correlation ID for session tracing.

#### Request Example (PowerShell / curl)

```powershell
curl.exe -X POST "http://localhost:8000/api/voice/agent" `
  -F "audio=@voice.mp3;type=audio/mpeg" `
  -F "tenant_id=00000000-0000-0000-0000-000000000001" `
  -F "language=en"
```

#### Response Example (`VoiceAgentResponse`)

```json
{
  "success": true,
  "transcript": "What is the annual leave policy for employees?",
  "response": "Under the Leave and Holiday Policy, full-time employees are entitled to 18 days of annual earned leave per calendar year, accrued on a pro-rata basis each month...",
  "intent": "knowledge_query",
  "intent_confidence": 0.99,
  "agent_mode": "rag",
  "status": "completed",
  "requires_clarification": false,
  "clarification_question": null,
  "rag_sources": [
    {
      "document_name": "HCL Leave and Holiday Policy",
      "source_file": "data/raw/policies/HCL Leave and Holiday Policy.pdf",
      "page_start": 3,
      "page_end": 4,
      "similarity_score": 0.784
    }
  ],
  "tool_intents": [],
  "duration": 2.15,
  "error": null
}
```

#### Short-Circuit on Empty Audio / Silence

If the audio contains no recognizable speech and Whisper returns an empty or whitespace transcript:
- The orchestrator **short-circuits immediately** without invoking the LangGraph agent or RAG retrieval.
- Saves LLM tokens and eliminates downstream API latency.
- Returns `status="empty_transcript"` with a polite fallback message.

#### Error Handling & HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Missing file, empty audio (0 bytes), or invalid tenant UUID format |
| `413 Payload Too Large` | Audio exceeds `STT_MAX_UPLOAD_SIZE_MB` (25 MB) |
| `415 Unsupported Media Type` | File extension not supported |
| `502 Bad Gateway` | Groq Whisper upstream API error or timeout |
| `500 Internal Server Error` | Unhandled server exception during orchestration |

#### Automated Testing

```powershell
# Run Module 6.2 specific unit tests
py -m pytest tests/test_voice_agent.py -v

# Run full project test suite (260 tests)
py -m pytest tests/ -v
```

#### Manual Verification Script

Test end-to-end voice-to-agent processing using a real audio file:

```powershell
py backend/scripts/test_voice_agent.py path/to/voice.mp3
```

#### Security & Tenant Isolation

- **Tenant Isolation**: Vector search and document lookups are filtered by validated `tenant_id` at the database level (`WHERE chunk_embeddings.tenant_id = :tenant_id`). Cross-tenant leakage is strictly prevented.
- **Temporary File Security**: Uploaded audio is stored in an ephemeral temp file and cleaned up unconditionally in a `finally` block.
- **Credential Protection**: Zero API keys, database connection strings, or internal tokens are exposed in API payloads, logs, or error responses.

---

### Module 6.3 — Text-to-Speech (TTS)

- **Purpose**: Converts textual agent answers into natural speech audio (`audio/wav`) ready for downstream browser playback.
- **Current Scope**: Implements the backend TTS subsystem: **Text Response $\rightarrow$ TTS Service $\rightarrow$ TTS Provider $\rightarrow$ Audio Bytes**.
- **Model**: `canopylabs/orpheus-v1-english` (Groq official TTS model).
- **Supported Voices**: `autumn`, `diana`, `hannah`, `austin`, `troy` (default: `autumn`).
- **Audio Output Format**: `wav` (16-bit PCM, browser-compatible).
- **Maximum Text Length**: 5,000 characters (configurable via `TTS_MAX_TEXT_LENGTH`).

#### Pipeline Architecture

```
Text Response
      │
      ▼
POST /api/voice/synthesize (application/json)
      │
      ├─► Validation (non-empty, non-whitespace, len <= 5000, format check)
      │
      ▼
TTS Service
      │
      ▼
TTS Provider Abstraction (Groq Orpheus / Mock)
      │
      ▼
Binary Audio Response (audio/wav, 200 OK)
```

#### API Endpoint

`POST /api/voice/synthesize`

- **Content-Type**: `application/json`
- **Request Body**:
  - `text` (required, string): Text to synthesize into speech (1–5000 characters).
  - `voice` (optional, string): Voice persona (e.g. `autumn`, `diana`, `hannah`, `austin`, `troy`).
  - `model` (optional, string): Model identifier (default: `canopylabs/orpheus-v1-english`).
  - `format` (optional, string): Target audio format (default: `wav`).
  - `speed` (optional, float): Speech rate multiplier (e.g. `1.0`).

#### Request Example (PowerShell / curl)

```powershell
curl.exe -X POST "http://localhost:8000/api/voice/synthesize" `
  -H "Content-Type: application/json" `
  -d '{"text": "According to company policy, employees receive 18 days of annual leave.", "voice": "autumn"}' `
  --output response.wav
```

#### Response Behavior

- **HTTP 200 OK**:
  - `Content-Type`: `audio/wav`
  - `Content-Disposition`: `inline; filename="synthesized.wav"`
  - `X-TTS-Provider`: `groq` (or `mock`)
  - `X-TTS-Model`: `canopylabs/orpheus-v1-english`
  - `X-TTS-Voice`: `autumn`
  - Body: Raw binary WAV audio bytes.

#### Error Handling & HTTP Status Codes

| Status Code | Description |
|-------------|-------------|
| `400 Bad Request` | Empty or whitespace-only text |
| `413 Payload Too Large` | Text exceeds `TTS_MAX_TEXT_LENGTH` (5000 chars) |
| `422 Unprocessable Entity` | Missing text field or unsupported audio format |
| `502 Bad Gateway` | Upstream Groq TTS provider error / terms acceptance required |
| `500 Internal Server Error` | Missing API key or internal server error |

#### Configuration Parameters (`backend/.env`)

```env
# Text-to-Speech Settings
TTS_PROVIDER=groq
TTS_MODEL=canopylabs/orpheus-v1-english
TTS_VOICE=autumn
TTS_FORMAT=wav
TTS_SPEED=1.0
TTS_MAX_TEXT_LENGTH=5000
TTS_TIMEOUT_SECONDS=60
```

#### Automated Testing

```powershell
# Run Module 6.3 dedicated unit tests (33 tests)
py -m pytest tests/test_text_to_speech.py -v

# Run full project test suite (293 tests across all modules)
py -m pytest tests/ -v
```

#### Manual Smoke Test Script

```powershell
# Real Groq TTS smoke test
py backend/scripts/test_text_to_speech.py "Hello, welcome to the employee assistant."

# Offline mock test (writes sample WAV to data/processed/tts/)
py backend/scripts/test_text_to_speech.py --mock "Testing offline speech synthesis."
```

> [!NOTE]
> **Groq Account Terms Acceptance**:
> Groq's official TTS model `canopylabs/orpheus-v1-english` requires one-time terms of service acceptance in the Groq console. If not yet accepted, the live test or endpoint reports the direct acceptance URL: `https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english`.

---

## Module 6.4 — React Voice UI

### Overview
Module 6.4 implements the voice interaction user interface for the Voice-First Agentic AI Employee Workplace SaaS. Built with React 18, Vite, and modern enterprise Vanilla CSS tokens, it connects the browser's native audio recording directly to the voice pipeline (`/api/voice/agent` and `/api/voice/synthesize`).

### Target Flow
```
User clicks Microphone
        │
        ▼
navigator.mediaDevices.getUserMedia (audio)
        │
        ▼
MediaRecorder captures chunks with dynamic MIME negotiation
        │
        ▼
Recording stopped → Audio Blob created
        │
        ▼
POST /api/voice/agent (Multipart form data: audio file)
        │
        ├─► STT (Whisper) -> Transcript
        ├─► LangGraph Agent Orchestrator -> Intent classification & RAG
        └─► Formatted text answer + clean citations
        │
        ▼
POST /api/voice/synthesize (Text-to-Speech)
        │
        ▼
Browser Audio playback of spoken response
```

### Key Components

- **`VoiceAssistant.jsx`**: Core enterprise voice assistant component:
  - Multi-state machine: `idle` → `recording` → `processing` → `speaking` → `completed` / `error`.
  - Dynamic audio MIME format detection (`audio/webm;codecs=opus`, `audio/webm`, `audio/mp4`, `audio/ogg`, `audio/wav`).
  - Active recording timer and animated visualizer pulse waves.
  - Automatic resource management: Microphone tracks are explicitly stopped (`track.stop()`) on recording completion and unmount. Audio blob object URLs are cleanly revoked (`URL.revokeObjectURL`).
  - Graceful TTS fallback: If TTS audio synthesis is unavailable or requires Groq terms acceptance, the text response and citations remain fully accessible, accompanied by an informative banner.
  - Sanitized citations: Raw server file paths (e.g. `data/raw/...`) are converted into human-readable document titles and page ranges.
- **`Assistant.jsx`**: Dedicated AI Workplace Assistant page featuring conversation history, active voice interaction, and text input fallback.
- **`Dashboard.jsx`**: Workplace overview dashboard with quick status tiles and instant voice query launcher.
- **`App.jsx`**: Responsive shell navigation with route switching between Dashboard and Assistant.
- **`api.js`**: Centralized API service with timeout guards and error sanitization for `/api/voice/agent`, `/api/voice/synthesize`, and `/api/agent`.

### Frontend Configuration & Proxy

- **`vite.config.js`**:
  ```javascript
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
  ```

### Running the Application

1. **Start Backend Server**:
   ```powershell
   py -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
   ```

2. **Start Frontend Dev Server**:
   ```powershell
   cd frontend
   npm run dev
   ```
   Open `http://localhost:5173` in your browser.

3. **Production Build**:
   ```powershell
   cd frontend
   npm run build
   ```


