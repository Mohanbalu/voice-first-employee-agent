"""FastAPI Application — Voice-First Agentic AI Employee Workplace Assistant.

Module 4: RAG Answer Engine endpoint.
Module 5: Agent Orchestrator (LangGraph) endpoint.
Future modules will add authentication, voice, agents, etc.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure both backend directory and project root are in sys.path
_current_file = Path(__file__).resolve()
_app_dir = _current_file.parent
_backend_dir = _app_dir.parent
_project_root = _backend_dir.parent

for _p in [str(_backend_dir), str(_project_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
import uuid

try:
    from backend.app.config import config
    from backend.app.database import get_db
    from backend.app.models.tenant import Tenant
    from backend.app.rag.retriever import RAGRetriever, RetrievalConfig
    from backend.app.rag.context_builder import ContextBuilder
    from backend.app.rag.answer_generator import AnswerGenerator
    from backend.app.routes.chat import router as chat_router
    from backend.app.routes.agent import router as agent_router
    from backend.app.routes.voice import router as voice_router
    from backend.app.routes.auth import router as auth_router
    from backend.app.routes.hr import router as hr_router
    from backend.app.routes.tickets import router as tickets_router
    from backend.app.routes.location import router as location_router
    from backend.app.routes.schedules import router as schedules_router
except ImportError:
    from app.config import config
    from app.database import get_db
    from app.models.tenant import Tenant
    from app.rag.retriever import RAGRetriever, RetrievalConfig
    from app.rag.context_builder import ContextBuilder
    from app.rag.answer_generator import AnswerGenerator
    from app.routes.chat import router as chat_router
    from app.routes.agent import router as agent_router
    from app.routes.voice import router as voice_router
    from app.routes.auth import router as auth_router
    from app.routes.hr import router as hr_router
    from app.routes.tickets import router as tickets_router
    from app.routes.location import router as location_router
    from app.routes.schedules import router as schedules_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("app.main")

app = FastAPI(
    title="Voice-First Agentic AI Employee Workplace Assistant",
    description=(
        "Production-oriented SaaS employee onboarding assistant. "
        "Module 4: RAG-powered company policy Q&A. "
        "Module 5: LangGraph Agent Orchestrator. "
        "Module 6.1: Speech-to-Text via Groq Whisper Large V3 Turbo."
    ),
    version="0.6.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — Development & production origins
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://frontend-lac-alpha-31.vercel.app",
]

env_cors = os.getenv("CORS_ORIGINS", "")
if env_cors:
    for origin in env_cors.split(","):
        clean_origin = origin.strip()
        if clean_origin and clean_origin not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(clean_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://.*$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_headers_middleware(request: Request, call_next):
    """Guarantees Access-Control-Allow-Origin headers on all responses, errors, and OPTIONS."""
    origin = request.headers.get("origin")
    allow_origin = origin if origin else "*"

    if request.method == "OPTIONS":
        res = Response(status_code=204)
    else:
        try:
            res = await call_next(request)
        except Exception as exc:
            logger.exception("Global exception caught for %s: %s", request.url.path, exc)
            res = JSONResponse(
                status_code=500,
                content={"detail": f"Internal Server Error: {str(exc)}"},
            )

    res.headers["Access-Control-Allow-Origin"] = allow_origin
    res.headers["Access-Control-Allow-Credentials"] = "true"
    res.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD"
    res.headers["Access-Control-Allow-Headers"] = "*"
    return res


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin", "*")
    logger.exception("Global exception handler: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": origin if origin else "*",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD",
            "Access-Control-Allow-Headers": "*",
        },
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(auth_router)
app.include_router(hr_router)
app.include_router(tickets_router)
app.include_router(location_router)
app.include_router(schedules_router)


@app.get("/api/debug/rag", tags=["debug"])
def debug_rag(q: str = "What is the annual leave policy for employees?") -> dict:
    """Diagnostic endpoint to profile RAG retrieval and LLM answer generation."""
    import time
    from backend.app.database import get_session_factory
    timings = {}

    try:
        session_factory = get_session_factory()
        with session_factory() as db:
            t0 = time.monotonic()
            tenant = db.execute(
                select(Tenant).where(Tenant.id == uuid.UUID(config.tenant.default_id))
            ).scalar_one_or_none()
            timings["1_tenant_lookup_ms"] = round((time.monotonic() - t0) * 1000, 2)

            if tenant is None:
                first_tenant = db.execute(select(Tenant).limit(1)).scalar_one_or_none()
                tenant = first_tenant
                timings["tenant_fallback"] = str(tenant.id) if tenant else "none"

            t1 = time.monotonic()
            retriever = RAGRetriever(config=RetrievalConfig())
            results = retriever.retrieve(session=db, tenant_id=tenant.id, question=q) if tenant else []
            timings["2_retrieve_ms"] = round((time.monotonic() - t1) * 1000, 2)
            timings["retrieved_chunks"] = len(results)

            t2 = time.monotonic()
            builder = ContextBuilder()
            built = builder.build(results)
            timings["3_context_ms"] = round((time.monotonic() - t2) * 1000, 2)

            t3 = time.monotonic()
            gen = AnswerGenerator()
            ans = gen.generate(question=q, context_text=built.context_text)
            timings["4_answer_gen_ms"] = round((time.monotonic() - t3) * 1000, 2)
            timings["total_ms"] = round((time.monotonic() - t0) * 1000, 2)

            return {
                "status": "ok",
                "timings": timings,
                "env_groq_model": os.getenv("GROQ_MODEL"),
                "config_groq_model": config.ai.groq_model,
                "embedding_provider": config.embedding.provider,
                "is_render": bool(os.getenv("RENDER") or os.getenv("RENDER_SERVICE_ID")),
                "answer_preview": ans.answer[:300],
            }
    except Exception as exc:
        import traceback
        return {
            "status": "error",
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
            "timings": timings,
        }


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Service health probe."""
    return {"status": "ok", "module": "6.1 — Speech-to-Text"}
