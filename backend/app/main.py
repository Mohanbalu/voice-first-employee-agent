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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.app.routes.chat import router as chat_router
    from backend.app.routes.agent import router as agent_router
    from backend.app.routes.voice import router as voice_router
    from backend.app.routes.auth import router as auth_router
    from backend.app.routes.hr import router as hr_router
    from backend.app.routes.tickets import router as tickets_router
    from backend.app.routes.location import router as location_router
except ImportError:
    from app.routes.chat import router as chat_router
    from app.routes.agent import router as agent_router
    from app.routes.voice import router as voice_router
    from app.routes.auth import router as auth_router
    from app.routes.hr import router as hr_router
    from app.routes.tickets import router as tickets_router
    from app.routes.location import router as location_router

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
]

env_cors = os.getenv("CORS_ORIGINS", "")
if env_cors:
    for origin in env_cors.split(","):
        clean_origin = origin.strip()
        if clean_origin and clean_origin not in ALLOWED_ORIGINS:
            ALLOWED_ORIGINS.append(clean_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
    allow_origin_regex=r"^https?://.*$" if "*" in ALLOWED_ORIGINS else r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(voice_router)
app.include_router(auth_router)
app.include_router(hr_router)
app.include_router(tickets_router)
app.include_router(location_router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Service health probe."""
    return {"status": "ok", "module": "6.1 — Speech-to-Text"}
