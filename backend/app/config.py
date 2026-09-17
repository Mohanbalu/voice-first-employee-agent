"""Application Configuration Module.

Loads and validates environment configurations for the Voice-First Agentic AI
Employee Assistant, including PostgreSQL database connection parameters, pgvector
dimensions, and multi-tenant settings.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv

    # Load from backend/.env if available
    backend_env = Path(__file__).resolve().parent.parent.parent / "backend" / ".env"
    if backend_env.exists():
        load_dotenv(backend_env)
    else:
        load_dotenv()
except ImportError:
    pass


def get_project_root() -> Path:
    """Locates the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "backend").exists():
            return parent
    return Path.cwd()


def sanitize_db_url(url: Optional[str]) -> str:
    """Masks credentials in a database URL for safe logging and reporting."""
    if not url:
        return "None"
    # Matches ://user:password@host
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)


@dataclass
class DatabaseSettings:
    """PostgreSQL and pgvector connection settings."""

    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 1800
    ssl_mode: Optional[str] = None
    vector_dimension: int = 384


@dataclass
class EmbeddingSettings:
    """Local and API embedding provider settings."""

    provider: str = "local"  # "local" | "mock" | "openai"
    model: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 64
    query_instruction: str = "Represent this sentence for searching relevant passages: "


@dataclass
class TenantSettings:
    """Default development tenant settings for initial seed data."""

    default_id: str = "00000000-0000-0000-0000-000000000001"
    default_name: str = "Development Organization"
    default_slug: str = "dev-org"


@dataclass
class AISettings:
    """AI and LLM provider configuration (Groq, Puter AI, OpenAI)."""

    provider: str = "groq"  # "groq" | "puter" | "openai"
    groq_api_key: Optional[str] = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    puter_auth_token: Optional[str] = None
    puter_base_url: str = "https://api.puter.com/puterai/openai/v1/"
    puter_model: str = "gpt-4o-mini"
    openai_api_key: Optional[str] = None
    openai_chat_model: str = "gpt-4o-mini"
    timeout_seconds: float = 30.0
    max_retries: int = 2


@dataclass
class STTSettings:
    """Speech-to-Text configuration (Groq Whisper)."""

    provider: str = "groq"
    model: str = "whisper-large-v3-turbo"
    language: Optional[str] = None
    temperature: float = 0.0
    max_upload_size_mb: int = 25
    timeout_seconds: float = 60.0


@dataclass
class TTSSettings:
    """Text-to-Speech configuration (Groq Orpheus / Mock)."""

    provider: str = "groq"
    model: str = "canopylabs/orpheus-v1-english"
    voice: str = "autumn"
    format: str = "wav"
    speed: float = 1.0
    max_text_length: int = 5000
    timeout_seconds: float = 60.0


@dataclass
class AppConfig:
    """Consolidated application configuration."""

    project_name: str
    environment: str
    db: DatabaseSettings
    tenant: TenantSettings
    ai: AISettings
    embedding: EmbeddingSettings
    stt: STTSettings
    tts: TTSSettings

    @classmethod
    def load(cls) -> "AppConfig":
        """Instantiates AppConfig from environment variables."""
        # Database URL with fallback for development
        raw_db_url = os.getenv("DATABASE_URL")
        if not raw_db_url or "postgres:postgres@localhost" in raw_db_url:
            # Check individual POSTGRES_* params if set
            pg_user = os.getenv("POSTGRES_USER", "postgres")
            pg_pass = os.getenv("POSTGRES_PASSWORD", "postgres")
            pg_server = os.getenv("POSTGRES_SERVER", "localhost")
            pg_port = os.getenv("POSTGRES_PORT", "5432")
            pg_db = os.getenv("POSTGRES_DB", "workplace_assistant")
            db_url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_server}:{pg_port}/{pg_db}"
        else:
            db_url = raw_db_url

        # Normalise to psycopg2 driver (psycopg2-binary is installed on all envs)
        if db_url.startswith("postgresql+psycopg://"):
            # Downgrade psycopg v3 prefix → psycopg2
            db_url = db_url.replace("postgresql+psycopg://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql+asyncpg://"):
            db_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)
        elif db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)

        try:
            pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        except ValueError:
            pool_size = 10

        try:
            max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        except ValueError:
            max_overflow = 20

        try:
            vector_dim = int(os.getenv("VECTOR_DIMENSION", "384"))
        except ValueError:
            vector_dim = 384

        ssl_mode = os.getenv("DB_SSL_MODE")

        db_settings = DatabaseSettings(
            url=db_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            ssl_mode=ssl_mode,
            vector_dimension=vector_dim,
        )

        tenant_settings = TenantSettings(
            default_id=os.getenv("DEV_TENANT_ID", "00000000-0000-0000-0000-000000000001"),
            default_name=os.getenv("DEV_TENANT_NAME", "Development Organization"),
            default_slug=os.getenv("DEV_TENANT_SLUG", "dev-org"),
        )

        try:
            llm_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError:
            llm_timeout = 30.0

        try:
            llm_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        except ValueError:
            llm_retries = 2

        ai_settings = AISettings(
            provider=(os.getenv("AI_PROVIDER") or "groq").strip().lower(),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            puter_auth_token=os.getenv("PUTER_AUTH_TOKEN"),
            puter_base_url=os.getenv("PUTER_BASE_URL", "https://api.puter.com/puterai/openai/v1/"),
            puter_model=os.getenv("PUTER_MODEL", "gpt-4o-mini"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini",
            timeout_seconds=llm_timeout,
            max_retries=llm_retries,
        )

        try:
            emb_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
        except ValueError:
            emb_batch_size = 64

        try:
            emb_dimension = int(os.getenv("EMBEDDING_DIMENSION", str(vector_dim)))
        except ValueError:
            emb_dimension = vector_dim

        embedding_settings = EmbeddingSettings(
            provider=(os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower(),
            model=os.getenv("EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5",
            dimension=emb_dimension,
            batch_size=emb_batch_size,
            query_instruction=os.getenv(
                "EMBEDDING_QUERY_INSTRUCTION",
                "Represent this sentence for searching relevant passages: ",
            ),
        )

        try:
            stt_max_size = int(os.getenv("STT_MAX_UPLOAD_SIZE_MB", "25"))
        except ValueError:
            stt_max_size = 25

        try:
            stt_timeout = float(os.getenv("STT_TIMEOUT_SECONDS", "60"))
        except ValueError:
            stt_timeout = 60.0

        try:
            stt_temperature = float(os.getenv("STT_TEMPERATURE", "0.0"))
        except ValueError:
            stt_temperature = 0.0

        stt_settings = STTSettings(
            provider=(os.getenv("STT_PROVIDER") or "groq").strip().lower(),
            model=os.getenv("STT_MODEL") or "whisper-large-v3-turbo",
            language=os.getenv("STT_LANGUAGE"),
            temperature=stt_temperature,
            max_upload_size_mb=stt_max_size,
            timeout_seconds=stt_timeout,
        )

        try:
            tts_speed = float(os.getenv("TTS_SPEED", "1.0"))
        except ValueError:
            tts_speed = 1.0

        try:
            tts_max_len = int(os.getenv("TTS_MAX_TEXT_LENGTH", "5000"))
        except ValueError:
            tts_max_len = 5000

        try:
            tts_timeout = float(os.getenv("TTS_TIMEOUT_SECONDS", "60.0"))
        except ValueError:
            tts_timeout = 60.0

        tts_settings = TTSSettings(
            provider=(os.getenv("TTS_PROVIDER") or "groq").strip().lower(),
            model=os.getenv("TTS_MODEL") or "canopylabs/orpheus-v1-english",
            voice=os.getenv("TTS_VOICE") or "autumn",
            format=(os.getenv("TTS_FORMAT") or "wav").strip().lower(),
            speed=tts_speed,
            max_text_length=tts_max_len,
            timeout_seconds=tts_timeout,
        )

        return cls(
            project_name=os.getenv(
                "PROJECT_NAME", "Voice-First Agentic AI Employee Workplace Assistant"
            ),
            environment=os.getenv("ENVIRONMENT", "development"),
            db=db_settings,
            tenant=tenant_settings,
            ai=ai_settings,
            embedding=embedding_settings,
            stt=stt_settings,
            tts=tts_settings,
        )


# Global configuration instance
config = AppConfig.load()

