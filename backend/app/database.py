"""Database Connection & Health Module.

Provides SQLAlchemy 2.0 engine, session factory, connection pooling,
FastAPI dependency injection, and health verification for PostgreSQL with pgvector.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional

# Ensure project root is in sys.path for direct script execution
_current_file = Path(__file__).resolve()
for _parent in [_current_file] + list(_current_file.parents):
    if (_parent / "backend").exists() and (_parent / "data").exists():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

try:
    from backend.app.config import config, sanitize_db_url
except ImportError:
    from app.config import config, sanitize_db_url

logger = logging.getLogger("knowledge_base.database")


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy models."""

    pass


# Global engine and sessionmaker (lazily initialized)
_engine: Optional[Engine] = None
_SessionFactory: Optional[sessionmaker[Session]] = None


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Returns or lazily initializes the SQLAlchemy engine.

    Uses connection pooling with pre-ping validation to ensure stale
    connections are recycled safely in SaaS environments.
    """
    global _engine, _SessionFactory
    target_url = db_url or config.db.url

    # Check if existing engine can be reused
    if _engine is not None and str(_engine.url) == target_url:
        return _engine

    connect_args: Dict[str, Any] = {}
    if config.db.ssl_mode:
        connect_args["sslmode"] = config.db.ssl_mode

    # Configure engine with robust pooling
    _engine = create_engine(
        target_url,
        pool_size=config.db.pool_size,
        max_overflow=config.db.max_overflow,
        pool_timeout=config.db.pool_timeout,
        pool_recycle=config.db.pool_recycle,
        pool_pre_ping=True,
        connect_args=connect_args,
    )
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    return _engine


def get_session_factory(db_url: Optional[str] = None) -> sessionmaker[Session]:
    """Returns the configured SessionLocal factory."""
    global _SessionFactory
    if _SessionFactory is None or db_url is not None:
        get_engine(db_url)
    assert _SessionFactory is not None
    return _SessionFactory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions with automatic cleanup."""
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


def enable_pgvector_extension(engine: Optional[Engine] = None) -> bool:
    """Enables the pgvector extension on the target database."""
    target_engine = engine or get_engine()
    with target_engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    logger.info("Ensured 'vector' extension is enabled.")
    return True


def check_db_health(db_url: Optional[str] = None) -> Dict[str, Any]:
    """Inspects database connectivity, PostgreSQL version, pgvector, and tables.

    Never exposes sensitive credentials.
    """
    target_url = db_url or config.db.url
    sanitized = sanitize_db_url(target_url)

    result: Dict[str, Any] = {
        "database_url": sanitized,
        "connection_status": "FAIL",
        "postgres_version": None,
        "pgvector_available": False,
        "tables_detected": [],
        "errors": [],
    }

    try:
        # Use NullPool with reasonable connect_timeout for probe so health checks never hang
        probe_connect_args: Dict[str, Any] = {"connect_timeout": 10}
        if config.db.ssl_mode:
            probe_connect_args["sslmode"] = config.db.ssl_mode

        probe_engine = create_engine(
            target_url,
            connect_args=probe_connect_args,
            pool_pre_ping=False,
        )
        with probe_engine.connect() as conn:
            # 1. Connectivity & PostgreSQL version
            version_row = conn.execute(text("SELECT version();")).scalar()
            result["postgres_version"] = str(version_row)
            result["connection_status"] = "PASS"

            # 2. Check pgvector extension
            ext_row = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ).scalar()
            result["pgvector_available"] = bool(ext_row)

            # 3. Check existing tables
            tables_query = text(
                """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public';
                """
            )
            tables = [row[0] for row in conn.execute(tables_query).fetchall()]
            result["tables_detected"] = sorted(tables)

    except Exception as exc:
        logger.error("Database health check failed for %s: %s", sanitized, exc)
        result["errors"].append(str(exc))

    return result


def print_health_report(health: Dict[str, Any]) -> None:
    """Prints formatted health check summary to console."""
    sep = "=" * 55
    print(f"\n{sep}")
    print("Database Health & Connectivity Check")
    print(sep)
    print(f"Target URL:         {health['database_url']}")
    print(f"Connection Status:  {health['connection_status']}")

    if health["postgres_version"]:
        print(f"PostgreSQL Version: {health['postgres_version'][:40]}...")
    else:
        print("PostgreSQL Version: Not detected")

    print(f"pgvector Extension: {'DETECTED' if health['pgvector_available'] else 'NOT DETECTED'}")
    print(f"Tables Detected:    {', '.join(health['tables_detected']) or 'None'}")
    print(sep)

    if health["errors"]:
        print("\nErrors / Warnings:")
        for err in health["errors"]:
            print(f"- {err}")
        print(sep)


def main() -> None:
    """CLI entrypoint for database verification."""
    health = check_db_health()
    print_health_report(health)


if __name__ == "__main__":
    main()
