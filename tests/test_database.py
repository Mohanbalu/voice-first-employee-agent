"""Unit tests for Database Connection and Health Module (Module 3.3).

Covers:
1. Database configuration parsing
2. Password and credential sanitization
3. Safe engine initialization without crashing on unreachable host
4. Health check error handling and report formatting
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.app.config import AppConfig, sanitize_db_url
from backend.app.database import check_db_health, print_health_report


class TestDatabaseConfig:
    """Tests configuration and credential masking."""

    def test_sanitize_db_url_with_password(self):
        url = "postgresql+psycopg://myuser:supersecretpass123@db.example.com:5432/mycorp_db"
        sanitized = sanitize_db_url(url)

        assert "supersecretpass123" not in sanitized
        assert "myuser" in sanitized
        assert "****" in sanitized
        assert "db.example.com:5432/mycorp_db" in sanitized

    def test_sanitize_db_url_none(self):
        assert sanitize_db_url(None) == "None"

    def test_sanitize_db_url_without_password(self):
        url = "postgresql+psycopg://localhost:5432/mydb"
        sanitized = sanitize_db_url(url)
        assert sanitized == url

    def test_config_loader_respects_environment(self):
        with patch.dict(
            "os.environ",
            {
                "DATABASE_URL": "postgresql+psycopg://test_user:test_pass@testhost:5432/testdb",
                "DB_POOL_SIZE": "15",
                "DB_MAX_OVERFLOW": "25",
                "VECTOR_DIMENSION": "1536",
            },
            clear=True,
        ):
            app_cfg = AppConfig.load()
            assert app_cfg.db.url == "postgresql+psycopg://test_user:test_pass@testhost:5432/testdb"
            assert app_cfg.db.pool_size == 15
            assert app_cfg.db.max_overflow == 25
            assert app_cfg.db.vector_dimension == 1536


class TestDatabaseHealth:
    """Tests health check error handling without exposing secrets."""

    def test_unreachable_db_returns_clean_fail(self):
        dummy_url = "postgresql+psycopg://user:hiddenpass@127.0.0.1:59999/nonexistent"

        # Simulate connection error cleanly
        with patch("sqlalchemy.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.connect.side_effect = ConnectionRefusedError("Connection refused to 127.0.0.1:59999")
            mock_create.return_value = mock_engine

            health = check_db_health(dummy_url)

            assert health["connection_status"] == "FAIL"
            assert health["postgres_version"] is None
            assert health["pgvector_available"] is False
            assert len(health["errors"]) > 0

            # Verify password is never in the sanitized URL or report
            assert "hiddenpass" not in health["database_url"]
            assert "****" in health["database_url"]

    def test_health_report_formatting(self, capsys):
        mock_health = {
            "database_url": "postgresql+psycopg://dbuser:****@testhost:5432/testdb",
            "connection_status": "PASS",
            "postgres_version": "PostgreSQL 16.2 on x86_64",
            "pgvector_available": True,
            "tables_detected": ["tenants", "documents", "chunks", "chunk_embeddings"],
            "errors": [],
        }
        print_health_report(mock_health)
        captured = capsys.readouterr().out

        assert "Database Health & Connectivity Check" in captured
        assert "Connection Status:  PASS" in captured
        assert "pgvector Extension: DETECTED" in captured
        assert "chunks, documents, tenants" in captured or "chunk_embeddings" in captured
