"""Tests for application settings and configuration loading."""

from app.config import Settings


def test_default_settings():
    """Verify default configuration attributes."""
    settings = Settings()
    assert settings.PROJECT_NAME == "CARE-BOT"
    assert settings.PROJECT_VERSION == "0.1.0"
    assert settings.API_V1_STR == "/api/v1"
    assert settings.is_sqlite is True


def test_cors_origins_parsing():
    """Verify parsing of comma-separated CORS origins string."""
    settings = Settings(CORS_ORIGINS="http://localhost:3000, https://app.example.com , http://127.0.0.1:8000")
    origins = settings.cors_origins_list
    assert len(origins) == 3
    assert "http://localhost:3000" in origins
    assert "https://app.example.com" in origins
    assert "http://127.0.0.1:8000" in origins


def test_is_sqlite_property():
    """Verify is_sqlite flag behaves accurately for different URL schemes."""
    sqlite_settings = Settings(DATABASE_URL="sqlite:///./test.db")
    assert sqlite_settings.is_sqlite is True

    pg_settings = Settings(DATABASE_URL="postgresql://user:pass@localhost/db")
    assert pg_settings.is_sqlite is False
