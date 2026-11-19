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


def test_device_api_key_has_no_insecure_default():
    """Verify DEVICE_API_KEY defaults to None and has no insecure default in settings."""
    settings = Settings()
    assert settings.DEVICE_API_KEY is None


def test_production_settings_rejects_missing_or_weak_secrets():
    """Verify production environment strictly enforces strong secrets on boot."""
    import pytest

    # Missing SECRET_KEY in production
    with pytest.raises(ValueError, match="SECRET_KEY must be configured"):
        Settings(ENVIRONMENT="production", SECRET_KEY="", DEVICE_API_KEY="a-strong-device-key-16-bytes")

    # Weak/insecure SECRET_KEY in production
    with pytest.raises(ValueError, match="SECRET_KEY must be configured"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="short-secret",
            DEVICE_API_KEY="a-strong-device-key-16-bytes",
        )

    # Missing DEVICE_API_KEY in production
    with pytest.raises(ValueError, match="DEVICE_API_KEY must be explicitly set"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="a-very-long-cryptographically-strong-secret-key-32-chars!",
            DEVICE_API_KEY=None,
        )
