"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration parameters for CARE-BOT backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Project Metadata
    PROJECT_NAME: str = "CARE-BOT"
    PROJECT_VERSION: str = "0.1.0"
    ENVIRONMENT: str = Field(default="development", description="Environment mode: development, testing, production")
    DEBUG: bool = Field(default=True, description="Enable debug diagnostics")
    LOG_LEVEL: str = Field(default="INFO", description="Log level")

    # Cryptographic Security & Tokens
    # In production, SECRET_KEY must be a cryptographically strong string of at least 32 characters.
    SECRET_KEY: str = Field(
        default="",
        description="Cryptographic secret key for signing tokens. Must be configured via environment variable.",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=60,
        description="Access token lifespan in minutes",
    )
    # Hardware device pre-shared key (e.g. NFC tap, emergency button).
    # Has NO insecure default; must be explicitly configured in environment.
    DEVICE_API_KEY: Optional[str] = Field(
        default=None,
        description="Shared secret key authenticating physical smart-speaker hardware. Must be configured in .env",
    )
    # One-time bootstrap secret token for initial administrator / caregiver provisioning.
    # Passed via X-Bootstrap-Token header. Disabled once an administrator account exists.
    BOOTSTRAP_TOKEN: Optional[str] = Field(
        default=None,
        description="One-time secret token for initial administrator provisioning. Must be configured in .env",
    )

    # Networking & Server
    HOST: str = Field(default="127.0.0.1", description="Server bind host")
    PORT: int = Field(default=8000, description="Server port")
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000",
        description="Comma-separated allowed origins",
    )

    # Persistence (SQLite Default)
    DATABASE_URL: str = Field(
        default="sqlite:///./carebot.db",
        description="SQLAlchemy database URL",
    )

    # API Configuration
    API_V1_STR: str = "/api/v1"

    # Local AI & LLM Engine Settings
    LOCAL_LLM_PROVIDER: str = Field(
        default="mock",
        description="Local LLM inference provider: 'mock' for tests/offline, 'ollama' for real local inference",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama daemon HTTP API",
    )
    LOCAL_LLM_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Model name/tag (e.g. qwen2.5:7b for primary development, qwen2.5:3b for lightweight alternative)",
    )
    LLM_TIMEOUT_SECONDS: float = Field(
        default=12.0,
        ge=1.0,
        description="Timeout in seconds for local LLM inference requests",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for local LLM inference",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list of cleaned strings."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_sqlite(self) -> bool:
        """Check if active database URL targets SQLite."""
        return self.DATABASE_URL.startswith("sqlite")

    @model_validator(mode="after")
    def validate_security_configuration(self) -> "Settings":
        """Enforce strict secret requirements in production and prevent insecure defaults."""
        # Validate LLM Provider
        provider = self.LOCAL_LLM_PROVIDER.lower().strip()
        if provider not in ("mock", "ollama"):
            raise ValueError(f"LOCAL_LLM_PROVIDER must be 'mock' or 'ollama', got '{self.LOCAL_LLM_PROVIDER}'")
        self.LOCAL_LLM_PROVIDER = provider

        is_prod = self.ENVIRONMENT.lower() == "production"
        if is_prod:
            if (
                not self.SECRET_KEY
                or len(self.SECRET_KEY) < 32
                or "dev" in self.SECRET_KEY.lower()
                or "insecure" in self.SECRET_KEY.lower()
            ):
                raise ValueError(
                    "In production, SECRET_KEY must be configured with a cryptographically secure string of at least 32 characters."
                )
            if not self.DEVICE_API_KEY or len(self.DEVICE_API_KEY) < 16 or "default" in self.DEVICE_API_KEY.lower():
                raise ValueError(
                    "In production, DEVICE_API_KEY must be explicitly set to a strong secret key (at least 16 characters)."
                )
            if self.BOOTSTRAP_TOKEN and (
                len(self.BOOTSTRAP_TOKEN) < 32 or "dev" in self.BOOTSTRAP_TOKEN.lower() or "default" in self.BOOTSTRAP_TOKEN.lower()
            ):
                raise ValueError(
                    "In production, BOOTSTRAP_TOKEN must be configured with a strong secret of at least 32 characters."
                )
        else:
            # In development/testing, if SECRET_KEY was not provided in .env, supply a 32+ character dev key
            if not self.SECRET_KEY:
                self.SECRET_KEY = "carebot-ephemeral-dev-key-32-bytes-minimum-length!"
            if self.BOOTSTRAP_TOKEN and len(self.BOOTSTRAP_TOKEN) < 16:
                raise ValueError("BOOTSTRAP_TOKEN, if configured, must be at least 16 characters long.")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
