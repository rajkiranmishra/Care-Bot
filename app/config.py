"""Application configuration management using Pydantic Settings."""

from functools import lru_cache
from typing import List
from pydantic import Field
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

    # Future AI Settings (Placeholders for upcoming phases)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LOCAL_LLM_MODEL: str = "qwen2.5:3b"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list of cleaned strings."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_sqlite(self) -> bool:
        """Check if active database URL targets SQLite."""
        return self.DATABASE_URL.startswith("sqlite")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
