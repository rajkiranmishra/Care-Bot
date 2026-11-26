"""CARE-BOT FastAPI Application Factory and Lifecycle Management."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.api.v1.api import api_router
from app.api.v1.health import router as health_router
from app.config import get_settings
from app.database.base import Base
from app.database.session import engine

# Configure structured logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("carebot")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager handling startup and shutdown hooks."""
    logger.info("Starting CARE-BOT backend [version: %s, env: %s]", settings.PROJECT_VERSION, settings.ENVIRONMENT)

    # Verify database connectivity and initialize tables
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        Base.metadata.create_all(bind=engine)
        logger.info("Database connectivity established successfully (%s)", settings.DATABASE_URL)
    except Exception as exc:
        logger.error("Failed to establish database connection during startup: %s", exc)

    # Initialize Local LLM Client if configured for Ollama
    app.state.llm_client = None
    if settings.LOCAL_LLM_PROVIDER.lower() == "ollama":
        try:
            from app.intelligence.ollama_client import OllamaLLMClient
            app.state.llm_client = OllamaLLMClient(
                base_url=settings.OLLAMA_BASE_URL,
                model_name=settings.LOCAL_LLM_MODEL,
                timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
                temperature=settings.LLM_TEMPERATURE,
            )
            logger.info(
                "Initialized persistent OllamaLLMClient [model: %s, url: %s]",
                settings.LOCAL_LLM_MODEL,
                settings.OLLAMA_BASE_URL,
            )
        except Exception as exc:
            logger.warning("Could not initialize OllamaLLMClient during startup: %s", exc)

    yield

    logger.info("Shutting down CARE-BOT backend cleanly")
    if hasattr(app.state, "llm_client") and app.state.llm_client is not None:
        try:
            await app.state.llm_client.aclose()
        except Exception as exc:
            logger.warning("Error closing OllamaLLMClient during shutdown: %s", exc)
        finally:
            app.state.llm_client = None

    engine.dispose()


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.PROJECT_VERSION,
        description="Context-Aware, Multilingual AI-Powered Home Assistance System",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Configure Cross-Origin Resource Sharing (CORS)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount Root Health Endpoint for fast load balancer / monitoring checks
    application.include_router(health_router, prefix="")

    # Mount API Version 1 Router
    application.include_router(api_router, prefix=settings.API_V1_STR)

    @application.get("/", tags=["Root"])
    def root_info():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.PROJECT_VERSION,
            "status": "online",
            "docs": "/docs" if settings.DEBUG else "disabled",
            "health": "/health",
        }

    return application


app = create_application()
