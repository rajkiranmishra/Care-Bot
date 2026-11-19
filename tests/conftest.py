"""Global pytest fixtures and test configuration."""

from typing import Dict, Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import Settings, get_settings
from app.core.security import create_access_token
from app.database.base import Base
from app.database.session import get_db
from app.main import app

# In-memory SQLite engine configured for isolated unit testing
TEST_DATABASE_URL = "sqlite:///:memory:"
TEST_SECRET_KEY = "carebot-test-cryptographic-secret-key-32-chars!"
TEST_DEVICE_KEY = "carebot-hardware-device-key-local-2026"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def make_auth_headers(
    user_id: int,
    username: str = "testuser",
    role: str = "resident",
    secret_key: str = TEST_SECRET_KEY,
    expires_in_seconds: int = 3600,
) -> Dict[str, str]:
    """Helper to generate standard Authorization: Bearer headers for tests."""
    token = create_access_token(
        claims={"sub": str(user_id), "username": username, "role": role},
        secret_key=secret_key,
        expires_in_seconds=expires_in_seconds,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all database tables for test session and tear down at completion."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide a clean, transaction-scoped database session per test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def custom_settings() -> Settings:
    """Return test-specific settings instance."""
    return Settings(
        PROJECT_NAME="CARE-BOT-TEST",
        PROJECT_VERSION="0.1.0-test",
        ENVIRONMENT="testing",
        DEBUG=True,
        DATABASE_URL=TEST_DATABASE_URL,
        SECRET_KEY=TEST_SECRET_KEY,
        DEVICE_API_KEY=TEST_DEVICE_KEY,
    )


@pytest.fixture
def client(db_session: Session, custom_settings: Settings) -> Generator[TestClient, None, None]:
    """Provide a TestClient with database session and settings overrides."""
    def override_get_db():
        yield db_session

    def override_get_settings():
        return custom_settings

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_settings] = override_get_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
