"""Tests for health check and diagnostic endpoints."""

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.api.deps import get_db
from app.main import app


def test_root_endpoint(client: TestClient):
    """Verify GET / returns root service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "CARE-BOT" in data["name"]
    assert "version" in data


def test_health_check_endpoint_root(client: TestClient):
    """Verify GET /health returns 200 OK and healthy status structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "CARE-BOT-TEST"
    assert data["environment"] == "testing"
    assert "timestamp" in data

    # Database diagnostics
    assert data["database"]["status"] == "connected"
    assert data["database"]["engine"] == "sqlite"
    assert isinstance(data["database"]["latency_ms"], float)
    assert data["database"]["latency_ms"] >= 0.0

    # System diagnostics
    assert "python_version" in data["system"]
    assert "platform" in data["system"]
    assert "machine" in data["system"]


def test_health_check_endpoint_v1(client: TestClient):
    """Verify GET /api/v1/health functions identically under the versioned prefix."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"]["status"] == "connected"


def test_health_check_degraded_on_database_error(client: TestClient):
    """Verify that database query failures gracefully report degraded status without 500 crash."""
    mock_db = MagicMock()
    mock_db.execute.side_effect = RuntimeError("Simulated connection timeout")

    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert "unhealthy" in data["database"]["status"]
        assert "Simulated connection timeout" in data["database"]["status"]
    finally:
        app.dependency_overrides.pop(get_db, None)
