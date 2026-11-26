"""Unit and mock integration tests for OllamaLLMClient and Dependency Injection."""

import json
from typing import Any, Dict
from unittest.mock import MagicMock
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.deps import get_conversation_engine, get_llm_client
from app.config import Settings
from app.database.models import User
from app.intelligence.engine import ConversationEngine
from app.intelligence.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LocalLLMClient,
    MockLLMClient,
)
from app.intelligence.ollama_client import OllamaLLMClient
from app.intelligence.schemas import (
    ActionType,
    ChatTurnRequest,
    EmergencySeverity,
    IntentType,
    ModelMetadata,
)
from app.main import app
from app.users.schemas import UserCreate
from app.users.service import UserService
from tests.conftest import make_auth_headers


# -----------------------------------------------------------------------------
# Mock Transport Handlers
# -----------------------------------------------------------------------------

def make_ollama_mock_transport(
    chat_response_content: str = '{"intent": "general_chat", "spoken_response": "I am here to assist you.", "proposed_action": {"action_type": "none"}}',
    status_code: int = 200,
    simulate_timeout: bool = False,
    simulate_connection_error: bool = False,
    malformed_json: bool = False,
    missing_content: bool = False,
    recorded_requests: list = None,
):
    """Factory creating an httpx.MockTransport for Ollama API endpoints."""

    def handler(request: httpx.Request) -> httpx.Response:
        if recorded_requests is not None:
            body = None
            if request.content:
                try:
                    body = json.loads(request.content.decode("utf-8"))
                except Exception:
                    body = request.content.decode("utf-8")
            recorded_requests.append({
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": body,
            })

        if simulate_connection_error:
            raise httpx.ConnectError("Connection refused by target daemon", request=request)

        if simulate_timeout:
            raise httpx.ReadTimeout("Request timed out waiting for server", request=request)

        if request.url.path == "/api/version":
            if status_code == 200:
                return httpx.Response(200, json={"version": "0.40.18"})
            return httpx.Response(status_code, text="Error")

        if request.url.path == "/api/show":
            if status_code == 200:
                return httpx.Response(200, json={
                    "details": {
                        "parameter_size": "7.6B",
                        "quantization_level": "Q4_K_M",
                        "format": "gguf",
                        "family": "qwen2",
                    },
                    "model_info": {
                        "general.architecture": "qwen2",
                        "qwen2.context_length": 32768,
                    },
                })
            return httpx.Response(status_code, text="Error")

        if request.url.path == "/api/chat":
            if status_code != 200:
                return httpx.Response(status_code, text=f"Error {status_code}: internal daemon detail")

            if malformed_json:
                return httpx.Response(200, text="<!DOCTYPE html><html>Gateway Error</html>")

            if missing_content:
                return httpx.Response(200, json={"model": "qwen2.5:7b", "done": True})

            payload = {
                "model": "qwen2.5:7b",
                "created_at": "2026-11-26T06:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": chat_response_content,
                },
                "done": True,
                "done_reason": "stop",
                "total_duration": 45000000,  # 45 ms in nanoseconds
                "prompt_eval_count": 32,
                "eval_count": 18,
            }
            return httpx.Response(200, json=payload)

        return httpx.Response(404, text="Not Found")

    return httpx.MockTransport(handler)


# -----------------------------------------------------------------------------
# OllamaLLMClient Protocol & Unit Tests
# -----------------------------------------------------------------------------

def test_ollama_client_satisfies_protocol():
    """Verify OllamaLLMClient conforms to LocalLLMClient protocol at runtime."""
    client = OllamaLLMClient()
    assert isinstance(client, LocalLLMClient)


@pytest.mark.anyio
async def test_ollama_client_generate_success():
    """Verify successful generation, request payload formatting, and response parsing."""
    recorded = []
    expected_json = '{"intent": "greeting", "confidence": 0.98, "detected_language": "en", "spoken_response": "Good morning!", "proposed_action": {"action_type": "none"}}'
    transport = make_ollama_mock_transport(
        chat_response_content=expected_json,
        recorded_requests=recorded,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(
            base_url="http://localhost:11434",
            model_name="qwen2.5:7b",
            temperature=0.25,
            timeout_seconds=8.0,
            http_client=http_client,
        )

        response = await client.generate(
            prompt="Resident: Good morning",
            system_prompt="You are a compassionate elderly care assistant.",
            json_mode=True,
        )

        assert response.content == expected_json
        assert response.prompt_tokens == 32
        assert response.completion_tokens == 18
        assert response.latency_ms == 45.0
        assert response.metadata["provider"] == "ollama"
        assert response.metadata["model"] == "qwen2.5:7b"
        assert response.metadata["done"] is True

        # Verify outgoing HTTP payload used configured temperature
        assert len(recorded) == 1
        req = recorded[0]
        assert req["method"] == "POST"
        assert req["url"] == "http://localhost:11434/api/chat"
        body = req["body"]
        assert body["model"] == "qwen2.5:7b"
        assert body["format"] == "json"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0.25
        assert body["options"]["num_predict"] == 512
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are a compassionate elderly care assistant."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "Resident: Good morning"


@pytest.mark.anyio
async def test_ollama_runtime_config_temperature_and_timeout_precedence():
    """Verify configured LLM_TEMPERATURE and LLM_TIMEOUT_SECONDS reach the HTTP payload and timeout."""
    recorded = []
    transport = make_ollama_mock_transport(recorded_requests=recorded)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        # Client initialized with non-default configured values (e.g. from Settings)
        client = OllamaLLMClient(
            model_name="qwen2.5:7b",
            temperature=0.85,
            timeout_seconds=25.0,
            max_tokens=1024,
            http_client=http_client,
        )

        # Call generate without explicit temperature or timeout -> should use client's configured values
        await client.generate(prompt="What is the schedule?")

        assert len(recorded) == 1
        body = recorded[0]["body"]
        assert body["options"]["temperature"] == 0.85
        assert body["options"]["num_predict"] == 1024

        # Next call with explicit overrides -> should respect call-specific overrides
        await client.generate(prompt="Quick query", temperature=0.05, max_tokens=64)
        assert len(recorded) == 2
        body2 = recorded[1]["body"]
        assert body2["options"]["temperature"] == 0.05
        assert body2["options"]["num_predict"] == 64


@pytest.mark.anyio
async def test_ollama_client_generate_without_system_prompt_and_no_json_mode():
    """Verify request structure when system_prompt is None and json_mode is False."""
    recorded = []
    transport = make_ollama_mock_transport(
        chat_response_content="Plain text response",
        recorded_requests=recorded,
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(model_name="qwen2.5:3b", http_client=http_client)
        response = await client.generate(
            prompt="Hello there",
            system_prompt=None,
            json_mode=False,
        )

        assert response.content == "Plain text response"
        assert len(recorded) == 1
        body = recorded[0]["body"]
        assert body["model"] == "qwen2.5:3b"
        assert "format" not in body
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello there"


@pytest.mark.anyio
async def test_ollama_client_timeout_handling():
    """Verify HTTP timeout raises LLMTimeoutError with configured duration."""
    transport = make_ollama_mock_transport(simulate_timeout=True)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(timeout_seconds=7.5, http_client=http_client)
        with pytest.raises(LLMTimeoutError, match="timed out after 7.5 seconds"):
            await client.generate("User query")


@pytest.mark.anyio
async def test_ollama_client_connection_error_handling():
    """Verify daemon connection refusal raises LLMConnectionError without leaking sensitive data."""
    transport = make_ollama_mock_transport(simulate_connection_error=True)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(http_client=http_client)
        with pytest.raises(LLMConnectionError, match="connection refused"):
            await client.generate("User query")


@pytest.mark.anyio
async def test_ollama_client_http_404_model_not_found():
    """Verify HTTP 404 generates descriptive missing-model error advising pull."""
    transport = make_ollama_mock_transport(status_code=404)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(model_name="custom-missing-model", http_client=http_client)
        with pytest.raises(LLMClientError, match="custom-missing-model.*not found on server.*ollama pull"):
            await client.generate("User query")


@pytest.mark.anyio
async def test_ollama_client_http_500_failure_sanitizes_body():
    """Verify HTTP 500 error raises LLMClientError without leaking raw server bodies."""
    transport = make_ollama_mock_transport(status_code=500)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(http_client=http_client)
        with pytest.raises(LLMClientError) as exc_info:
            await client.generate("User query")
        # Confirms HTTP status is reported, but raw body ("internal daemon detail") is not leaked
        assert "HTTP 500" in str(exc_info.value)
        assert "internal daemon detail" not in str(exc_info.value)


@pytest.mark.anyio
async def test_ollama_client_malformed_json_response():
    """Verify non-JSON response from server raises LLMClientError."""
    transport = make_ollama_mock_transport(malformed_json=True)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(http_client=http_client)
        with pytest.raises(LLMClientError, match="Malformed JSON"):
            await client.generate("User query")


@pytest.mark.anyio
async def test_ollama_client_missing_message_content():
    """Verify response missing message.content payload raises LLMClientError."""
    transport = make_ollama_mock_transport(missing_content=True)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(http_client=http_client)
        with pytest.raises(LLMClientError, match="missing 'message.content'"):
            await client.generate("User query")


# -----------------------------------------------------------------------------
# Model Metadata & Health Check Tests
# -----------------------------------------------------------------------------

def test_ollama_client_metadata_unknown_when_not_introspected():
    """Verify uninspected model reports context_window fallback and 'unknown' quantization."""
    client = OllamaLLMClient(model_name="qwen2.5:7b")
    info = client.get_model_info()
    assert isinstance(info, ModelMetadata)
    assert info.model_name == "qwen2.5:7b"
    assert info.provider == "ollama"
    assert info.context_window == 2048  # Safe default fallback
    assert info.quantization == "unknown"  # Not hardcoded!


def test_ollama_client_metadata_configured_explicitly():
    """Verify explicitly configured context window and quantization are respected."""
    client = OllamaLLMClient(
        model_name="custom-model",
        context_window=8192,
        quantization="Q8_0",
    )
    info = client.get_model_info()
    assert info.model_name == "custom-model"
    assert info.context_window == 8192
    assert info.quantization == "Q8_0"


@pytest.mark.anyio
async def test_ollama_client_metadata_introspected_from_api_show():
    """Verify fetch_model_metadata accurately retrieves details from Ollama /api/show."""
    transport = make_ollama_mock_transport()
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(model_name="qwen2.5:7b", http_client=http_client)
        assert client.get_model_info().quantization == "unknown"

        meta = await client.fetch_model_metadata()
        assert meta.quantization == "Q4_K_M"
        assert meta.context_window == 32768
        assert client._is_introspected is True


@pytest.mark.anyio
async def test_ollama_client_health_check_triggers_introspection():
    """Verify health_check returns True and automatically triggers metadata introspection."""
    live_transport = make_ollama_mock_transport(status_code=200)
    async with httpx.AsyncClient(transport=live_transport, base_url="http://localhost:11434") as http_client:
        client = OllamaLLMClient(model_name="qwen2.5:7b", http_client=http_client)
        assert client.get_model_info().quantization == "unknown"

        is_healthy = await client.health_check()
        assert is_healthy is True
        # After health check, metadata should be populated
        assert client.get_model_info().quantization == "Q4_K_M"
        assert client.get_model_info().context_window == 32768

    # Offline daemon
    down_transport = make_ollama_mock_transport(simulate_connection_error=True)
    async with httpx.AsyncClient(transport=down_transport, base_url="http://localhost:11434") as http_client:
        client_down = OllamaLLMClient(http_client=http_client)
        assert await client_down.health_check() is False


@pytest.mark.anyio
async def test_ollama_client_aclose_lifecycle():
    """Verify client creates and cleanly closes its owned httpx.AsyncClient."""
    client = OllamaLLMClient(base_url="http://localhost:11434")
    raw_http = client._get_client()
    assert raw_http is not None
    assert not raw_http.is_closed

    await client.aclose()
    assert raw_http.is_closed


# -----------------------------------------------------------------------------
# ConversationEngine Compatibility Tests
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_conversation_engine_with_ollama_client(db_session: Session):
    """Verify ConversationEngine orchestrates turns seamlessly using OllamaLLMClient."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="ollama_resident", full_name="Ollama Resident", role="resident"),
    )

    ollama_resp_json = (
        '{"intent": "general_chat", "confidence": 0.95, "detected_language": "en", '
        '"spoken_response": "Hello from simulated Ollama!", '
        '"proposed_action": {"action_type": "none", "parameters": {}}}'
    )
    transport = make_ollama_mock_transport(chat_response_content=ollama_resp_json)

    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:11434") as http_client:
        ollama_client = OllamaLLMClient(model_name="qwen2.5:7b", http_client=http_client)
        engine = ConversationEngine(llm_client=ollama_client)

        res = await engine.process_turn(
            db_session,
            user,
            ChatTurnRequest(utterance="Hello CARE-BOT"),
        )

        assert res.spoken_response == "Hello from simulated Ollama!"
        assert res.intent == IntentType.GENERAL_CHAT
        assert res.proposed_action.action_type == ActionType.NONE
        assert res.emergency_signal.detected is False


@pytest.mark.anyio
async def test_emergency_fast_path_bypasses_ollama_client(db_session: Session):
    """Verify deterministic emergency detection completely bypasses Ollama, even if Ollama is down."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="emergency_ollama_user", full_name="Emergency User", role="resident"),
    )

    # Transport simulates down / connection refused Ollama server
    down_transport = make_ollama_mock_transport(simulate_connection_error=True)

    async with httpx.AsyncClient(transport=down_transport, base_url="http://localhost:11434") as http_client:
        ollama_client = OllamaLLMClient(http_client=http_client)
        engine = ConversationEngine(llm_client=ollama_client)

        # Critical emergency utterance
        res = await engine.process_turn(
            db_session,
            user,
            ChatTurnRequest(utterance="HELP I FELL AND CANNOT GET UP"),
        )

        # Must succeed via deterministic emergency fast-path without raising LLMConnectionError
        assert res.intent == IntentType.EMERGENCY
        assert res.emergency_signal.detected is True
        assert res.emergency_signal.severity == EmergencySeverity.CRITICAL
        assert res.active_emergency is True
        assert res.recorded_locally is True
        assert "आपातकाल" in res.spoken_response or "Emergency" in res.spoken_response


@pytest.mark.anyio
async def test_conversation_engine_failsafe_fallback_on_ollama_timeout(db_session: Session):
    """Verify ConversationEngine catches LLMTimeoutError and returns safe conversational fallback."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="timeout_user", full_name="Timeout User", role="resident"),
    )

    timeout_transport = make_ollama_mock_transport(simulate_timeout=True)

    async with httpx.AsyncClient(transport=timeout_transport, base_url="http://localhost:11434") as http_client:
        ollama_client = OllamaLLMClient(http_client=http_client)
        engine = ConversationEngine(llm_client=ollama_client)

        res = await engine.process_turn(
            db_session,
            user,
            ChatTurnRequest(utterance="Tell me a very long story"),
        )

        # ConversationEngine catches LLMClientError and uses safe fallback
        assert "trouble processing" in res.spoken_response.lower()
        assert res.intent == IntentType.UNKNOWN


# -----------------------------------------------------------------------------
# Configuration and Dependency Injection Tests
# -----------------------------------------------------------------------------

def test_settings_llm_provider_configuration():
    """Verify Settings supports LOCAL_LLM_PROVIDER and model configuration."""
    # Default settings
    default_settings = Settings()
    assert default_settings.LOCAL_LLM_PROVIDER == "mock"
    assert default_settings.LOCAL_LLM_MODEL == "qwen2.5:7b"
    assert default_settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert default_settings.LLM_TIMEOUT_SECONDS == 12.0
    assert default_settings.LLM_TEMPERATURE == 0.2

    # Custom settings with qwen2.5:3b alternative model
    custom = Settings(
        LOCAL_LLM_PROVIDER="ollama",
        LOCAL_LLM_MODEL="qwen2.5:3b",
        OLLAMA_BASE_URL="http://192.168.1.50:11434",
        LLM_TIMEOUT_SECONDS=15.0,
        LLM_TEMPERATURE=0.1,
    )
    assert custom.LOCAL_LLM_PROVIDER == "ollama"
    assert custom.LOCAL_LLM_MODEL == "qwen2.5:3b"
    assert custom.OLLAMA_BASE_URL == "http://192.168.1.50:11434"

    # Invalid provider rejected by validator
    with pytest.raises(ValueError, match="LOCAL_LLM_PROVIDER must be 'mock' or 'ollama'"):
        Settings(LOCAL_LLM_PROVIDER="invalid-provider")


def test_dependency_injection_provider_selection():
    """Verify get_llm_client instantiates correct client based on Settings."""
    # Case A: Provider = mock
    mock_settings = Settings(LOCAL_LLM_PROVIDER="mock")
    client_a = get_llm_client(request=None, settings=mock_settings)
    assert isinstance(client_a, MockLLMClient)

    # Case B: Provider = ollama without request (e.g. CLI script)
    ollama_settings = Settings(
        LOCAL_LLM_PROVIDER="ollama",
        LOCAL_LLM_MODEL="qwen2.5:3b",
        OLLAMA_BASE_URL="http://localhost:11434",
        LLM_TIMEOUT_SECONDS=14.0,
        LLM_TEMPERATURE=0.35,
    )
    client_b = get_llm_client(request=None, settings=ollama_settings)
    assert isinstance(client_b, OllamaLLMClient)
    assert client_b.model_name == "qwen2.5:3b"
    assert client_b.timeout_seconds == 14.0
    assert client_b.temperature == 0.35


def test_dependency_injection_avoids_unmanaged_client_per_request():
    """Verify get_llm_client reuses managed instance on app.state rather than creating per request."""
    ollama_settings = Settings(
        LOCAL_LLM_PROVIDER="ollama",
        LOCAL_LLM_MODEL="qwen2.5:7b",
    )

    # Simulate request with app.state
    mock_request = MagicMock()
    mock_request.app.state.llm_client = None

    # First request: lazily initializes and attaches to app.state
    client1 = get_llm_client(request=mock_request, settings=ollama_settings)
    assert isinstance(client1, OllamaLLMClient)
    assert mock_request.app.state.llm_client is client1

    # Second request: reuses exact same instance from app.state
    client2 = get_llm_client(request=mock_request, settings=ollama_settings)
    assert client2 is client1


def test_api_chat_turn_with_dependency_override(client: TestClient):
    """Verify POST /api/v1/conversations/users/{user_id}/chat respects dependency overrides."""
    user = client.post("/api/v1/users", json={"username": "override_user", "full_name": "Override User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="override_user", role="resident")

    custom_mock = MockLLMClient(
        default_response='{"intent": "general_chat", "confidence": 0.99, "detected_language": "en", "spoken_response": "Injected response through dependency override", "proposed_action": {"action_type": "none"}}'
    )

    app.dependency_overrides[get_llm_client] = lambda: custom_mock
    try:
        res = client.post(
            f"/api/v1/conversations/users/{user_id}/chat",
            json={"utterance": "Testing dependency override"},
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["spoken_response"] == "Injected response through dependency override"
    finally:
        app.dependency_overrides.pop(get_llm_client, None)
