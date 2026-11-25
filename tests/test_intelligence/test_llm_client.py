"""Unit tests for MockLLMClient and LocalLLMClient protocol conformance."""

import pytest
from app.intelligence.llm_client import (
    LLMConnectionError,
    LLMTimeoutError,
    LocalLLMClient,
    MockLLMClient,
)


def test_mock_llm_client_protocol_conformance():
    """Verify MockLLMClient satisfies the LocalLLMClient runtime checkable protocol."""
    mock_client = MockLLMClient()
    assert isinstance(mock_client, LocalLLMClient)


@pytest.mark.anyio
async def test_mock_llm_client_generation_and_history():
    """Verify standard generation, call recording, and token metrics."""
    client = MockLLMClient(model_name="mock-qwen-3b")
    response = await client.generate(
        prompt="User: What is the weather?",
        system_prompt="System instructions",
        temperature=0.1,
    )

    assert response.content is not None
    assert response.prompt_tokens > 0
    assert response.completion_tokens > 0
    assert len(client.calls) == 1
    assert client.calls[0]["temperature"] == 0.1
    assert client.calls[0]["system_prompt"] == "System instructions"


@pytest.mark.anyio
async def test_mock_llm_client_queued_responses():
    """Verify custom queued responses are yielded sequentially."""
    client = MockLLMClient()
    client.enqueue_response('{"intent": "weather_query", "spoken_response": "Sunny"}')
    client.enqueue_response('{"intent": "greeting", "spoken_response": "Hello"}')

    r1 = await client.generate("Weather?")
    assert "Sunny" in r1.content

    r2 = await client.generate("Hi")
    assert "Hello" in r2.content

    # Third call falls back to default
    r3 = await client.generate("Anything")
    assert "here to help" in r3.content


@pytest.mark.anyio
async def test_mock_llm_client_simulated_timeout():
    """Verify timeout simulation raises LLMTimeoutError."""
    client = MockLLMClient(simulate_timeout=True)
    with pytest.raises(LLMTimeoutError, match="timed out"):
        await client.generate("Test prompt", timeout_seconds=2.0)


@pytest.mark.anyio
async def test_mock_llm_client_simulated_connection_error():
    """Verify connection failure simulation raises LLMConnectionError."""
    client = MockLLMClient(simulate_connection_error=True)
    with pytest.raises(LLMConnectionError, match="Failed to connect"):
        await client.generate("Test prompt")

    is_healthy = await client.health_check()
    assert is_healthy is False


def test_mock_llm_client_model_metadata():
    """Verify get_model_info returns expected ModelMetadata."""
    client = MockLLMClient(model_name="test-model-4b")
    info = client.get_model_info()
    assert info.model_name == "test-model-4b"
    assert info.provider == "mock-in-memory"
    assert info.context_window == 2048
