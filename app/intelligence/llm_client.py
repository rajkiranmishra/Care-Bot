"""Local LLM Client Protocol and Mock Implementation for CARE-BOT."""

import asyncio
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from app.intelligence.schemas import LLMResponse, ModelMetadata


class LLMClientError(Exception):
    """Base exception for local LLM inference client failures."""
    pass


class LLMTimeoutError(LLMClientError):
    """Inference request exceeded the allotted timeout threshold."""
    pass


class LLMConnectionError(LLMClientError):
    """Failed to establish a network or socket connection to the local model server."""
    pass


@runtime_checkable
class LocalLLMClient(Protocol):
    """Model-independent interface for local inference engines (Ollama, llama.cpp, mock).

    Architectural Invariant:
    Implementations of this protocol have NO database access and NO authorization authority.
    They receive pure strings and return pure inference payloads.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 10.0,
    ) -> LLMResponse:
        """Execute text generation or structured JSON completion against the local model."""
        ...

    async def health_check(self) -> bool:
        """Verify the inference server is alive and ready to process requests."""
        ...

    def get_model_info(self) -> ModelMetadata:
        """Return metadata for the currently active local model."""
        ...


class MockLLMClient:
    """Deterministic, controllable mock implementation of LocalLLMClient for testing.

    Permits testing structured JSON generation, malformed payloads, markdown wrappers,
    network timeouts, and connection outages without requiring physical GPU or weights.
    """

    def __init__(
        self,
        default_response: Optional[str] = None,
        canned_responses: Optional[List[str]] = None,
        simulate_timeout: bool = False,
        simulate_connection_error: bool = False,
        latency_ms: float = 15.0,
        model_name: str = "mock-carebot-v1",
    ):
        self._default_response = (
            default_response
            or '{"intent": "general_chat", "confidence": 0.95, "detected_language": "en", "spoken_response": "I am here to help you.", "proposed_action": {"action_type": "none", "parameters": {}}}'
        )
        self._canned_responses: List[str] = list(canned_responses or [])
        self.simulate_timeout = simulate_timeout
        self.simulate_connection_error = simulate_connection_error
        self.latency_ms = latency_ms
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    def enqueue_response(self, response_text: str) -> None:
        """Add a planned response to the queue."""
        self._canned_responses.append(response_text)

    def clear_calls(self) -> None:
        """Reset call history log."""
        self.calls.clear()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 10.0,
    ) -> LLMResponse:
        """Simulate generation with call recording and configurable error modes."""
        call_record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "json_mode": json_mode,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_seconds": timeout_seconds,
        }
        self.calls.append(call_record)

        if self.simulate_connection_error:
            raise LLMConnectionError("Failed to connect to mock inference backend: connection refused")

        if self.simulate_timeout:
            # Yield control briefly then raise timeout
            await asyncio.sleep(0.01)
            raise LLMTimeoutError(f"Inference timed out after {timeout_seconds} seconds")

        # Simulate minimal non-blocking asynchronous processing latency
        await asyncio.sleep(0.001)

        # Retrieve next canned response or fallback to default
        if self._canned_responses:
            content = self._canned_responses.pop(0)
        else:
            content = self._default_response

        return LLMResponse(
            content=content,
            prompt_tokens=len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0),
            completion_tokens=len(content.split()),
            latency_ms=self.latency_ms,
            metadata={"mock": True, "model": self.model_name},
        )

    async def health_check(self) -> bool:
        """Mock health check."""
        if self.simulate_connection_error:
            return False
        return True

    def get_model_info(self) -> ModelMetadata:
        """Return mock model metadata."""
        return ModelMetadata(
            model_name=self.model_name,
            provider="mock-in-memory",
            context_window=2048,
            quantization="none",
        )
