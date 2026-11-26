"""Ollama Local LLM Client implementation for CARE-BOT."""

import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from app.intelligence.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
)
from app.intelligence.schemas import LLMResponse, ModelMetadata

logger = logging.getLogger("carebot.intelligence.ollama")


class OllamaLLMClient:
    """Asynchronous client connecting to a local Ollama server over HTTP.

    Conforms strictly to the LocalLLMClient protocol.

    ARCHITECTURAL & SAFETY INVARIANTS:
    1. Zero Database Access: The client has no database session or SQL execution capability.
    2. Advisory Only: Outputs are raw strings; actions must be whitelisted and authorized server-side.
    3. Failure Isolation: Network errors, timeouts, or malformed responses raise defined LLMClientError
       subclasses and never expose internal traces or secrets to residents.
    4. Lifecycle Management: Can reuse an external httpx.AsyncClient connection pool or manage its own.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b",
        timeout_seconds: float = 12.0,
        temperature: float = 0.2,
        max_tokens: int = 512,
        context_window: int = 32768,
        quantization: Optional[str] = "Q4_K_M",
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url: str = base_url.rstrip("/")
        self.model_name: str = model_name
        self.timeout_seconds: float = timeout_seconds
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.context_window: int = context_window
        self.quantization: Optional[str] = quantization
        self._client: Optional[httpx.AsyncClient] = http_client
        self._owns_client: bool = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Retrieve or lazily initialize the asynchronous HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
            )
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the underlying HTTP client session if owned by this instance."""
        if self._owns_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "OllamaLLMClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout_seconds: float = 10.0,
    ) -> LLMResponse:
        """Execute chat completion against the local Ollama API (/api/chat).

        Args:
            prompt: User utterance or formatted dialogue prompt.
            system_prompt: Optional system persona and schema instructions.
            json_mode: Whether to enforce structured JSON output in Ollama decoding.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens: Maximum completion tokens to generate (num_predict).
            timeout_seconds: Request timeout in seconds.

        Returns:
            LLMResponse containing raw response content, token metrics, and latency.

        Raises:
            LLMTimeoutError: Request timed out.
            LLMConnectionError: Cannot connect to Ollama daemon.
            LLMClientError: HTTP error, missing model, or malformed payload.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        request_timeout = httpx.Timeout(timeout_seconds, connect=5.0)
        start_time = time.perf_counter()

        try:
            client = self._get_client()
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Ollama inference timed out after %s seconds: %s", timeout_seconds, exc)
            raise LLMTimeoutError(
                f"Ollama inference timed out after {timeout_seconds} seconds"
            ) from exc
        except httpx.ConnectError as exc:
            logger.warning("Failed to connect to Ollama server at %s: %s", self.base_url, exc)
            raise LLMConnectionError(
                f"Failed to connect to Ollama server at {self.base_url}: connection refused"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Network communication error with Ollama at %s: %s", self.base_url, exc)
            raise LLMClientError(f"Network error communicating with Ollama: {exc}") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code == 404:
            logger.error("Configured Ollama model '%s' was not found on the server", self.model_name)
            raise LLMClientError(
                f"Ollama model '{self.model_name}' not found at {self.base_url}. "
                f"Ensure the model is pulled using 'ollama pull {self.model_name}'."
            )

        if response.is_error:
            logger.error("Ollama HTTP failure [%s]: %s", response.status_code, response.text[:200])
            raise LLMClientError(
                f"Ollama server returned HTTP {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.error("Failed to decode JSON from Ollama response: %s", exc)
            raise LLMClientError(f"Malformed JSON response received from Ollama server: {exc}") from exc

        if not isinstance(data, dict):
            raise LLMClientError("Ollama response root must be a JSON object")

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            logger.error("Ollama response missing message.content payload: %s", data)
            raise LLMClientError("Ollama response missing 'message.content' field")

        content = str(message.get("content", ""))

        total_duration_ns = data.get("total_duration")
        latency_ms = (total_duration_ns / 1_000_000.0) if total_duration_ns is not None else elapsed_ms

        return LLMResponse(
            content=content,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            latency_ms=round(latency_ms, 2),
            metadata={
                "provider": "ollama",
                "model": data.get("model", self.model_name),
                "done": data.get("done", True),
                "done_reason": data.get("done_reason"),
            },
        )

    async def health_check(self) -> bool:
        """Verify the inference server is alive and ready to process requests."""
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.base_url}/api/version",
                timeout=httpx.Timeout(3.0, connect=2.0),
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    def get_model_info(self) -> ModelMetadata:
        """Return metadata for the currently active local model."""
        return ModelMetadata(
            model_name=self.model_name,
            provider="ollama",
            context_window=self.context_window,
            quantization=self.quantization,
        )
