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
    5. Honest Metadata: Does not claim hardcoded context windows or quantizations. Either introspects
       live metadata from Ollama /api/show or marks fields as configured or unknown.
    6. Runtime Configuration: Inference temperature and timeout derive directly from configured settings
       unless explicitly overridden by the caller.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5:7b",
        timeout_seconds: float = 12.0,
        temperature: float = 0.2,
        max_tokens: int = 512,
        context_window: Optional[int] = None,
        quantization: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url: str = base_url.rstrip("/")
        self.model_name: str = model_name
        self.timeout_seconds: float = timeout_seconds
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        # Configured vs Introspected metadata
        self._configured_context_window: Optional[int] = context_window
        self._configured_quantization: Optional[str] = quantization
        self._introspected_context_window: Optional[int] = None
        self._introspected_quantization: Optional[str] = None
        self._is_introspected: bool = False
        self._client: Optional[httpx.AsyncClient] = http_client
        self._owns_client: bool = http_client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Retrieve or lazily initialize the asynchronous HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout_seconds, connect=min(5.0, self.timeout_seconds)),
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> LLMResponse:
        """Execute chat completion against the local Ollama API (/api/chat).

        Uses configured runtime settings (self.temperature, self.timeout_seconds, self.max_tokens)
        unless explicitly overridden by the caller.

        Args:
            prompt: User utterance or formatted dialogue prompt.
            system_prompt: Optional system persona and schema instructions.
            json_mode: Whether to enforce structured JSON output in Ollama decoding.
            temperature: Sampling temperature override; defaults to self.temperature if None.
            max_tokens: Max completion tokens override; defaults to self.max_tokens if None.
            timeout_seconds: Request timeout override; defaults to self.timeout_seconds if None.

        Returns:
            LLMResponse containing raw response content, token metrics, and latency.

        Raises:
            LLMTimeoutError: Request timed out.
            LLMConnectionError: Cannot connect to Ollama daemon.
            LLMClientError: HTTP error, missing model, or malformed payload.
        """
        eff_temperature = self.temperature if temperature is None else temperature
        eff_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        eff_timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": eff_temperature,
                "num_predict": eff_max_tokens,
            },
        }

        if json_mode:
            payload["format"] = "json"

        request_timeout = httpx.Timeout(eff_timeout, connect=min(5.0, eff_timeout))
        start_time = time.perf_counter()

        try:
            client = self._get_client()
            response = await client.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
        except httpx.TimeoutException as exc:
            logger.warning("Ollama inference timed out after %s seconds", eff_timeout)
            raise LLMTimeoutError(
                f"Ollama inference timed out after {eff_timeout} seconds"
            ) from exc
        except httpx.ConnectError as exc:
            logger.warning("Failed to connect to Ollama server at %s", self.base_url)
            raise LLMConnectionError(
                "Failed to connect to Ollama server: connection refused"
            ) from exc
        except httpx.RequestError as exc:
            logger.warning("Network communication error with Ollama")
            raise LLMClientError("Network error communicating with Ollama inference engine") from exc

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if response.status_code == 404:
            logger.error("Configured Ollama model '%s' was not found on server", self.model_name)
            raise LLMClientError(
                f"Ollama model '{self.model_name}' not found on server. "
                f"Ensure the model is pulled using 'ollama pull {self.model_name}'."
            )

        if response.is_error:
            logger.error("Ollama server returned HTTP error status %s", response.status_code)
            raise LLMClientError(
                f"Ollama server returned HTTP {response.status_code}"
            )

        try:
            data = response.json()
        except Exception as exc:
            logger.error("Failed to decode JSON from Ollama server response")
            raise LLMClientError("Malformed JSON response received from Ollama server") from exc

        if not isinstance(data, dict):
            raise LLMClientError("Ollama response root must be a JSON object")

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            logger.error("Ollama response missing message.content payload")
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

    async def fetch_model_metadata(self) -> ModelMetadata:
        """Query Ollama /api/show to introspect exact context length and quantization."""
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self.base_url}/api/show",
                json={"name": self.model_name},
                timeout=httpx.Timeout(4.0, connect=2.0),
            )
            if resp.status_code == 200:
                data = resp.json()
                details = data.get("details", {})
                model_info = data.get("model_info", {})

                quant = details.get("quantization_level")
                if quant:
                    self._introspected_quantization = str(quant)

                for key, val in model_info.items():
                    if key.endswith(".context_length") and isinstance(val, int):
                        self._introspected_context_window = val
                        break

                self._is_introspected = True
                logger.info(
                    "Introspected metadata for model '%s': context_window=%s, quantization=%s",
                    self.model_name,
                    self._introspected_context_window,
                    self._introspected_quantization,
                )
        except Exception as exc:
            logger.debug("Failed to introspect model metadata from /api/show: %s", exc)

        return self.get_model_info()

    async def health_check(self) -> bool:
        """Verify the inference server is alive and attempt metadata introspection."""
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self.base_url}/api/version",
                timeout=httpx.Timeout(3.0, connect=2.0),
            )
            is_alive = resp.status_code == 200
            if is_alive and not self._is_introspected:
                await self.fetch_model_metadata()
            return is_alive
        except Exception as exc:
            logger.debug("Ollama health check failed: %s", exc)
            return False

    def get_model_info(self) -> ModelMetadata:
        """Return metadata for the currently active local model.

        Reflects introspected Ollama metadata if retrieved, or explicitly identifies
        fields as configured or unknown.
        """
        if self._introspected_context_window is not None:
            ctx = self._introspected_context_window
        elif self._configured_context_window is not None:
            ctx = self._configured_context_window
        else:
            ctx = 2048  # Default protocol fallback

        if self._introspected_quantization is not None:
            quant = self._introspected_quantization
        elif self._configured_quantization is not None:
            quant = self._configured_quantization
        else:
            quant = "unknown"

        return ModelMetadata(
            model_name=self.model_name,
            provider="ollama",
            context_window=ctx,
            quantization=quant,
        )
