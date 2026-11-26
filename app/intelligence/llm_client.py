"""Local LLM Client Protocol and Mock Implementation for CARE-BOT."""

import asyncio
import re
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
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
        self._custom_default = default_response is not None
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

    def _detect_unsupported_personal_query(self, prompt: str, system_prompt: Optional[str]) -> Optional[str]:
        """Detect queries requesting resident-specific facts that are not available.

        SAFETY INVARIANT:
        The mock LLM MUST NOT invent medication names, schedules, reminders, medical history,
        appointments, or other resident-specific facts. If verified data is not present in
        the context, it states clearly that it cannot access the information.
        """
        combined = f"{system_prompt or ''}\n{prompt}".lower()

        # Check language
        is_hindi = "hindi" in combined or "devanagari" in combined
        is_bengali = "bengali" in combined

        # 1. Medication or medical history queries
        if re.search(
            r"\b(medication|medications|medicine|medicines|pill|pills|prescription|dose|dosage|medical\s+history|medical\s+records|diagnosis)\b",
            prompt,
            re.IGNORECASE,
        ):
            if is_hindi:
                resp = "मेरे पास आपकी दवा की समय-सारणी या चिकित्सा रिकॉर्ड की जानकारी नहीं है। कृपया अपने देखभालकर्ता या डॉक्टर से संपर्क करें।"
            elif is_bengali:
                resp = "আমার কাছে আপনার ওষুধের সময়সূচী বা চিকিৎসা রেকর্ডের তথ্য নেই। অনুগ্রহ করে আপনার কেয়ারগিভার বা ডাক্তারের সাথে পরামর্শ করুন।"
            else:
                resp = "I do not have access to your medication schedule or medical records. Please consult your caregiver or healthcare provider."
            return (
                '{"intent": "reminder_query", "confidence": 0.95, "detected_language": "'
                + ("hi" if is_hindi else ("bn" if is_bengali else "en"))
                + f'", "spoken_response": "{resp}", "proposed_action": {{"action_type": "none", "parameters": {{}}}}}}'
            )

        # 2. Reminder or schedule queries
        if re.search(
            r"\b(what\s+are\s+my\s+reminders|what\s+do\s+i\s+have\s+scheduled|my\s+schedule|my\s+agenda|remind\s+query|check\s+reminders|calendar)\b",
            prompt,
            re.IGNORECASE,
        ):
            if is_hindi:
                resp = "मेरे पास इस समय आपके रिमाइंडर्स या शेड्यूल की जानकारी उपलब्ध नहीं है।"
            elif is_bengali:
                resp = "এই মুহূর্তে আমার কাছে আপনার রিমাইন্ডার বা সময়সূচীর তথ্য উপলব্ধ নেই।"
            else:
                resp = "I cannot access your reminders or schedule at this time because that service is not connected."
            return (
                '{"intent": "reminder_query", "confidence": 0.95, "detected_language": "'
                + ("hi" if is_hindi else ("bn" if is_bengali else "en"))
                + f'", "spoken_response": "{resp}", "proposed_action": {{"action_type": "none", "parameters": {{}}}}}}'
            )

        # 3. Doctor appointment queries
        if re.search(
            r"\b(doctor\s+appointment|clinic\s+appointment|when\s+is\s+my\s+appointment)\b",
            prompt,
            re.IGNORECASE,
        ):
            if is_hindi:
                resp = "मेरे पास आपके अपॉइंटमेंट शेड्यूल की जानकारी नहीं है।"
            elif is_bengali:
                resp = "আমার কাছে আপনার অ্যাপয়েন্টমেন্ট সময়সূচীর তথ্য নেই।"
            else:
                resp = "I do not have access to your appointment schedule at this time."
            return (
                '{"intent": "reminder_query", "confidence": 0.95, "detected_language": "'
                + ("hi" if is_hindi else ("bn" if is_bengali else "en"))
                + f'", "spoken_response": "{resp}", "proposed_action": {{"action_type": "none", "parameters": {{}}}}}}'
            )

        return None

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> LLMResponse:
        """Simulate generation with call recording and configurable error modes."""
        eff_temperature = 0.2 if temperature is None else temperature
        eff_max_tokens = 512 if max_tokens is None else max_tokens
        eff_timeout = 10.0 if timeout_seconds is None else timeout_seconds

        call_record = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            "json_mode": json_mode,
            "temperature": eff_temperature,
            "max_tokens": eff_max_tokens,
            "timeout_seconds": eff_timeout,
        }
        self.calls.append(call_record)

        if self.simulate_connection_error:
            raise LLMConnectionError("Failed to connect to mock inference backend: connection refused")

        if self.simulate_timeout:
            # Yield control briefly then raise timeout
            await asyncio.sleep(0.01)
            raise LLMTimeoutError(f"Inference timed out after {eff_timeout} seconds")

        # Simulate minimal non-blocking asynchronous processing latency
        await asyncio.sleep(0.001)

        # Retrieve next canned response, detect unsupported personal facts, or fallback to default
        if self._canned_responses:
            content = self._canned_responses.pop(0)
        elif not self._custom_default:
            unsupported = self._detect_unsupported_personal_query(prompt, system_prompt)
            if unsupported:
                content = unsupported
            else:
                content = self._default_response
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
