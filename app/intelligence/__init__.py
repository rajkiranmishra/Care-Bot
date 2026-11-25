"""CARE-BOT Intelligence Engine: Local LLMs, decision layers, and intent orchestration."""

from app.intelligence.emergency import EmergencySignalDetector
from app.intelligence.intent import IntentClassifier, normalize_intent
from app.intelligence.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LocalLLMClient,
    MockLLMClient,
)
from app.intelligence.parser import ResponseParser
from app.intelligence.permissions import ActionPermissionDeniedError, PermissionChecker
from app.intelligence.prompt_builder import PromptBuilder
from app.intelligence.schemas import (
    ActionType,
    EmergencySeverity,
    EmergencySignal,
    IntentType,
    LLMResponse,
    ModelMetadata,
    ParsedLLMResponse,
    ProposedAction,
)
from app.intelligence.validator import ActionValidationError, ActionValidator

__all__ = [
    "LocalLLMClient",
    "MockLLMClient",
    "LLMClientError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "IntentType",
    "ActionType",
    "EmergencySeverity",
    "EmergencySignal",
    "ProposedAction",
    "ParsedLLMResponse",
    "LLMResponse",
    "ModelMetadata",
    "EmergencySignalDetector",
    "PromptBuilder",
    "ResponseParser",
    "IntentClassifier",
    "normalize_intent",
    "ActionValidator",
    "ActionValidationError",
    "PermissionChecker",
    "ActionPermissionDeniedError",
]
