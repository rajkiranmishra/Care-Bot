"""CARE-BOT Intelligence Engine: Local LLMs, decision layers, and intent orchestration."""

from app.intelligence.emergency import EmergencySignalDetector
from app.intelligence.engine import ConversationEngine
from app.intelligence.intent import IntentClassifier, normalize_intent
from app.intelligence.llm_client import (
    LLMClientError,
    LLMConnectionError,
    LLMTimeoutError,
    LocalLLMClient,
    MockLLMClient,
)
from app.intelligence.ollama_client import OllamaLLMClient
from app.intelligence.parser import ResponseParser
from app.intelligence.permissions import ActionPermissionDeniedError, PermissionChecker
from app.intelligence.prompt_builder import PromptBuilder
from app.intelligence.schemas import (
    ActionType,
    ChatTurnRequest,
    ChatTurnResponse,
    EmergencySeverity,
    EmergencySignal,
    IntentType,
    LLMResponse,
    ModelMetadata,
    ParsedLLMResponse,
    ProposedAction,
    WelfareEscalationPolicy,
)
from app.intelligence.validator import ActionValidationError, ActionValidator

__all__ = [
    "ConversationEngine",
    "ChatTurnRequest",
    "ChatTurnResponse",
    "LocalLLMClient",
    "MockLLMClient",
    "OllamaLLMClient",
    "LLMClientError",
    "LLMTimeoutError",
    "LLMConnectionError",
    "IntentType",
    "ActionType",
    "EmergencySeverity",
    "EmergencySignal",
    "WelfareEscalationPolicy",
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
