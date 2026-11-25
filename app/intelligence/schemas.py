"""Pydantic data schemas for CARE-BOT intelligence, intents, and action proposals."""

import enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class IntentType(str, enum.Enum):
    """Categorized user intent resolved by deterministic logic or semantic model."""
    EMERGENCY = "emergency"
    GREETING = "greeting"
    REMINDER_QUERY = "reminder_query"
    REMINDER_CREATE = "reminder_create"
    WEATHER_QUERY = "weather_query"
    HEALTH_CHECKIN = "health_checkin"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"


class ActionType(str, enum.Enum):
    """Strict whitelist of valid application actions the intelligence engine may propose.

    SECURITY RULE: The LLM may only propose actions from this enumerated whitelist.
    Arbitrary tool execution and dynamically generated action names are strictly prohibited.
    """
    NONE = "none"
    CREATE_REMINDER = "create_reminder"
    QUERY_WEATHER = "query_weather"
    TRIGGER_EMERGENCY_ALERT = "trigger_emergency_alert"
    HEALTH_LOG = "health_log"


class EmergencySeverity(str, enum.Enum):
    """Severity levels for detected emergency signals.

    Distinguishes NORMAL (safe/routine), UNCERTAIN (ambiguous distress requiring check),
    URGENT (high concern needing prompt assistance), and CRITICAL (immediate danger/trauma).
    """
    CRITICAL = "critical"      # Immediate danger: physical fall, loss of consciousness, acute chest pain
    URGENT = "urgent"          # High concern: acute dizziness, bleeding, unable to rise from bed
    UNCERTAIN = "uncertain"    # Ambiguous or muffled distress; requires active confirmation check
    NORMAL = "normal"          # Standard non-distress state
    NONE = "none"              # Backward-compatible alias for NORMAL


class WelfareEscalationPolicy(BaseModel):
    """Configurable parameters governing welfare checks and silence escalation.

    BEHAVIOR RULES:
    1. unanswered_threshold:
       Number of consecutive unanswered welfare prompts required before escalating beyond UNCERTAIN.
       While unanswered_checks < unanswered_threshold, state remains UNCERTAIN with repeat prompts.
    2. auto_escalate_to_critical:
       - If True: When unanswered_checks reaches unanswered_threshold, escalates to CRITICAL severity.
       - If False: When unanswered_checks reaches unanswered_threshold, escalates to URGENT severity
         (alerting caregivers for follow-up without triggering immediate critical siren/dispatch).
    3. user_vulnerability_multiplier:
       Scales response window (effective_timeout = welfare_timeout_seconds / user_vulnerability_multiplier).
       Higher values for vulnerable residents produce faster escalation.
    """
    welfare_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Base seconds to wait for user voice response to a welfare check",
    )
    unanswered_threshold: int = Field(
        default=2,
        ge=1,
        description="Number of unanswered welfare prompts before escalating from UNCERTAIN to URGENT or CRITICAL",
    )
    auto_escalate_to_critical: bool = Field(
        default=False,
        description="If True, silence exceeding unanswered_threshold escalates to CRITICAL; if False, escalates to URGENT",
    )
    user_vulnerability_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Multiplier adjusting timeout sensitivity for high-risk residents (e.g. 1.5 for faster escalation)",
    )

    model_config = ConfigDict(frozen=True)


class EmergencySignal(BaseModel):
    """Result of deterministic pre-LLM emergency signal evaluation.

    ARCHITECTURAL INVARIANT:
    EmergencySignal represents pure diagnostic signal identification.
    It does NOT initiate network notifications or contact emergency services.
    External side-effects are strictly deferred to a separately authorized service.
    """
    detected: bool = Field(default=False, description="Whether an emergency pattern was recognized")
    severity: EmergencySeverity = Field(default=EmergencySeverity.NORMAL)
    trigger_phrase: Optional[str] = Field(default=None, description="Matched keyword or utterance phrase")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="Diagnostic rationale for the detection")
    requires_immediate_escalation: bool = Field(
        default=False,
        description="True for CRITICAL/URGENT cases where authorized service should alert caregivers",
    )
    requires_confirmation: bool = Field(
        default=False,
        description="True for UNCERTAIN cases where assistant must explicitly ask if user is OK",
    )
    is_false_positive: bool = Field(
        default=False,
        description="Flagged true if conversational negation or recreational context detected",
    )
    is_safe_confirmed: bool = Field(
        default=False,
        description="True ONLY if the user has explicitly and affirmatively stated they are safe/okay",
    )
    user_affirmation: Optional[str] = Field(
        default=None,
        description="User's verbal affirmative response (e.g. 'I am fine'), recorded for review",
    )
    pending_resolution: bool = Field(
        default=False,
        description="True when an affirmative statement was spoken during an active CRITICAL emergency, requiring authorized caregiver clearance",
    )
    repeat_count: int = Field(
        default=1,
        ge=1,
        description="Consecutive repetitions of distress triggers within active window",
    )
    unanswered_checks: int = Field(
        default=0,
        ge=0,
        description="Count of consecutive unanswered welfare prompts",
    )

    model_config = ConfigDict(frozen=True)


class ProposedAction(BaseModel):
    """Structured action proposed by the intelligence layer.

    SECURITY RULE: An action proposal is merely a request to the host system.
    It does not execute until validated by ActionValidator and authorized by PermissionChecker.
    """
    action_type: ActionType = Field(default=ActionType.NONE)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    requires_confirmation: bool = Field(
        default=False,
        description="Whether user voice confirmation is required before execution",
    )

    model_config = ConfigDict(frozen=True)


class ParsedLLMResponse(BaseModel):
    """Normalized, strictly validated intelligence output structure."""
    intent: IntentType = Field(default=IntentType.GENERAL_CHAT)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    detected_language: Optional[str] = Field(default=None, max_length=8)
    spoken_response: str = Field(..., min_length=1, description="Synthesized utterance to be spoken by assistant")
    proposed_action: ProposedAction = Field(default_factory=ProposedAction)
    raw_text: Optional[str] = Field(default=None, description="Original unparsed model text for diagnostics")

    model_config = ConfigDict(frozen=True)


class LLMResponse(BaseModel):
    """Raw inference result from a local LLM client."""
    content: str = Field(..., description="Raw text generated by the model")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class ModelMetadata(BaseModel):
    """Metadata describing a loaded or configured local inference model."""
    model_name: str
    provider: str
    context_window: int = Field(default=2048, ge=256)
    quantization: Optional[str] = None

    model_config = ConfigDict(frozen=True)
