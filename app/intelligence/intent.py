"""Intent classification interfaces and pre-LLM routing logic."""

from typing import Optional
from app.intelligence.emergency import EmergencySignalDetector
from app.intelligence.schemas import EmergencySeverity, EmergencySignal, IntentType


def normalize_intent(raw_intent: Optional[str]) -> IntentType:
    """Safely map any raw string to an enumerated IntentType."""
    if not raw_intent:
        return IntentType.UNKNOWN

    cleaned = str(raw_intent).lower().strip()
    try:
        return IntentType(cleaned)
    except ValueError:
        return IntentType.UNKNOWN


class IntentClassifier:
    """Two-tier intent classification interface.

    Tier 1: Pre-LLM deterministic emergency signal detection.
    Tier 2: Semantic LLM intent normalization.
    """

    @classmethod
    def evaluate_emergency(cls, utterance: str) -> EmergencySignal:
        """Execute pre-LLM deterministic emergency analysis.

        Must be called BEFORE invoking any LLM inference client.
        """
        return EmergencySignalDetector.evaluate_utterance(utterance)

    @classmethod
    def check_pre_llm_intent(cls, utterance: str) -> Optional[IntentType]:
        """Check if an utterance deterministically maps to an intent without requiring LLM evaluation."""
        emergency_signal = cls.evaluate_emergency(utterance)
        if emergency_signal.detected and emergency_signal.severity in (
            EmergencySeverity.CRITICAL,
            EmergencySeverity.URGENT,
        ):
            return IntentType.EMERGENCY

        return None
