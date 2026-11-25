"""Defensive Response Parser for Local LLM Outputs."""

import json
import logging
import re
from typing import Any, Dict, Optional
from pydantic import ValidationError
from app.intelligence.schemas import (
    ActionType,
    IntentType,
    ParsedLLMResponse,
    ProposedAction,
)

logger = logging.getLogger("carebot.intelligence.parser")


class ResponseParser:
    """Defensive, fail-safe parser converting unconstrained LLM strings into strict Pydantic schemas.

    Architectural Invariant:
    A malformed or adversarial LLM output MUST NEVER crash the backend.
    All outputs are sanitized, extracted, validated, and safely defaulted if unparseable.
    """

    @classmethod
    def extract_json_candidate(cls, raw_text: str) -> Optional[str]:
        """Extract a JSON object substring from raw text, stripping markdown codeblocks and chatter."""
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # 1. Check for markdown codeblocks: ```json ... ``` or ``` ... ```
        codeblock_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
        if codeblock_match:
            return codeblock_match.group(1).strip()

        # 2. Heuristic: locate the outermost balanced { ... } brackets
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1].strip()

        return None

    @classmethod
    def parse(
        cls,
        raw_output: str,
        fallback_language: str = "en",
    ) -> ParsedLLMResponse:
        """Parse raw LLM output into a strict ParsedLLMResponse.

        Never raises an exception; falls back to a safe conversational response on any failure.
        """
        safe_fallback_text = {
            "hi": "मुझे आपकी बात पूरी तरह समझ नहीं आई। क्या आप दोबारा कह सकते हैं?",
            "bn": "আমি আপনার কথাটি ঠিক বুঝতে পারিনি। আপনি কি আরেকবার বলবেন?",
            "en": "I'm sorry, I had trouble processing that. Could you please repeat what you said?",
        }.get(fallback_language, "I'm sorry, I had trouble processing that. Could you please repeat?")

        if not raw_output or not raw_output.strip():
            logger.warning("Empty response received from LLM; activating safe fallback.")
            return ParsedLLMResponse(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                detected_language=fallback_language,
                spoken_response=safe_fallback_text,
                proposed_action=ProposedAction(action_type=ActionType.NONE),
                raw_text=raw_output,
            )

        json_candidate = cls.extract_json_candidate(raw_output)

        if not json_candidate:
            logger.warning("No JSON structure found in LLM output: %r", raw_output[:120])
            # If the raw text is clean conversational prose without JSON, use it directly as spoken response
            # provided it doesn't look like code or instructions.
            cleaned_text = raw_output.strip()
            if len(cleaned_text) < 500 and not cleaned_text.startswith(("{", "<", "[")):
                return ParsedLLMResponse(
                    intent=IntentType.GENERAL_CHAT,
                    confidence=0.5,
                    detected_language=fallback_language,
                    spoken_response=cleaned_text,
                    proposed_action=ProposedAction(action_type=ActionType.NONE),
                    raw_text=raw_output,
                )

            return ParsedLLMResponse(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                detected_language=fallback_language,
                spoken_response=safe_fallback_text,
                proposed_action=ProposedAction(action_type=ActionType.NONE),
                raw_text=raw_output,
            )

        try:
            data = json.loads(json_candidate)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON root must be an object/dict")
        except Exception as exc:
            logger.warning("JSON decoding failed for candidate string: %s", exc)
            return ParsedLLMResponse(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                detected_language=fallback_language,
                spoken_response=safe_fallback_text,
                proposed_action=ProposedAction(action_type=ActionType.NONE),
                raw_text=raw_output,
            )

        # Normalize intent field
        raw_intent = str(data.get("intent", "general_chat")).lower().strip()
        try:
            intent = IntentType(raw_intent)
        except ValueError:
            logger.info("Unrecognized intent '%s'; falling back to 'unknown'", raw_intent)
            intent = IntentType.UNKNOWN

        # Normalize proposed action
        raw_action_data = data.get("proposed_action")
        proposed_action = ProposedAction(action_type=ActionType.NONE)

        if isinstance(raw_action_data, dict):
            raw_action_type = str(raw_action_data.get("action_type", "none")).lower().strip()
            try:
                action_type = ActionType(raw_action_type)
                parameters = raw_action_data.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                requires_conf = bool(raw_action_data.get("requires_confirmation", False))
                proposed_action = ProposedAction(
                    action_type=action_type,
                    parameters=parameters,
                    requires_confirmation=requires_conf,
                )
            except ValueError:
                logger.warning("Unrecognized action_type '%s' rejected; defaulted to none", raw_action_type)
                proposed_action = ProposedAction(action_type=ActionType.NONE)

        spoken_response = str(data.get("spoken_response", "")).strip()
        if not spoken_response:
            spoken_response = safe_fallback_text

        # Clamp confidence to valid 0.0 - 1.0 range
        try:
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.5

        detected_lang = data.get("detected_language")
        if detected_lang and isinstance(detected_lang, str):
            detected_lang = detected_lang.lower().strip()[:8]
        else:
            detected_lang = fallback_language

        return ParsedLLMResponse(
            intent=intent,
            confidence=confidence,
            detected_language=detected_lang,
            spoken_response=spoken_response,
            proposed_action=proposed_action,
            raw_text=raw_output,
        )
