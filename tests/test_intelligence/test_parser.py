"""Unit tests for the defensive ResponseParser."""

import pytest
from app.intelligence.parser import ResponseParser
from app.intelligence.schemas import ActionType, IntentType


def test_parse_valid_clean_json():
    """Verify parsing of clean, compliant JSON responses."""
    raw = """{
        "intent": "reminder_create",
        "confidence": 0.96,
        "detected_language": "en",
        "spoken_response": "I have set your medicine reminder for 8 PM.",
        "proposed_action": {
            "action_type": "create_reminder",
            "parameters": {"title": "Take blood pressure pill", "time": "20:00"},
            "requires_confirmation": false
        }
    }"""
    res = ResponseParser.parse(raw)
    assert res.intent == IntentType.REMINDER_CREATE
    assert res.confidence == 0.96
    assert res.detected_language == "en"
    assert "medicine reminder" in res.spoken_response
    assert res.proposed_action.action_type == ActionType.CREATE_REMINDER
    assert res.proposed_action.parameters["time"] == "20:00"


def test_parse_markdown_wrapped_json():
    """Verify extraction from markdown code fences (```json and ```)."""
    raw = """Here is the formatted response:
```json
{
    "intent": "weather_query",
    "confidence": 0.92,
    "detected_language": "en",
    "spoken_response": "The weather today is cloudy with light rain.",
    "proposed_action": {
        "action_type": "query_weather",
        "parameters": {"location": "Kolkata"}
    }
}
```
Hope this helps!"""
    res = ResponseParser.parse(raw)
    assert res.intent == IntentType.WEATHER_QUERY
    assert res.spoken_response == "The weather today is cloudy with light rain."
    assert res.proposed_action.action_type == ActionType.QUERY_WEATHER
    assert res.proposed_action.parameters["location"] == "Kolkata"


def test_parse_malformed_json_fails_safely():
    """Verify broken JSON syntax does not crash the server and returns safe fallback."""
    raw = '{"intent": "reminder_create", "spoken_response": "Broken JSON missing close bracket'
    res = ResponseParser.parse(raw, fallback_language="en")
    assert res.intent == IntentType.UNKNOWN
    assert res.confidence == 0.0
    assert "trouble processing" in res.spoken_response
    assert res.proposed_action.action_type == ActionType.NONE


def test_parse_empty_and_whitespace_output():
    """Verify empty model outputs return safe localized fallback."""
    res_en = ResponseParser.parse("", fallback_language="en")
    assert res_en.intent == IntentType.UNKNOWN
    assert "trouble processing" in res_en.spoken_response

    res_hi = ResponseParser.parse("   \n   ", fallback_language="hi")
    assert res_hi.intent == IntentType.UNKNOWN
    assert "समझ नहीं आई" in res_hi.spoken_response


def test_parse_missing_and_unexpected_fields():
    """Verify resilient handling of missing fields and unexpected extra keys."""
    raw = """{
        "intent": "greeting",
        "spoken_response": "Good morning! How are you?",
        "unexpected_extra_field_from_model": 12345
    }"""
    res = ResponseParser.parse(raw)
    assert res.intent == IntentType.GREETING
    assert res.spoken_response == "Good morning! How are you?"
    assert res.proposed_action.action_type == ActionType.NONE
    assert res.confidence == 0.8  # Default applied safely


def test_parse_unknown_intent_and_action_type():
    """Verify unknown or hallucinated enum values map safely to UNKNOWN / NONE."""
    raw = """{
        "intent": "hallucinated_alien_intent",
        "spoken_response": "Checking systems.",
        "proposed_action": {
            "action_type": "execute_arbitrary_shell_command",
            "parameters": {"cmd": "rm -rf"}
        }
    }"""
    res = ResponseParser.parse(raw)
    assert res.intent == IntentType.UNKNOWN
    assert res.proposed_action.action_type == ActionType.NONE
    assert res.proposed_action.parameters == {}


def test_parse_conversational_text_without_json():
    """Verify clean conversational text from model without JSON tags is parsed safely."""
    raw = "Good afternoon! It's always a pleasure to assist you. What can I do for you today?"
    res = ResponseParser.parse(raw)
    assert res.intent == IntentType.GENERAL_CHAT
    assert res.spoken_response == raw
    assert res.proposed_action.action_type == ActionType.NONE
