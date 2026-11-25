"""Unit tests for ActionValidator enforcing strict whitelist and parameter boundaries."""

import pytest
from app.intelligence.schemas import ActionType, ProposedAction
from app.intelligence.validator import ActionValidationError, ActionValidator


def test_validator_accepts_valid_actions():
    """Verify whitelisted actions with conformant parameters pass validation."""
    actions = [
        ProposedAction(action_type=ActionType.NONE),
        ProposedAction(
            action_type=ActionType.CREATE_REMINDER,
            parameters={"title": "Take evening pills", "time": "19:00"},
        ),
        ProposedAction(
            action_type=ActionType.QUERY_WEATHER,
            parameters={"location": "Bangalore"},
        ),
        ProposedAction(
            action_type=ActionType.HEALTH_LOG,
            parameters={"metric_name": "blood_pressure", "value": "120/80"},
        ),
        ProposedAction(
            action_type=ActionType.TRIGGER_EMERGENCY_ALERT,
            parameters={"reason": "Manual emergency button press"},
            requires_confirmation=True,
        ),
    ]
    for action in actions:
        validated = ActionValidator.validate_action(action)
        assert validated.action_type == action.action_type


def test_validator_rejects_missing_reminder_title():
    """Verify reminder creation without title is rejected."""
    action = ProposedAction(
        action_type=ActionType.CREATE_REMINDER,
        parameters={"time": "20:00"},
    )
    with pytest.raises(ActionValidationError, match="requires a non-empty string parameter 'title'"):
        ActionValidator.validate_action(action)


def test_validator_rejects_oversized_parameters():
    """Verify parameters exceeding string limits are rejected."""
    long_title = "A" * 150
    action = ProposedAction(
        action_type=ActionType.CREATE_REMINDER,
        parameters={"title": long_title},
    )
    with pytest.raises(ActionValidationError, match="exceeds maximum length"):
        ActionValidator.validate_action(action)


def test_validator_rejects_missing_health_log_parameters():
    """Verify health log requires metric_name and value."""
    action_no_val = ProposedAction(
        action_type=ActionType.HEALTH_LOG,
        parameters={"metric_name": "pulse"},
    )
    with pytest.raises(ActionValidationError, match="requires a 'value' parameter"):
        ActionValidator.validate_action(action_no_val)


def test_validator_rejects_invalid_action_type_objects():
    """Verify non-whitelisted arbitrary strings are strictly rejected."""
    class FakeAction:
        action_type = "arbitrary_custom_tool_run"
        parameters = {}

    with pytest.raises(ActionValidationError, match="Security violation"):
        ActionValidator.validate_action(FakeAction())  # type: ignore
