"""Action Validator enforcing whitelist integrity and parameter schema bounds."""

from typing import Any, Dict
from app.intelligence.schemas import ActionType, ProposedAction


class ActionValidationError(Exception):
    """Raised when an action proposal violates whitelist rules or parameter constraints."""
    pass


class ActionValidator:
    """Enforces the inviolable rule: The LLM is strictly an advisor, never an executor.

    Validates that proposed actions adhere to strict whitelists and safe parameter bounds.
    Rejects arbitrary tool invocations, unknown action names, and out-of-bounds parameters.
    """

    @classmethod
    def validate_action(cls, proposed_action: ProposedAction) -> ProposedAction:
        """Validate action type and parameter schemas.

        Raises:
            ActionValidationError: If the action type is unknown or parameters fail schema bounds.
        """
        if not isinstance(proposed_action.action_type, ActionType):
            raise ActionValidationError(
                f"Security violation: Action type '{proposed_action.action_type}' is not recognized. "
                f"Only whitelisted actions {[a.value for a in ActionType]} are permitted."
            )

        action_type = proposed_action.action_type
        params = proposed_action.parameters or {}

        if action_type == ActionType.NONE:
            return proposed_action

        elif action_type == ActionType.CREATE_REMINDER:
            title = params.get("title")
            if not title or not isinstance(title, str) or not title.strip():
                raise ActionValidationError("create_reminder action requires a non-empty string parameter 'title'")
            if len(title) > 128:
                raise ActionValidationError("create_reminder parameter 'title' exceeds maximum length of 128 characters")

            # Check for time/due parameter
            time_param = params.get("time") or params.get("reminder_time")
            if time_param and (not isinstance(time_param, str) or len(time_param) > 64):
                raise ActionValidationError("create_reminder parameter 'time' must be a valid string under 64 characters")

        elif action_type == ActionType.QUERY_WEATHER:
            location = params.get("location")
            if location is not None:
                if not isinstance(location, str) or len(location) > 64:
                    raise ActionValidationError("query_weather parameter 'location' must be a string under 64 characters")

        elif action_type == ActionType.TRIGGER_EMERGENCY_ALERT:
            # Triggering emergency alerts is a critical safety operation.
            # In accordance with safety rules, it always requires explicit confirmation unless
            # pre-validated by the deterministic emergency layer.
            reason = params.get("reason", "User requested emergency alert")
            if not isinstance(reason, str) or len(reason) > 256:
                raise ActionValidationError("trigger_emergency_alert parameter 'reason' must be a string under 256 characters")

        elif action_type == ActionType.HEALTH_LOG:
            metric_name = params.get("metric_name")
            if not metric_name or not isinstance(metric_name, str) or not metric_name.strip():
                raise ActionValidationError("health_log action requires a non-empty string parameter 'metric_name'")
            if len(metric_name) > 64:
                raise ActionValidationError("health_log parameter 'metric_name' exceeds maximum length of 64 characters")

            if "value" not in params:
                raise ActionValidationError("health_log action requires a 'value' parameter")

        return proposed_action
