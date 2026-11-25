"""Permission Checker enforcing Role-Based Access Control on proposed actions."""

from app.database.models.user import User, UserRole
from app.intelligence.schemas import ActionType, ProposedAction


class ActionPermissionDeniedError(Exception):
    """Raised when an authenticated caller lacks authorization to execute a proposed action."""
    pass


class PermissionChecker:
    """Enforces server-side authorization boundaries on proposed actions.

    CRITICAL SECURITY INVARIANT:
    Model output can NEVER override server-side authorization.
    Even if an LLM claims a user has permission or proposes a sensitive action,
    this checker verifies whether the authenticated caller's actual role permits the operation.
    """

    @classmethod
    def check_permission(cls, current_user: User, action: ProposedAction) -> None:
        """Validate that the authenticated caller has permission for the proposed action.

        Raises:
            ActionPermissionDeniedError: If the caller role is not authorized.
        """
        # ActionType.NONE and QUERY_WEATHER are read-only and permitted for all active users
        if action.action_type in (ActionType.NONE, ActionType.QUERY_WEATHER):
            return

        # 1. Family Viewer restrictions: strictly read-only
        if current_user.role == UserRole.FAMILY_VIEWER:
            raise ActionPermissionDeniedError(
                f"Access denied: Family Viewers have strictly read-only privileges and cannot "
                f"execute state-mutating action '{action.action_type.value}'"
            )

        # 2. Resident access boundaries:
        # Residents are permitted personal home actions (create_reminder, health_log, emergency alert)
        if current_user.role == UserRole.RESIDENT:
            allowed_resident_actions = {
                ActionType.CREATE_REMINDER,
                ActionType.QUERY_WEATHER,
                ActionType.HEALTH_LOG,
                ActionType.TRIGGER_EMERGENCY_ALERT,
            }
            if action.action_type not in allowed_resident_actions:
                raise ActionPermissionDeniedError(
                    f"Access denied: Resident role is not authorized to execute action '{action.action_type.value}'"
                )
            return

        # 3. Caregiver permissions: Caregivers have full management oversight
        if current_user.role == UserRole.CAREGIVER:
            return

        # Any unhandled role
        raise ActionPermissionDeniedError(
            f"Access denied: Role '{current_user.role}' is not authorized to execute '{action.action_type.value}'"
        )
