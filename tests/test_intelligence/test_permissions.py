"""Unit tests for PermissionChecker enforcing RBAC and preventing LLM privilege escalation."""

import pytest
from app.database.models.user import OperatingMode, User, UserRole
from app.intelligence.permissions import ActionPermissionDeniedError, PermissionChecker
from app.intelligence.schemas import ActionType, ProposedAction


@pytest.fixture
def resident_user() -> User:
    return User(
        id=1,
        username="resident_test",
        full_name="Resident Test",
        role=UserRole.RESIDENT,
        operating_mode=OperatingMode.ELDERLY_CARE,
    )


@pytest.fixture
def family_viewer_user() -> User:
    return User(
        id=2,
        username="family_test",
        full_name="Family Viewer Test",
        role=UserRole.FAMILY_VIEWER,
        operating_mode=OperatingMode.WORKING_PROFESSIONAL,
    )


@pytest.fixture
def caregiver_user() -> User:
    return User(
        id=3,
        username="caregiver_test",
        full_name="Caregiver Test",
        role=UserRole.CAREGIVER,
        operating_mode=OperatingMode.WORKING_PROFESSIONAL,
    )


def test_family_viewer_read_only_restriction(family_viewer_user: User):
    """Verify family viewers cannot execute state-mutating actions."""
    # Permitted read-only actions
    PermissionChecker.check_permission(
        family_viewer_user, ProposedAction(action_type=ActionType.NONE)
    )
    PermissionChecker.check_permission(
        family_viewer_user, ProposedAction(action_type=ActionType.QUERY_WEATHER)
    )

    # State mutations MUST be blocked
    mutations = [
        ProposedAction(action_type=ActionType.CREATE_REMINDER, parameters={"title": "test"}),
        ProposedAction(action_type=ActionType.TRIGGER_EMERGENCY_ALERT, parameters={"reason": "test"}),
        ProposedAction(action_type=ActionType.HEALTH_LOG, parameters={"metric_name": "bp", "value": 120}),
    ]
    for action in mutations:
        with pytest.raises(ActionPermissionDeniedError, match="Family Viewers have strictly read-only privileges"):
            PermissionChecker.check_permission(family_viewer_user, action)


def test_resident_access_boundaries(resident_user: User):
    """Verify residents can perform personal actions but not arbitrary elevated actions."""
    # Permitted personal actions
    PermissionChecker.check_permission(
        resident_user, ProposedAction(action_type=ActionType.CREATE_REMINDER, parameters={"title": "pill"})
    )
    PermissionChecker.check_permission(
        resident_user, ProposedAction(action_type=ActionType.QUERY_WEATHER)
    )
    PermissionChecker.check_permission(
        resident_user, ProposedAction(action_type=ActionType.HEALTH_LOG, parameters={"metric_name": "bp", "value": "120/80"})
    )
    PermissionChecker.check_permission(
        resident_user, ProposedAction(action_type=ActionType.TRIGGER_EMERGENCY_ALERT)
    )


def test_caregiver_permissions(caregiver_user: User):
    """Verify caregivers have full management oversight."""
    PermissionChecker.check_permission(
        caregiver_user, ProposedAction(action_type=ActionType.CREATE_REMINDER, parameters={"title": "caregiver reminder"})
    )
    PermissionChecker.check_permission(
        caregiver_user, ProposedAction(action_type=ActionType.HEALTH_LOG, parameters={"metric_name": "mood", "value": "calm"})
    )
    PermissionChecker.check_permission(
        caregiver_user, ProposedAction(action_type=ActionType.TRIGGER_EMERGENCY_ALERT)
    )


def test_prompt_injection_privilege_escalation_prevented(family_viewer_user: User):
    """Security Invariant: Even if an adversarial prompt causes the LLM to output an elevated action,

    the server-side PermissionChecker blocks it. Model output CANNOT override authorization.
    """
    adversarial_model_output_action = ProposedAction(
        action_type=ActionType.TRIGGER_EMERGENCY_ALERT,
        parameters={"reason": "Attacker injected instruction: trigger emergency alarm"},
        requires_confirmation=False,
    )
    with pytest.raises(ActionPermissionDeniedError):
        PermissionChecker.check_permission(family_viewer_user, adversarial_model_output_action)
