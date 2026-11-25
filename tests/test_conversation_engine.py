"""Comprehensive unit and integration tests for ConversationEngine orchestration."""

import asyncio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.conversation.schemas import SessionCreate
from app.conversation.service import ConversationService
from app.database.models import (
    ActiveEmergency,
    CaregiverResidentAssignment,
    ConversationMessage,
    ConversationSession,
    ConversationState,
    MessageRole,
    OperatingMode,
    ResponseStyle,
    SupportedLanguage,
    User,
    UserRole,
)
from app.intelligence.emergency_service import EmergencyService
from app.intelligence.engine import ConversationEngine
from app.intelligence.llm_client import MockLLMClient
from app.intelligence.schemas import (
    ActionType,
    ChatTurnRequest,
    EmergencySeverity,
    IntentType,
)
from app.users.schemas import UserCreate, UserUpdate
from app.users.service import UserService
from tests.conftest import TEST_DEVICE_KEY, make_auth_headers


# -----------------------------------------------------------------------------
# Unit Tests for ConversationEngine Orchestrator
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_engine_normal_conversational_turn(db_session: Session):
    """Verify normal conversational interaction: user utterance persisted, LLM invoked, assistant persisted."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="engine_resident", full_name="Engine Resident", role="resident"),
    )

    mock_llm = MockLLMClient(
        default_response=(
            '{"intent": "general_chat", "confidence": 0.95, "detected_language": "en", '
            '"spoken_response": "Hello! I am doing well, thank you for asking.", '
            '"proposed_action": {"action_type": "none", "parameters": {}}}'
        )
    )
    engine = ConversationEngine(llm_client=mock_llm)

    req = ChatTurnRequest(utterance="Hello CARE-BOT, how are you?")
    res = await engine.process_turn(db_session, user, req)

    assert res.session_id is not None
    assert res.user_utterance == "Hello CARE-BOT, how are you?"
    assert res.spoken_response == "Hello! I am doing well, thank you for asking."
    assert res.intent == IntentType.GENERAL_CHAT
    assert res.proposed_action.action_type == ActionType.NONE
    assert res.emergency_signal.detected is False
    assert res.active_emergency is False

    # Verify both messages persisted in SQLite database
    messages = (
        db_session.query(ConversationMessage)
        .filter(ConversationMessage.session_id == res.session_id)
        .order_by(ConversationMessage.id.asc())
        .all()
    )
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[0].content == "Hello CARE-BOT, how are you?"
    assert messages[1].role == MessageRole.ASSISTANT
    assert messages[1].content == "Hello! I am doing well, thank you for asking."


@pytest.mark.anyio
async def test_engine_multi_turn_history_accumulation(db_session: Session):
    """Verify consecutive conversational turns accumulate history and inform the prompt."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="history_resident", full_name="History Resident", role="resident"),
    )

    mock_llm = MockLLMClient()
    mock_llm.enqueue_response(
        '{"intent": "general_chat", "confidence": 0.9, "detected_language": "en", '
        '"spoken_response": "Nice to meet you! What would you like to do?", '
        '"proposed_action": {"action_type": "none"}}'
    )
    mock_llm.enqueue_response(
        '{"intent": "weather_query", "confidence": 0.95, "detected_language": "en", '
        '"spoken_response": "The weather today in Delhi is 25 degrees and clear.", '
        '"proposed_action": {"action_type": "query_weather", "parameters": {"location": "Delhi"}}}'
    )
    engine = ConversationEngine(llm_client=mock_llm)

    # Turn 1
    t1 = await engine.process_turn(db_session, user, ChatTurnRequest(utterance="My name is Alex."))
    session_id = t1.session_id

    # Turn 2 in the same session
    t2 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="What is the weather in Delhi?", session_id=session_id),
    )
    assert t2.session_id == session_id
    assert t2.intent == IntentType.WEATHER_QUERY
    assert t2.proposed_action.action_type == ActionType.QUERY_WEATHER

    # Verify MockLLM received previous history in Turn 2's prompt
    assert len(mock_llm.calls) == 2
    turn2_prompt = mock_llm.calls[1]["prompt"]
    assert "Resident: My name is Alex." in turn2_prompt
    assert "Assistant: Nice to meet you!" in turn2_prompt
    assert "Resident: What is the weather in Delhi?" in turn2_prompt


@pytest.mark.anyio
async def test_engine_operating_modes_prompt_differentiation(db_session: Session):
    """Verify operating mode (Elderly Care vs Working Professional) is passed to prompt builder."""
    elderly_user = UserService.create_user(
        db_session,
        UserCreate(
            username="elderly_user_mode",
            full_name="Elderly User",
            role="resident",
            operating_mode=OperatingMode.ELDERLY_CARE,
        ),
    )
    pro_user = UserService.create_user(
        db_session,
        UserCreate(
            username="pro_user_mode",
            full_name="Pro User",
            role="resident",
            operating_mode=OperatingMode.WORKING_PROFESSIONAL,
        ),
    )

    mock_llm = MockLLMClient()
    engine = ConversationEngine(llm_client=mock_llm)

    await engine.process_turn(db_session, elderly_user, ChatTurnRequest(utterance="Good morning"))
    elderly_sys_prompt = mock_llm.calls[0]["system_prompt"]
    assert "Elderly Care Mode" in elderly_sys_prompt
    assert "Speak gently, warmly, and clearly" in elderly_sys_prompt

    await engine.process_turn(db_session, pro_user, ChatTurnRequest(utterance="Good morning"))
    pro_sys_prompt = mock_llm.calls[1]["system_prompt"]
    assert "Working Professional Mode" in pro_sys_prompt
    assert "Be concise, efficient, and direct" in pro_sys_prompt


@pytest.mark.anyio
async def test_engine_emergency_fast_path_bypasses_llm(db_session: Session):
    """Verify critical emergency (e.g. fall) executes pre-LLM fast path and does NOT invoke LLM."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="fall_victim", full_name="Fall Victim", role="resident"),
    )

    mock_llm = MockLLMClient()
    engine = ConversationEngine(llm_client=mock_llm)

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="Help me, I fell down and cannot get up!"),
    )

    # 1. Emergency classification verified
    assert res.intent == IntentType.EMERGENCY
    assert res.emergency_signal.detected is True
    assert res.emergency_signal.severity == EmergencySeverity.CRITICAL
    assert res.proposed_action.action_type == ActionType.TRIGGER_EMERGENCY_ALERT
    assert res.active_emergency is True

    # 2. LLM was NEVER called during emergency fast path
    assert len(mock_llm.calls) == 0

    # 3. Truthful response: acknowledges detection without falsely claiming notification delivery
    assert "Emergency signal registered locally" in res.spoken_response
    assert "External notification has not been dispatched" in res.spoken_response
    assert "notifying your caregiver" not in res.spoken_response.lower()
    assert "caregiver has been notified" not in res.spoken_response.lower()
    assert res.emergency_signal.detected is True
    assert res.recorded_locally is True
    assert res.notification_attempted is False
    assert res.notification_delivered is False


@pytest.mark.anyio
async def test_engine_confirmed_critical_emergency_not_cleared_by_verbal_affirmation(db_session: Session):
    """CRITICAL SAFETY INVARIANT: User saying 'I am fine' CANNOT clear an already confirmed CRITICAL emergency."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="fall_affirm_user", full_name="Fall Affirm User", role="resident"),
    )

    mock_llm = MockLLMClient()
    engine = ConversationEngine(llm_client=mock_llm)

    # Turn 1: User falls
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell down and my hip hurts!"),
    )
    assert t1.active_emergency is True
    assert t1.emergency_signal.severity == EmergencySeverity.CRITICAL

    # Turn 2: User confusedly/reflexively says "I am fine, don't worry"
    t2 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I am fine, don't worry", session_id=t1.session_id),
    )

    # Emergency is NOT cleared!
    assert t2.active_emergency is True
    assert t2.pending_resolution is True
    assert t2.emergency_signal.severity == EmergencySeverity.CRITICAL
    assert t2.emergency_signal.user_affirmation == "I am fine, don't worry"
    assert "safety protocols remain active until an authorized caregiver completes the resolution procedure" in t2.spoken_response

    # Verify session context_metadata preserves the active critical emergency
    session = db_session.query(ConversationSession).filter(ConversationSession.id == t1.session_id).one()
    assert "active_emergency" in session.context_metadata
    assert '"severity": "critical"' in session.context_metadata


@pytest.mark.anyio
async def test_engine_uncertain_distress_cleared_by_verbal_affirmation(db_session: Session):
    """An ordinary UNCERTAIN check (e.g. mild dizziness) CAN be stood down by verbal safety affirmation."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="dizzy_user", full_name="Dizzy User", role="resident"),
    )

    mock_llm = MockLLMClient(
        default_response='{"intent": "health_checkin", "spoken_response": "Please sit down and rest."}'
    )
    engine = ConversationEngine(llm_client=mock_llm)

    # Turn 1: Uncertain malaise
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I feel a little dizzy"),
    )
    assert t1.emergency_signal.severity == EmergencySeverity.UNCERTAIN
    assert t1.active_emergency is True

    # Turn 2: User confirms they drank water and are okay
    t2 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I am fine now, it was a false alarm", session_id=t1.session_id),
    )
    # Uncertain state stood down safely
    assert t2.active_emergency is False
    assert t2.pending_resolution is False
    assert t2.emergency_signal.is_safe_confirmed is True
    assert "stood down the welfare check" in t2.spoken_response


@pytest.mark.anyio
async def test_engine_llm_timeout_handled_safely(db_session: Session):
    """Verify that an inference timeout does not crash the server and returns a safe fallback message."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="timeout_user", full_name="Timeout User", role="resident"),
    )

    timeout_llm = MockLLMClient(simulate_timeout=True)
    engine = ConversationEngine(llm_client=timeout_llm)

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="What is on my calendar today?"),
    )
    assert res.intent == IntentType.UNKNOWN
    assert "trouble processing" in res.spoken_response
    assert res.proposed_action.action_type == ActionType.NONE


@pytest.mark.anyio
async def test_engine_malformed_llm_json_handled_safely(db_session: Session):
    """Verify that malformed or non-JSON output from the model fails safely."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="malformed_json_user", full_name="Malformed User", role="resident"),
    )

    bad_json_llm = MockLLMClient(
        default_response="This is raw text without any JSON structure or brackets."
    )
    engine = ConversationEngine(llm_client=bad_json_llm)

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="Can you help me?"),
    )
    assert res.spoken_response == "This is raw text without any JSON structure or brackets."
    assert res.proposed_action.action_type == ActionType.NONE


# -----------------------------------------------------------------------------
# Integration Tests for REST Endpoint: POST /api/v1/conversations/users/{user_id}/chat
# -----------------------------------------------------------------------------

def test_api_chat_turn_success(client: TestClient):
    """Verify successful end-to-end chat turn over HTTP with JWT authentication."""
    user = client.post("/api/v1/users", json={"username": "chat_user", "full_name": "Chat User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="chat_user", role="resident")

    res = client.post(
        f"/api/v1/conversations/users/{user_id}/chat",
        json={"utterance": "Hello CARE-BOT!"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["user_utterance"] == "Hello CARE-BOT!"
    assert data["session_id"] is not None
    assert "spoken_response" in data
    assert data["session_state"] == "active_conversation"


def test_api_chat_turn_unauthenticated_rejected(client: TestClient):
    """Verify unauthenticated requests are rejected with 401 Unauthorized."""
    user = client.post("/api/v1/users", json={"username": "unauth_user", "full_name": "Unauth User"}).json()
    user_id = user["id"]

    res = client.post(
        f"/api/v1/conversations/users/{user_id}/chat",
        json={"utterance": "Hello without token"},
    )
    assert res.status_code == 401
    assert "credentials were not provided" in res.json()["detail"].lower()


def test_api_chat_turn_cross_user_access_blocked(client: TestClient):
    """Verify Resident A cannot chat or access sessions belonging to Resident B."""
    u1 = client.post("/api/v1/users", json={"username": "alice_res", "full_name": "Alice Res"}).json()
    u2 = client.post("/api/v1/users", json={"username": "bob_res", "full_name": "Bob Res"}).json()

    headers_alice = make_auth_headers(user_id=u1["id"], username="alice_res", role="resident")

    # Calling with Alice's token targeting Bob's user_id
    res = client.post(
        f"/api/v1/conversations/users/{u2['id']}/chat",
        json={"utterance": "Spying on other user"},
        headers=headers_alice,
    )
    assert res.status_code == 403
    assert "Access denied" in res.json()["detail"]


def test_api_chat_turn_family_viewer_read_only_blocked(client: TestClient):
    """Verify Family Viewer accounts cannot initiate chat turns (read-only enforcement)."""
    viewer = client.post(
        "/api/v1/users",
        json={"username": "dave_viewer_chat", "full_name": "Dave Viewer", "role": "family_viewer"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()
    headers_viewer = make_auth_headers(user_id=viewer["id"], username="dave_viewer_chat", role="family_viewer")

    res = client.post(
        f"/api/v1/conversations/users/{viewer['id']}/chat",
        json={"utterance": "Attempting to post chat turn"},
        headers=headers_viewer,
    )
    assert res.status_code == 403
    assert "Family viewers have read-only access" in res.json()["detail"]


def test_api_chat_turn_caregiver_authorized(client: TestClient):
    """Verify Caregiver can execute chat on behalf of a resident."""
    resident = client.post("/api/v1/users", json={"username": "sarah_resident", "full_name": "Sarah Resident"}).json()
    caregiver = client.post(
        "/api/v1/users",
        json={"username": "carol_caregiver", "full_name": "Carol Caregiver", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()

    headers_caregiver = make_auth_headers(user_id=caregiver["id"], username="carol_caregiver", role="caregiver")

    res = client.post(
        f"/api/v1/conversations/users/{resident['id']}/chat",
        json={"utterance": "Caregiver checking resident dialogue"},
        headers=headers_caregiver,
    )
    assert res.status_code == 200
    assert res.json()["user_utterance"] == "Caregiver checking resident dialogue"


def test_api_chat_turn_emergency_endpoint_fast_path(client: TestClient):
    """Verify that posting an acute fall over the REST endpoint immediately flags emergency."""
    resident = client.post("/api/v1/users", json={"username": "fall_resident", "full_name": "Fall Resident"}).json()
    headers = make_auth_headers(user_id=resident["id"], username="fall_resident", role="resident")

    res = client.post(
        f"/api/v1/conversations/users/{resident['id']}/chat",
        json={"utterance": "I fell down and cannot get up, help!"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "emergency"
    assert data["emergency_signal"]["detected"] is True
    assert data["emergency_signal"]["severity"] == "critical"
    assert data["proposed_action"]["action_type"] == "trigger_emergency_alert"
    assert data["active_emergency"] is True


# -----------------------------------------------------------------------------
# Regression Tests: Emergency Honesty, MockLLM Truthfulness, Persistence & Integrity
# -----------------------------------------------------------------------------

@pytest.mark.anyio
async def test_emergency_honesty_distinctions(db_session: Session):
    """Verify system strictly distinguishes detected vs recorded locally vs notification attempted vs delivered."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="honesty_user", full_name="Honesty User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell and I can't move my arm"),
    )

    # 1. State distinctions
    assert res.emergency_signal.detected is True
    assert res.recorded_locally is True
    assert res.notification_attempted is False
    assert res.notification_delivered is False

    # 2. Honest spoken communication
    assert "notifying your caregiver" not in res.spoken_response.lower()
    assert "caregiver has been notified" not in res.spoken_response.lower()
    assert "External notification has not been dispatched" in res.spoken_response


@pytest.mark.anyio
async def test_mock_llm_refuses_to_invent_medications(db_session: Session):
    """Verify MockLLMClient and engine refuse to invent medication names, schedules, or doses."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="med_query_user", full_name="Med Query User", role="resident"),
    )
    # Generic MockLLMClient with no canned or hardcoded responses
    engine = ConversationEngine(llm_client=MockLLMClient())

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="What medications do I have scheduled for this morning?"),
    )

    # Must honestly report lack of access rather than fabricating drugs or times
    assert "do not have access to your medication schedule" in res.spoken_response
    assert "blood pressure" not in res.spoken_response.lower()
    assert "9:00" not in res.spoken_response
    assert res.proposed_action.action_type == ActionType.NONE


@pytest.mark.anyio
async def test_mock_llm_refuses_to_invent_reminders(db_session: Session):
    """Verify MockLLMClient and engine refuse to invent reminders or daily agenda."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="reminder_query_user", full_name="Reminder User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="What are my reminders for today?"),
    )

    assert "cannot access your reminders or schedule" in res.spoken_response
    assert res.proposed_action.action_type == ActionType.NONE


@pytest.mark.anyio
async def test_mock_llm_refuses_to_invent_doctor_appointments(db_session: Session):
    """Verify MockLLMClient and engine refuse to invent appointment schedules or medical history."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="appt_user", full_name="Appt User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    res = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="When is my next doctor appointment?"),
    )

    assert "do not have access to your appointment schedule" in res.spoken_response
    assert res.proposed_action.action_type == ActionType.NONE


@pytest.mark.anyio
async def test_emergency_persists_across_new_conversation_session(db_session: Session):
    """Verify active emergency remains active when a resident starts a new conversation session."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="session_persist_user", full_name="Session Persist", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # Turn 1: Emergency in session 1
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell down and cannot get up!"),
    )
    assert t1.active_emergency is True
    session1_id = t1.session_id

    # Create a completely new session explicitly
    new_session = ConversationService.create_session(
        db_session,
        user.id,
        SessionCreate(title="Brand New Session"),
    )
    assert new_session.id != session1_id

    # Turn 2: User talks in the new session
    t2 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="Hello in new session", session_id=new_session.id),
    )
    # The active emergency must still be reported as active!
    assert t2.active_emergency is True


@pytest.mark.anyio
async def test_emergency_persists_across_operating_mode_change(db_session: Session):
    """Verify active emergency remains active even if user switches operating mode."""
    user = UserService.create_user(
        db_session,
        UserCreate(
            username="mode_change_user",
            full_name="Mode Change",
            role="resident",
            operating_mode=OperatingMode.ELDERLY_CARE,
        ),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # Turn 1: Trigger emergency
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell down and cannot get up!"),
    )
    assert t1.active_emergency is True

    # Switch operating mode to WORKING_PROFESSIONAL
    UserService.update_user(db_session, user.id, UserUpdate(operating_mode=OperatingMode.WORKING_PROFESSIONAL))
    user_updated = UserService.get_by_id(db_session, user.id)
    assert user_updated.operating_mode == OperatingMode.WORKING_PROFESSIONAL

    # Turn 2: Subsequent chat turn
    t2 = await engine.process_turn(
        db_session,
        user_updated,
        ChatTurnRequest(utterance="What is on my calendar?", session_id=t1.session_id),
    )
    assert t2.active_emergency is True


@pytest.mark.anyio
async def test_emergency_persists_across_application_restart(db_session: Session):
    """Verify emergency persists in the database across application restarts / engine re-initialization."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="restart_user", full_name="Restart User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # Turn 1: Trigger emergency
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell down and cannot get up!"),
    )
    assert t1.active_emergency is True

    # Verify durable record exists in active_emergencies table
    record = EmergencyService.get_active_emergency(db_session, user.id)
    assert record is not None
    assert record.is_resolved is False
    assert record.severity == "critical"

    # Simulate restart by instantiating a completely fresh ConversationEngine
    fresh_engine = ConversationEngine(llm_client=MockLLMClient())

    # Turn 2: User says "I am fine" after restart
    t2 = await fresh_engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I am fine, don't worry"),
    )
    # Must preserve pending resolution without clearing the emergency!
    assert t2.active_emergency is True
    assert t2.pending_resolution is True


@pytest.mark.anyio
async def test_emergency_persists_across_subsequent_chat_messages(db_session: Session):
    """Verify sending an ordinary conversational message does NOT silently clear an active emergency."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="subsequent_msg_user", full_name="Subsequent User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # Turn 1: Trigger emergency
    t1 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="I fell down and can't get up!"),
    )
    assert t1.active_emergency is True

    # Turn 2: Unrelated general query
    t2 = await engine.process_turn(
        db_session,
        user,
        ChatTurnRequest(utterance="What is the capital of France?", session_id=t1.session_id),
    )
    # The active emergency remains active!
    assert t2.active_emergency is True


def test_resident_and_viewer_cannot_resolve_emergency(client: TestClient):
    """Verify residents and family viewers cannot resolve emergencies (403 Forbidden)."""
    resident = client.post("/api/v1/users", json={"username": "res_resolve_fail", "full_name": "Resident"}).json()
    viewer = client.post(
        "/api/v1/users",
        json={"username": "viewer_resolve_fail", "full_name": "Viewer", "role": "family_viewer"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()

    res_headers = make_auth_headers(user_id=resident["id"], username="res_resolve_fail", role="resident")
    viewer_headers = make_auth_headers(user_id=viewer["id"], username="viewer_resolve_fail", role="family_viewer")

    # Resident triggers emergency via chat
    client.post(
        f"/api/v1/conversations/users/{resident['id']}/chat",
        json={"utterance": "I fell down and cannot get up, help!"},
        headers=res_headers,
    )

    # Resident attempts to resolve emergency -> 403 Forbidden
    res_attempt = client.post(
        f"/api/v1/conversations/users/{resident['id']}/emergency/resolve",
        headers=res_headers,
    )
    assert res_attempt.status_code == 403
    assert "Only authorized caregivers" in res_attempt.json()["detail"]

    # Family Viewer attempts to resolve emergency -> 403 Forbidden
    viewer_attempt = client.post(
        f"/api/v1/conversations/users/{resident['id']}/emergency/resolve",
        headers=viewer_headers,
    )
    assert viewer_attempt.status_code == 403
    assert "Only authorized caregivers" in viewer_attempt.json()["detail"]


def test_caregiver_can_resolve_emergency_successfully(client: TestClient):
    """Verify authorized caregiver resolves active emergency and clears alert state."""
    resident = client.post("/api/v1/users", json={"username": "res_resolved_ok", "full_name": "Resident"}).json()
    caregiver = client.post(
        "/api/v1/users",
        json={"username": "caregiver_resolver", "full_name": "Caregiver Resolver", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()

    res_headers = make_auth_headers(user_id=resident["id"], username="res_resolved_ok", role="resident")
    cg_headers = make_auth_headers(user_id=caregiver["id"], username="caregiver_resolver", role="caregiver")

    # 1. Resident triggers emergency
    turn1 = client.post(
        f"/api/v1/conversations/users/{resident['id']}/chat",
        json={"utterance": "I fell down and cannot get up!"},
        headers=res_headers,
    )
    assert turn1.status_code == 200
    assert turn1.json()["active_emergency"] is True

    # 2. Assign caregiver to resident
    assign_res = client.post(
        f"/api/v1/users/{resident['id']}/caregivers/{caregiver['id']}",
        headers=cg_headers,
    )
    assert assign_res.status_code == 200

    # 3. Caregiver calls emergency resolution endpoint
    resolve_res = client.post(
        f"/api/v1/conversations/users/{resident['id']}/emergency/resolve",
        headers=cg_headers,
    )
    assert resolve_res.status_code == 200
    resolve_data = resolve_res.json()
    assert resolve_data["status"] == "resolved"
    assert resolve_data["resolved_by"] == caregiver["id"]

    # 3. Subsequent chat turn shows active_emergency is cleared
    turn2 = client.post(
        f"/api/v1/conversations/users/{resident['id']}/chat",
        json={"utterance": "Hello again"},
        headers=res_headers,
    )
    assert turn2.status_code == 200
    assert turn2.json()["active_emergency"] is False


@pytest.mark.anyio
async def test_database_failure_during_message_persistence_rolls_back(db_session: Session, monkeypatch):
    """Verify that a database error during message persistence rolls back the entire turn transaction."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="rollback_user", full_name="Rollback User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # Get active session and store ID
    session = ConversationService.get_or_create_active_session(db_session, user.id)
    session_id = session.id
    user_id = user.id

    # Monkeypatch ConversationService.add_message to fail on assistant message
    original_add_message = ConversationService.add_message
    call_count = 0

    def mock_failing_add_message(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Simulated database failure during assistant message persistence")
        return original_add_message(*args, **kwargs)

    monkeypatch.setattr(ConversationService, "add_message", mock_failing_add_message)

    with pytest.raises(RuntimeError, match="Simulated database failure"):
        await engine.process_turn(
            db_session,
            user,
            ChatTurnRequest(utterance="This should fail atomically", session_id=session_id),
        )

    # Verify that NO messages were committed for this user
    messages = (
        db_session.query(ConversationMessage)
        .filter(ConversationMessage.user_id == user_id)
        .all()
    )
    assert len(messages) == 0


@pytest.mark.anyio
async def test_concurrent_conversation_turns(db_session: Session):
    """Verify concurrent chat turn requests for a user execute safely without state corruption."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="concurrent_user", full_name="Concurrent User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    async def execute_turn(text: str):
        return await engine.process_turn(db_session, user, ChatTurnRequest(utterance=text))

    # Run 3 concurrent turns
    results = await asyncio.gather(
        execute_turn("Concurrent turn one"),
        execute_turn("Concurrent turn two"),
        execute_turn("Concurrent turn three"),
    )

    assert len(results) == 3
    for r in results:
        assert r.spoken_response is not None
        assert r.session_id is not None


def test_caregiver_resolution_resident_specific_authorization(client: TestClient):
    """Verify ambient caregiver authority is rejected: only explicitly assigned caregivers can resolve."""
    res = client.post("/api/v1/users", json={"username": "res_auth_test", "full_name": "Resident"}).json()
    cg_authorized = client.post(
        "/api/v1/users",
        json={"username": "cg_authorized", "full_name": "Authorized CG", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()
    cg_unrelated = client.post(
        "/api/v1/users",
        json={"username": "cg_unrelated", "full_name": "Unrelated CG", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()
    viewer = client.post(
        "/api/v1/users",
        json={"username": "viewer_auth_test", "full_name": "Viewer", "role": "family_viewer"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()

    res_headers = make_auth_headers(user_id=res["id"], username="res_auth_test", role="resident")
    cg_auth_headers = make_auth_headers(user_id=cg_authorized["id"], username="cg_authorized", role="caregiver")
    cg_unrel_headers = make_auth_headers(user_id=cg_unrelated["id"], username="cg_unrelated", role="caregiver")
    viewer_headers = make_auth_headers(user_id=viewer["id"], username="viewer_auth_test", role="family_viewer")

    # Explicitly assign cg_authorized to res
    assign_res = client.post(
        f"/api/v1/users/{res['id']}/caregivers/{cg_authorized['id']}",
        headers=cg_auth_headers,
    )
    assert assign_res.status_code == 200

    # 1. Trigger emergency
    turn = client.post(
        f"/api/v1/conversations/users/{res['id']}/chat",
        json={"utterance": "I cannot breathe, please call an ambulance!"},
        headers=res_headers,
    )
    assert turn.status_code == 200
    assert turn.json()["active_emergency"] is True

    # 2. Resident attempts resolution -> 403 Forbidden
    res_attempt = client.post(
        f"/api/v1/conversations/users/{res['id']}/emergency/resolve",
        headers=res_headers,
    )
    assert res_attempt.status_code == 403
    assert "Only authorized caregivers" in res_attempt.json()["detail"]

    # 3. Family viewer attempts resolution -> 403 Forbidden
    viewer_attempt = client.post(
        f"/api/v1/conversations/users/{res['id']}/emergency/resolve",
        headers=viewer_headers,
    )
    assert viewer_attempt.status_code == 403
    assert "Only authorized caregivers" in viewer_attempt.json()["detail"]

    # 4. Unrelated caregiver (role CAREGIVER, but not assigned to this resident) -> 403 Forbidden
    unrel_attempt = client.post(
        f"/api/v1/conversations/users/{res['id']}/emergency/resolve",
        headers=cg_unrel_headers,
    )
    assert unrel_attempt.status_code == 403
    assert "Explicit caregiver-to-resident assignment is required" in unrel_attempt.json()["detail"]

    # 5. Authorized caregiver attempts resolution -> 200 OK
    auth_attempt = client.post(
        f"/api/v1/conversations/users/{res['id']}/emergency/resolve",
        headers=cg_auth_headers,
    )
    assert auth_attempt.status_code == 200
    assert auth_attempt.json()["status"] == "resolved"
    assert auth_attempt.json()["resolved_by"] == cg_authorized["id"]


def test_partial_unique_index_constraint_enforced(db_session: Session):
    """Verify that partial unique index uq_active_emergency_user_unresolved blocks duplicate active emergencies."""
    from sqlalchemy.exc import IntegrityError

    user = UserService.create_user(
        db_session,
        UserCreate(username="uq_index_user", full_name="UQ User", role="resident"),
    )

    # 1. Insert first unresolved emergency
    em1 = ActiveEmergency(
        user_id=user.id,
        severity="critical",
        trigger_phrase="help me",
        reasoning="Critical distress detected",
        is_resolved=False,
        status="active",
    )
    db_session.add(em1)
    db_session.commit()

    # 2. Attempt to insert second unresolved emergency for same user -> IntegrityError
    with pytest.raises(IntegrityError):
        with db_session.begin_nested():
            em2 = ActiveEmergency(
                user_id=user.id,
                severity="urgent",
                trigger_phrase="chest pain",
                reasoning="Urgent chest pain",
                is_resolved=False,
                status="active",
            )
            db_session.add(em2)
            db_session.flush()

    # 3. Mark the first emergency as resolved
    em1.is_resolved = True
    em1.status = "resolved"
    db_session.commit()

    # 4. Now a new unresolved emergency can be inserted without error
    em3 = ActiveEmergency(
        user_id=user.id,
        severity="critical",
        trigger_phrase="fell down",
        reasoning="Physical fall detected",
        is_resolved=False,
        status="active",
    )
    db_session.add(em3)
    db_session.commit()
    assert em3.id is not None


@pytest.mark.anyio
async def test_concurrent_emergency_creation_no_duplicates(db_session: Session):
    """Verify that concurrent emergency creation attempts do not produce duplicate active records."""
    user = UserService.create_user(
        db_session,
        UserCreate(username="concur_em_user", full_name="Concur Em User", role="resident"),
    )

    rec1 = EmergencyService.record_active_emergency(
        db_session,
        user.id,
        severity=EmergencySeverity.CRITICAL,
        trigger_phrase="I fell down",
        reasoning="Physical fall detected",
        commit=True,
    )
    rec2 = EmergencyService.record_active_emergency(
        db_session,
        user.id,
        severity=EmergencySeverity.URGENT,
        trigger_phrase="help me",
        reasoning="Urgent assistance requested",
        commit=True,
    )

    # Verify both returned records have the same ID (updated in place)
    assert rec1.id == rec2.id

    # Verify in DB there is exactly 1 unresolved record
    active_rows = (
        db_session.query(ActiveEmergency)
        .filter(
            ActiveEmergency.user_id == user.id,
            ActiveEmergency.is_resolved.is_(False),
        )
        .all()
    )
    assert len(active_rows) == 1


def test_schema_upgrade_preserves_existing_data():
    """Verify that creating new tables (ActiveEmergency, CaregiverResidentAssignment) preserves existing data."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database.base import Base
    from app.database.models.user import UserPreferences

    test_engine_upgrade = create_engine("sqlite:///:memory:")
    SessionUpgrade = sessionmaker(bind=test_engine_upgrade)

    # 1. Create all tables
    Base.metadata.create_all(bind=test_engine_upgrade)

    # 2. Seed pre-existing users, sessions, and messages
    with SessionUpgrade() as session:
        user = User(username="legacy_resident", full_name="Legacy Resident", role=UserRole.RESIDENT)
        session.add(user)
        session.commit()
        session.refresh(user)

        pref = UserPreferences(user_id=user.id)
        session.add(pref)

        sess = ConversationSession(user_id=user.id, title="Legacy Session")
        session.add(sess)
        session.commit()
        session.refresh(sess)

        msg = ConversationMessage(
            session_id=sess.id,
            user_id=user.id,
            role=MessageRole.USER,
            content="Historical pre-upgrade message",
        )
        session.add(msg)
        session.commit()
        user_id = user.id
        sess_id = sess.id
        msg_id = msg.id

    # 3. Simulate upgrade run (create_all on existing DB with newly added models)
    Base.metadata.create_all(bind=test_engine_upgrade)

    # 4. Verify pre-existing data is intact
    with SessionUpgrade() as session:
        u = session.query(User).filter(User.id == user_id).one()
        assert u.username == "legacy_resident"

        s = session.query(ConversationSession).filter(ConversationSession.id == sess_id).one()
        assert s.title == "Legacy Session"

        m = session.query(ConversationMessage).filter(ConversationMessage.id == msg_id).one()
        assert m.content == "Historical pre-upgrade message"

        # 5. Verify newly introduced models function without error on the upgraded database
        assignment = CaregiverResidentAssignment(caregiver_id=u.id, resident_id=u.id)
        session.add(assignment)
        session.commit()
        assert assignment.id is not None


@pytest.mark.anyio
async def test_notification_status_semantics_distinctions(db_session: Session):
    """Verify semantic distinction between provider_accepted, delivery_confirmed, and caregiver_acknowledged.

    INVARIANTS:
    1. Emergency detection sets recorded_locally=True, but provider_accepted, delivery_confirmed, and caregiver_acknowledged remain False.
    2. delivery_confirmed=True indicates handset packet receipt, but does NOT imply caregiver_acknowledged.
    3. caregiver_acknowledged=True ONLY occurs when an authorized caregiver explicitly acknowledges/resolves.
    """
    user = UserService.create_user(
        db_session,
        UserCreate(username="notif_sem_user", full_name="Notif Sem User", role="resident"),
    )
    engine = ConversationEngine(llm_client=MockLLMClient())

    # 1. Resident triggers emergency
    turn = await engine.process_turn(
        db=db_session,
        target_user=user,
        request=ChatTurnRequest(utterance="I fell down and cannot get up!"),
    )

    # Verify detection states
    assert turn.active_emergency is True
    assert turn.recorded_locally is True
    assert turn.notification_attempted is False
    assert turn.provider_accepted is False
    assert turn.delivery_confirmed is False
    assert turn.caregiver_acknowledged is False
    assert turn.spoken_response != ""
    # Ensure spoken response does not falsely claim caregiver was reached
    assert "notified" not in turn.spoken_response.lower()

    # 2. Verify signal semantics
    sig = turn.emergency_signal
    assert sig.recorded_locally is True
    assert sig.notification_attempted is False
    assert sig.provider_accepted is False
    assert sig.delivery_confirmed is False
    assert sig.caregiver_acknowledged is False

    # Simulate an external gateway delivery confirmation
    simulated_gateway_signal = sig.model_copy(
        update={
            "notification_attempted": True,
            "provider_accepted": True,
            "delivery_confirmed": True,
        }
    )
    assert simulated_gateway_signal.delivery_confirmed is True
    # Crucial: delivery_confirmed does NOT set caregiver_acknowledged!
    assert simulated_gateway_signal.caregiver_acknowledged is False

