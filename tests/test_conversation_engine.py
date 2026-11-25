"""Comprehensive unit and integration tests for ConversationEngine orchestration."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.database.models.conversation import ConversationMessage, ConversationSession, MessageRole
from app.database.models.user import OperatingMode, ResponseStyle, SupportedLanguage, User, UserRole
from app.intelligence.engine import ConversationEngine
from app.intelligence.llm_client import MockLLMClient
from app.intelligence.schemas import (
    ActionType,
    ChatTurnRequest,
    EmergencySeverity,
    IntentType,
)
from app.users.schemas import UserCreate
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

    # 3. Truthful response: acknowledges detection without falsely claiming live 911 dispatch
    assert "Emergency signal registered" in res.spoken_response
    assert "Emergency alert prioritized" in res.spoken_response


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
