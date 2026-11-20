"""Automated tests for conversation sessions, message persistence, and cross-user data isolation."""

from fastapi.testclient import TestClient
from tests.conftest import make_auth_headers


def test_create_and_list_sessions(client: TestClient):
    """Verify creating a conversation session associated with a user."""
    user = client.post("/api/v1/users", json={"username": "convo_user_1", "full_name": "Convo User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="convo_user_1")

    # Create session
    session_res = client.post(
        f"/api/v1/conversations/users/{user_id}/sessions",
        json={"title": "Morning Routine", "initial_state": "idle"},
        headers=headers,
    )
    assert session_res.status_code == 201
    session_data = session_res.json()
    assert session_data["title"] == "Morning Routine"
    assert session_data["status"] == "active"
    assert session_data["current_state"] == "idle"
    session_id = session_data["id"]

    # List sessions
    list_res = client.get(f"/api/v1/conversations/users/{user_id}/sessions", headers=headers)
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert len(sessions) >= 1
    assert any(s["id"] == session_id for s in sessions)


def test_message_persistence_and_retrieval(client: TestClient):
    """Verify posting user utterances preserves history chronologically."""
    user = client.post("/api/v1/users", json={"username": "msg_user", "full_name": "Message User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="msg_user")

    session = client.post(
        f"/api/v1/conversations/users/{user_id}/sessions",
        json={"title": "Medication Chat"},
        headers=headers,
    ).json()
    session_id = session["id"]

    # Post user utterance
    msg1 = client.post(
        f"/api/v1/conversations/users/{user_id}/sessions/{session_id}/messages",
        json={"role": "user", "content": "Did I take my blood pressure medicine today?", "detected_language": "en"},
        headers=headers,
    )
    assert msg1.status_code == 201
    assert msg1.json()["content"] == "Did I take my blood pressure medicine today?"

    # Post second user utterance
    msg2 = client.post(
        f"/api/v1/conversations/users/{user_id}/sessions/{session_id}/messages",
        json={"role": "user", "content": "Remind me at 8:00 PM."},
        headers=headers,
    )
    assert msg2.status_code == 201

    # Fetch chronological messages
    msgs_res = client.get(f"/api/v1/conversations/users/{user_id}/sessions/{session_id}/messages", headers=headers)
    assert msgs_res.status_code == 200
    msgs = msgs_res.json()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "Did I take my blood pressure medicine today?"
    assert msgs[1]["content"] == "Remind me at 8:00 PM."


def test_update_session_state(client: TestClient):
    """Verify transitioning session FSM state."""
    user = client.post("/api/v1/users", json={"username": "state_user", "full_name": "State User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="state_user")

    session = client.post(f"/api/v1/conversations/users/{user_id}/sessions", json={}, headers=headers).json()
    session_id = session["id"]

    # Transition to active_conversation and store context metadata
    update_res = client.put(
        f"/api/v1/conversations/users/{user_id}/sessions/{session_id}",
        json={
            "current_state": "waiting_for_confirmation",
            "context_metadata": '{"pending_slot": "time", "action": "create_reminder"}',
        },
        headers=headers,
    )
    assert update_res.status_code == 200
    updated = update_res.json()
    assert updated["current_state"] == "waiting_for_confirmation"
    assert "pending_slot" in updated["context_metadata"]


def test_cross_user_data_isolation_enforced(client: TestClient):
    """Security verification: User B is strictly prohibited from accessing User A's session."""
    user_a = client.post("/api/v1/users", json={"username": "victim_alice", "full_name": "Alice"}).json()
    user_b = client.post("/api/v1/users", json={"username": "attacker_bob", "full_name": "Bob"}).json()

    headers_a = make_auth_headers(user_id=user_a["id"], username="victim_alice")
    headers_b = make_auth_headers(user_id=user_b["id"], username="attacker_bob")

    # Alice creates a private session and posts sensitive message
    session_a = client.post(
        f"/api/v1/conversations/users/{user_a['id']}/sessions",
        json={"title": "Private Medical Chat"},
        headers=headers_a,
    ).json()
    session_a_id = session_a["id"]

    client.post(
        f"/api/v1/conversations/users/{user_a['id']}/sessions/{session_a_id}/messages",
        json={"role": "user", "content": "Doctor prescribed 50mg Metformin."},
        headers=headers_a,
    )

    # Bob attempts to read Alice's session -> Must be 403 Forbidden
    unauthorized_read = client.get(
        f"/api/v1/conversations/users/{user_b['id']}/sessions/{session_a_id}",
        headers=headers_b,
    )
    assert unauthorized_read.status_code == 403

    # Bob attempts to read Alice's messages -> Must be 403 Forbidden
    unauthorized_msgs = client.get(
        f"/api/v1/conversations/users/{user_b['id']}/sessions/{session_a_id}/messages",
        headers=headers_b,
    )
    assert unauthorized_msgs.status_code == 403

    # Bob attempts to inject a message into Alice's session -> Must be 403 Forbidden
    unauthorized_inject = client.post(
        f"/api/v1/conversations/users/{user_b['id']}/sessions/{session_a_id}/messages",
        json={"role": "user", "content": "Malicious payload injection."},
        headers=headers_b,
    )
    assert unauthorized_inject.status_code == 403


def test_session_not_found_handling(client: TestClient):
    """Verify non-existent session ID returns 404."""
    user = client.post("/api/v1/users", json={"username": "nf_user", "full_name": "NF User"}).json()
    headers = make_auth_headers(user_id=user["id"], username="nf_user")

    res = client.get(f"/api/v1/conversations/users/{user['id']}/sessions/NON-EXISTENT-UUID", headers=headers)
    assert res.status_code == 404


def test_empty_message_validation(client: TestClient):
    """Verify empty message string is rejected with 422."""
    user = client.post("/api/v1/users", json={"username": "empty_msg_user", "full_name": "Empty"}).json()
    headers = make_auth_headers(user_id=user["id"], username="empty_msg_user")

    session = client.post(f"/api/v1/conversations/users/{user['id']}/sessions", json={}, headers=headers).json()

    bad_res = client.post(
        f"/api/v1/conversations/users/{user['id']}/sessions/{session['id']}/messages",
        json={"role": "user", "content": ""},
        headers=headers,
    )
    assert bad_res.status_code == 422
