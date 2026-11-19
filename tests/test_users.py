"""Automated tests for user profiles, operating modes, preferences, and validation."""

from fastapi.testclient import TestClient
from tests.conftest import TEST_DEVICE_KEY, make_auth_headers


def test_create_user_elderly_mode_success(client: TestClient):
    """Verify creating a user in Elderly Care Mode with tailored default preferences."""
    payload = {
        "username": "alice_elderly",
        "full_name": "Alice Smith",
        "email": "alice@example.com",
        "role": "resident",
        "operating_mode": "elderly_care",
        "nfc_card_id": "NFC-CARD-001",
    }
    response = client.post("/api/v1/users", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["username"] == "alice_elderly"
    assert data["operating_mode"] == "elderly_care"
    assert data["nfc_card_id"] == "NFC-CARD-001"
    assert data["is_active"] is True

    # Check elderly defaults
    prefs = data["preferences"]
    assert prefs is not None
    assert prefs["response_style"] == "simple_slow"
    assert prefs["high_contrast_dashboard"] is True
    assert prefs["spoken_speed"] == 0.9


def test_create_user_professional_mode_success(client: TestClient):
    """Verify creating a user in Working Professional Mode with tailored default preferences."""
    payload = {
        "username": "bob_professional",
        "full_name": "Bob Miller",
        "email": "bob@work.com",
        "role": "resident",
        "operating_mode": "working_professional",
        "nfc_card_id": "NFC-CARD-002",
        "initial_language": "hi",
    }
    response = client.post("/api/v1/users", json=payload)
    assert response.status_code == 201
    data = response.json()

    assert data["operating_mode"] == "working_professional"
    prefs = data["preferences"]
    assert prefs["response_style"] == "concise_fast"
    assert prefs["preferred_language"] == "hi"
    assert prefs["spoken_speed"] == 1.0


def test_duplicate_username_fails(client: TestClient):
    """Verify registering a duplicate username returns 409 Conflict."""
    payload = {
        "username": "duplicate_user",
        "full_name": "User One",
    }
    res1 = client.post("/api/v1/users", json=payload)
    assert res1.status_code == 201

    res2 = client.post("/api/v1/users", json=payload)
    assert res2.status_code == 409
    assert "already registered" in res2.json()["detail"]


def test_duplicate_email_fails(client: TestClient):
    """Verify duplicate email returns 409 Conflict."""
    client.post("/api/v1/users", json={"username": "user_a", "full_name": "A", "email": "shared@domain.com"})
    res = client.post("/api/v1/users", json={"username": "user_b", "full_name": "B", "email": "shared@domain.com"})
    assert res.status_code == 409
    assert "already registered" in res.json()["detail"]


def test_duplicate_nfc_card_fails(client: TestClient):
    """Verify duplicate NFC card ID returns 409 Conflict."""
    client.post("/api/v1/users", json={"username": "user_nfc_1", "full_name": "NFC 1", "nfc_card_id": "CARD-DUP"})
    res = client.post("/api/v1/users", json={"username": "user_nfc_2", "full_name": "NFC 2", "nfc_card_id": "CARD-DUP"})
    assert res.status_code == 409
    assert "already assigned" in res.json()["detail"]


def test_invalid_username_format_rejected(client: TestClient):
    """Verify non-alphanumeric characters in username are rejected with 422."""
    res = client.post("/api/v1/users", json={"username": "user with spaces!", "full_name": "Test User"})
    assert res.status_code == 422


def test_get_user_by_id_and_not_found(client: TestClient):
    """Verify fetching user by ID and 404 for non-existent users with valid auth."""
    res = client.post("/api/v1/users", json={"username": "fetch_user", "full_name": "Fetch Test"})
    user_id = res.json()["id"]
    headers = make_auth_headers(user_id=user_id, username="fetch_user")

    get_res = client.get(f"/api/v1/users/{user_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["username"] == "fetch_user"

    # Register an active Caregiver (via authorized device provisioning) to query a non-existent ID -> 404
    cg = client.post(
        "/api/v1/users",
        json={"username": "cg_inspector", "full_name": "Caregiver", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()
    cg_headers = make_auth_headers(user_id=cg["id"], username="cg_inspector", role="caregiver")
    not_found = client.get("/api/v1/users/99999", headers=cg_headers)
    assert not_found.status_code == 404


def test_update_operating_mode(client: TestClient):
    """Verify switching a user's operating mode updates without affecting other users."""
    user1 = client.post("/api/v1/users", json={"username": "mode_switch_1", "full_name": "User 1", "operating_mode": "elderly_care"}).json()
    user2 = client.post("/api/v1/users", json={"username": "mode_switch_2", "full_name": "User 2", "operating_mode": "elderly_care"}).json()

    headers1 = make_auth_headers(user_id=user1["id"], username="mode_switch_1")
    headers2 = make_auth_headers(user_id=user2["id"], username="mode_switch_2")

    # Switch user 1 to working professional
    update_res = client.put(f"/api/v1/users/{user1['id']}", json={"operating_mode": "working_professional"}, headers=headers1)
    assert update_res.status_code == 200
    assert update_res.json()["operating_mode"] == "working_professional"

    # Verify user 2 remains elderly care (individualized mode configuration)
    check_user2 = client.get(f"/api/v1/users/{user2['id']}", headers=headers2).json()
    assert check_user2["operating_mode"] == "elderly_care"


def test_update_user_preferences(client: TestClient):
    """Verify updating language and response style preferences."""
    user = client.post("/api/v1/users", json={"username": "pref_user", "full_name": "Preference User"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="pref_user")

    update_payload = {
        "preferred_language": "bn",
        "response_style": "simple_slow",
        "spoken_speed": 0.85,
        "high_contrast_dashboard": True,
    }
    pref_res = client.put(f"/api/v1/users/{user_id}/preferences", json=update_payload, headers=headers)
    assert pref_res.status_code == 200
    data = pref_res.json()
    assert data["preferred_language"] == "bn"
    assert data["response_style"] == "simple_slow"
    assert data["spoken_speed"] == 0.85
    assert data["high_contrast_dashboard"] is True


def test_lookup_user_by_nfc_card(client: TestClient):
    """Verify resolving user identity via physical NFC card ID with device key."""
    client.post("/api/v1/users", json={"username": "nfc_resident", "full_name": "NFC Resident", "nfc_card_id": "RFID-HEX-1234"})
    device_headers = {"X-Device-Key": "carebot-hardware-device-key-local-2026"}

    found = client.get("/api/v1/users/lookup/nfc/RFID-HEX-1234", headers=device_headers)
    assert found.status_code == 200
    assert found.json()["username"] == "nfc_resident"

    missing = client.get("/api/v1/users/lookup/nfc/UNKNOWN-RFID", headers=device_headers)
    assert missing.status_code == 404


def test_delete_user_cascades(client: TestClient):
    """Verify deleting a user removes record and cascades."""
    # Register an active Caregiver (via authorized device provisioning) so ID cannot collide
    cg = client.post(
        "/api/v1/users",
        json={"username": "cg_checker", "full_name": "Caregiver Checker", "role": "caregiver"},
        headers={"X-Device-Key": TEST_DEVICE_KEY},
    ).json()
    cg_headers = make_auth_headers(user_id=cg["id"], username="cg_checker", role="caregiver")

    user = client.post("/api/v1/users", json={"username": "delete_me", "full_name": "Delete Me"}).json()
    user_id = user["id"]
    headers = make_auth_headers(user_id=user_id, username="delete_me")

    del_res = client.delete(f"/api/v1/users/{user_id}", headers=headers)
    assert del_res.status_code == 204

    # Caregiver queries deleted user_id -> 404 Not Found
    get_res = client.get(f"/api/v1/users/{user_id}", headers=cg_headers)
    assert get_res.status_code == 404
