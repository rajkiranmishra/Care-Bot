"""Automated tests for Context Engine synthesis and prompt preparation."""

from fastapi.testclient import TestClient
from app.core.context_engine import ContextEngine
from app.database.models.user import OperatingMode, ResponseStyle, SupportedLanguage


def test_system_instructions_modes():
    """Verify distinct system instruction generation for Elderly Care vs. Working Professional."""
    elderly_prompt = ContextEngine.build_system_instructions(
        full_name="Grandma Sarah",
        mode=OperatingMode.ELDERLY_CARE,
        lang=SupportedLanguage.ENGLISH,
        style=ResponseStyle.SIMPLE_AND_SLOW,
    )
    assert "Grandma Sarah" in elderly_prompt
    assert "Elderly Care Mode" in elderly_prompt
    assert "Speak gently, warmly, and clearly" in elderly_prompt
    assert "simple vocabulary" in elderly_prompt

    pro_prompt = ContextEngine.build_system_instructions(
        full_name="Alex Johnson",
        mode=OperatingMode.WORKING_PROFESSIONAL,
        lang=SupportedLanguage.ENGLISH,
        style=ResponseStyle.CONCISE_AND_FAST,
    )
    assert "Alex Johnson" in pro_prompt
    assert "Working Professional Mode" in pro_prompt
    assert "concise, efficient, and direct" in pro_prompt
    assert "avoiding unnecessary conversational filler" in pro_prompt


def test_system_instructions_multilingual():
    """Verify language-specific guidance for Hindi and Bengali."""
    hindi_prompt = ContextEngine.build_system_instructions(
        full_name="Ramesh",
        mode=OperatingMode.ELDERLY_CARE,
        lang=SupportedLanguage.HINDI,
        style=ResponseStyle.STANDARD,
    )
    assert "Language Rule: Interact primarily in Hindi" in hindi_prompt

    bengali_prompt = ContextEngine.build_system_instructions(
        full_name="Subhash",
        mode=OperatingMode.ELDERLY_CARE,
        lang=SupportedLanguage.BENGALI,
        style=ResponseStyle.STANDARD,
    )
    assert "Language Rule: Interact primarily in Bengali" in bengali_prompt


def test_context_engine_endpoint_synthesis(client: TestClient):
    """Verify GET /api/v1/conversations/users/{user_id}/context synthesizes complete LLM payload."""
    user = client.post(
        "/api/v1/users",
        json={
            "username": "context_resident",
            "full_name": "Aparna Sen",
            "operating_mode": "elderly_care",
            "initial_language": "bn",
        },
    ).json()
    user_id = user["id"]

    # Initialize a session and post messages
    session = client.post(f"/api/v1/conversations/users/{user_id}/sessions", json={"title": "Tea Request"}).json()
    session_id = session["id"]

    client.post(
        f"/api/v1/conversations/users/{user_id}/sessions/{session_id}/messages",
        json={"role": "user", "content": "আমাকে চা দিতে বলো।", "detected_language": "bn"},
    )
    client.post(
        f"/api/v1/conversations/users/{user_id}/sessions/{session_id}/messages",
        json={"role": "assistant", "content": "আমি আপনার পরিবারকে জানাচ্ছি।", "detected_language": "bn"},
    )

    # Fetch compiled context
    context_res = client.get(f"/api/v1/conversations/users/{user_id}/context?history_window=5")
    assert context_res.status_code == 200
    context = context_res.json()

    assert context["user_id"] == user_id
    assert context["full_name"] == "Aparna Sen"
    assert context["operating_mode"] == "elderly_care"
    assert context["preferred_language"] == "bn"
    assert "Elderly Care Mode" in context["system_instructions"]
    assert "Bengali" in context["system_instructions"]

    # History window verification
    assert len(context["recent_messages"]) == 2
    assert context["recent_messages"][0]["content"] == "আমাকে চা দিতে বলো।"
    assert context["recent_messages"][1]["content"] == "আমি আপনার পরিবারকে জানাচ্ছি।"


def test_context_engine_cross_user_isolation(client: TestClient):
    """Verify context compilation for User 1 never leaks User 2's dialogue history."""
    u1 = client.post("/api/v1/users", json={"username": "user_one", "full_name": "User 1"}).json()
    u2 = client.post("/api/v1/users", json={"username": "user_two", "full_name": "User 2"}).json()

    # User 1 session & message
    s1 = client.post(f"/api/v1/conversations/users/{u1['id']}/sessions", json={}).json()
    client.post(
        f"/api/v1/conversations/users/{u1['id']}/sessions/{s1['id']}/messages",
        json={"role": "user", "content": "User 1 Secret Note"},
    )

    # User 2 context fetch
    u2_context = client.get(f"/api/v1/conversations/users/{u2['id']}/context").json()
    messages_content = [m["content"] for m in u2_context["recent_messages"]]

    assert "User 1 Secret Note" not in messages_content
