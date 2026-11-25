"""Unit tests for PromptBuilder."""

from datetime import datetime, timezone
from app.conversation.schemas import MessageResponse, UserContextPayload
from app.database.models.conversation import ConversationState, MessageRole
from app.database.models.user import OperatingMode, ResponseStyle, SupportedLanguage
from app.intelligence.prompt_builder import PromptBuilder


def test_prompt_builder_elderly_care_mode():
    """Verify prompt builder formats persona, language rules, and history for Elderly Care."""
    ctx = UserContextPayload(
        user_id=10,
        username="grandma_mary",
        full_name="Mary Smith",
        operating_mode=OperatingMode.ELDERLY_CARE,
        preferred_language=SupportedLanguage.ENGLISH,
        response_style=ResponseStyle.SIMPLE_AND_SLOW,
        active_session_id="session-123",
        session_state=ConversationState.ACTIVE_CONVERSATION,
        system_instructions="Speak gently, warmly, and clearly.",
        recent_messages=[
            MessageResponse(
                id=1,
                session_id="session-123",
                user_id=10,
                role=MessageRole.USER,
                content="Hello CARE-BOT",
                created_at=datetime.now(timezone.utc),
            ),
            MessageResponse(
                id=2,
                session_id="session-123",
                user_id=10,
                role=MessageRole.ASSISTANT,
                content="Hello Mary, how are you feeling today?",
                created_at=datetime.now(timezone.utc),
            ),
        ],
    )
    system_prompt, user_prompt = PromptBuilder.build_prompt(
        context=ctx,
        user_utterance="I am feeling a little tired.",
    )

    assert "Speak gently, warmly, and clearly." in system_prompt
    assert "SCHEMA:" in system_prompt
    assert "EXEMPLAR:" in system_prompt
    assert "heart medicine" in system_prompt  # Elderly care exemplar

    assert "--- RECENT CONVERSATION HISTORY ---" in user_prompt
    assert "Resident: Hello CARE-BOT" in user_prompt
    assert "Assistant: Hello Mary" in user_prompt
    assert "Resident: I am feeling a little tired." in user_prompt
    assert user_prompt.endswith("Assistant:")


def test_prompt_builder_multilingual_exemplars():
    """Verify prompt builder incorporates language-specific few-shots."""
    ctx_hi = UserContextPayload(
        user_id=11,
        username="shanti_devi",
        full_name="Shanti Devi",
        operating_mode=OperatingMode.ELDERLY_CARE,
        preferred_language=SupportedLanguage.HINDI,
        response_style=ResponseStyle.STANDARD,
        active_session_id="session-456",
        session_state=ConversationState.IDLE,
        system_instructions="Interact in Hindi.",
        recent_messages=[],
    )
    sys_hi, _ = PromptBuilder.build_prompt(ctx_hi, "नमस्ते")
    assert "दवा" in sys_hi  # Hindi exemplar

    ctx_bn = UserContextPayload(
        user_id=12,
        username="anita_sen",
        full_name="Anita Sen",
        operating_mode=OperatingMode.ELDERLY_CARE,
        preferred_language=SupportedLanguage.BENGALI,
        response_style=ResponseStyle.STANDARD,
        active_session_id="session-789",
        session_state=ConversationState.IDLE,
        system_instructions="Interact in Bengali.",
        recent_messages=[],
    )
    sys_bn, _ = PromptBuilder.build_prompt(ctx_bn, "নমস্কার")
    assert "ওষুধ" in sys_bn  # Bengali exemplar
