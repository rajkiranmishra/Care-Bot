"""Context Engine synthesizing user profiles, modes, preferences, and dialogue history for AI."""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.conversation.schemas import MessageResponse, UserContextPayload
from app.conversation.service import ConversationService
from app.database.models.conversation import ConversationState
from app.database.models.user import OperatingMode, ResponseStyle, SupportedLanguage
from app.users.service import UserService


class ContextEngine:
    """Builds isolated, context-aware prompt payloads for downstream local LLMs."""

    @staticmethod
    def build_system_instructions(
        full_name: str,
        mode: OperatingMode,
        lang: SupportedLanguage,
        style: ResponseStyle,
    ) -> str:
        """Generate tailored system instructions based on mode and language."""
        lang_instruction = {
            SupportedLanguage.ENGLISH: "Interact primarily in English.",
            SupportedLanguage.HINDI: "Interact primarily in Hindi (using Devanagari script or clean transliteration as appropriate).",
            SupportedLanguage.BENGALI: "Interact primarily in Bengali (using Bengali script).",
        }.get(lang, "Interact in English.")

        if mode == OperatingMode.ELDERLY_CARE:
            mode_guidance = (
                f"You are CARE-BOT, assisting {full_name} in Elderly Care Mode. "
                "Speak gently, warmly, and clearly. Keep sentences simple, unhurried, and reassuring. "
                "Prioritize health reminders, daily wellness, and family connection. "
                "If a request is unclear or incomplete, ask simple clarifying questions one at a time. "
                "Never diagnose medical conditions or give definitive medical advice."
            )
        else:
            mode_guidance = (
                f"You are CARE-BOT, assisting {full_name} in Working Professional Mode. "
                "Be concise, efficient, and direct. Focus on task management, scheduling, daily agendas, "
                "and contextual information like weather. Keep responses brief and action-oriented."
            )

        style_guidance = {
            ResponseStyle.SIMPLE_AND_SLOW: "Use short sentences, simple vocabulary, and extra clarity.",
            ResponseStyle.STANDARD: "Use natural, balanced conversational depth.",
            ResponseStyle.CONCISE_AND_FAST: "Be extremely concise, avoiding unnecessary conversational filler.",
        }.get(style, "Use balanced conversational depth.")

        return f"{mode_guidance}\nLanguage Rule: {lang_instruction}\nStyle Rule: {style_guidance}"

    @staticmethod
    def get_user_context(
        db: Session,
        user_id: int,
        history_window: int = 10,
    ) -> UserContextPayload:
        """Assemble the complete context payload for a user's active dialogue session."""
        user = UserService.get_by_id(db, user_id)
        prefs = user.preferences

        preferred_language = prefs.preferred_language if prefs else SupportedLanguage.ENGLISH
        response_style = prefs.response_style if prefs else ResponseStyle.STANDARD

        # Retrieve active session or create one
        session = ConversationService.get_or_create_active_session(db, user_id)

        # Retrieve recent message history
        raw_messages = ConversationService.get_messages(
            db,
            user_id=user_id,
            session_id=session.id,
            limit=history_window,
        )
        recent_messages = [MessageResponse.model_validate(msg) for msg in raw_messages]

        system_instructions = ContextEngine.build_system_instructions(
            full_name=user.full_name,
            mode=user.operating_mode,
            lang=preferred_language,
            style=response_style,
        )

        return UserContextPayload(
            user_id=user.id,
            username=user.username,
            full_name=user.full_name,
            operating_mode=user.operating_mode,
            preferred_language=preferred_language,
            response_style=response_style,
            active_session_id=session.id,
            session_state=session.current_state,
            system_instructions=system_instructions,
            recent_messages=recent_messages,
        )
