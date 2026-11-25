"""ConversationEngine orchestrating dialogue turns, deterministic emergency checks, and local LLMs."""

import json
import logging
from typing import Optional
from sqlalchemy.orm import Session

from app.conversation.schemas import MessageCreate
from app.conversation.service import ConversationService
from app.core.context_engine import ContextEngine
from app.database.models.conversation import ConversationSession, ConversationState, MessageRole
from app.database.models.user import OperatingMode, SupportedLanguage, User
from app.intelligence.emergency import EmergencySignalDetector
from app.intelligence.llm_client import LLMClientError, LocalLLMClient, MockLLMClient
from app.intelligence.parser import ResponseParser
from app.intelligence.permissions import ActionPermissionDeniedError, PermissionChecker
from app.intelligence.prompt_builder import PromptBuilder
from app.intelligence.schemas import (
    ActionType,
    ChatTurnRequest,
    ChatTurnResponse,
    EmergencySeverity,
    EmergencySignal,
    IntentType,
    ProposedAction,
)
from app.intelligence.validator import ActionValidationError, ActionValidator

logger = logging.getLogger("carebot.intelligence.engine")


class ConversationEngine:
    """Central orchestrator managing multi-turn conversational dialogue for CARE-BOT.

    INVIOLABLE SAFETY & SECURITY INVARIANTS:
    1. Deterministic Emergency Detection Precedes LLM: Emergency identification runs BEFORE any LLM call.
    2. Confirmed Emergencies Remain Active: A confirmed CRITICAL emergency cannot be cleared by
       verbal affirmation alone; it remains active until authorized caregiver resolution is completed.
    3. LLM is Advisory Only: The LLM has zero database access, cannot execute actions, and cannot
       override server-side authorization boundaries.
    4. Identity Security: All access is bound to the authenticated caller; arbitrary request body
       user identifiers are strictly ignored.
    5. Truthful System Claims: The engine does NOT claim live caregiver dispatch or 911 calls
       unless an actual delivery service exists and is verified.
    """

    def __init__(self, llm_client: Optional[LocalLLMClient] = None):
        self.llm_client: LocalLLMClient = llm_client or MockLLMClient()

    def _get_localized_emergency_response(
        self,
        language: SupportedLanguage,
        reasoning: str,
        is_pending_resolution: bool = False,
    ) -> str:
        """Construct truthful, localized emergency guidance without claiming unverified dispatch."""
        if is_pending_resolution:
            if language == SupportedLanguage.HINDI:
                return (
                    "आपकी बात दर्ज कर ली गई है। लेकिन पहले दर्ज की गई गंभीर आपातकालीन स्थिति के कारण, "
                    "अधिकृत देखभालकर्ता द्वारा समाधान प्रक्रिया पूरी होने तक सुरक्षा अलर्ट सक्रिय रहेगा।"
                )
            elif language == SupportedLanguage.BENGALI:
                return (
                    "আপনার বার্তা নথিভুক্ত করা হয়েছে। তবে পূর্বে গুরুতর জরুরি অবস্থা রেকর্ড থাকায়, "
                    "অনুমোদিত কেয়ারগিভার সমাধান না করা পর্যন্ত সুরক্ষা অ্যালার্ট সক্রিয় থাকবে।"
                )
            return (
                "Your verbal statement has been recorded. However, because a critical emergency was "
                "previously confirmed, safety protocols remain active until an authorized caregiver "
                "completes the resolution procedure."
            )

        # Standard emergency response (Truthful: acknowledges detection without falsely claiming live 911 dispatch)
        if language == SupportedLanguage.HINDI:
            return (
                f"आपातकालीन संकेत दर्ज किया गया: {reasoning}। कृपया शांत रहें और सुरक्षित स्थिति में रहें। "
                "(नोट: सिस्टम ने इस आपातकालीन स्थिति को सर्वोच्च प्राथमिकता दी है।)"
            )
        elif language == SupportedLanguage.BENGALI:
            return (
                f"জরুরি সংকেত নথিভুক্ত করা হয়েছে: {reasoning}। অনুগ্রহ করে শান্ত ও নিরাপদ অবস্থানে থাকুন। "
                "(নোট: সিস্টেমটি এই জরুরি অবস্থাকে সর্বোচ্চ অগ্রাধিকার দিয়েছে।)"
            )
        return (
            f"Emergency signal registered: {reasoning}. Please remain calm and stay in a safe position. "
            "(System note: Emergency alert prioritized. Live dispatch delivery will occur upon external integration.)"
        )

    def _get_localized_safe_confirmation_response(self, language: SupportedLanguage) -> str:
        """Construct localized response acknowledging affirmative stand-down of uncertain checks."""
        if language == SupportedLanguage.HINDI:
            return "यह सुनकर बहुत अच्छा लगा कि आप ठीक हैं। मैंने वेलफेयर चेक रोक दिया है। यदि कोई और मदद चाहिए तो बताइए।"
        elif language == SupportedLanguage.BENGALI:
            return "আপনি ভালো আছেন শুনে স্বস্তি পেলাম। আমি চেক-ইন বন্ধ করেছি। অন্য কোনো প্রয়োজন হলে জানাবেন।"
        return "I'm glad to hear that you are okay. I have stood down the welfare check. Please let me know if you need anything else."

    async def process_turn(
        self,
        db: Session,
        target_user: User,
        request: ChatTurnRequest,
    ) -> ChatTurnResponse:
        """Execute a complete conversational interaction turn."""
        user_utterance = request.utterance.strip()

        # 1. Retrieve or initialize conversation session
        if request.session_id:
            session = ConversationService.get_session(db, target_user.id, request.session_id)
        else:
            session = ConversationService.get_or_create_active_session(db, target_user.id)

        # 2. Extract persistent session context metadata (e.g. prior active emergency state)
        context_data = {}
        if session.context_metadata:
            try:
                context_data = json.loads(session.context_metadata)
            except Exception:
                context_data = {}

        prior_emergency_signal: Optional[EmergencySignal] = None
        if "active_emergency" in context_data and isinstance(context_data["active_emergency"], dict):
            try:
                prior_emergency_signal = EmergencySignal.model_validate(context_data["active_emergency"])
            except Exception:
                prior_emergency_signal = None

        pref_lang = (
            target_user.preferences.preferred_language
            if target_user.preferences
            else SupportedLanguage.ENGLISH
        )

        # 3. Step 1: Pre-LLM Deterministic Emergency Signal Evaluation
        emergency_signal = EmergencySignalDetector.evaluate_utterance(
            user_utterance,
            prior_state=prior_emergency_signal,
        )

        # --- BRANCH A: Confirmed Critical Emergency with User Affirmation ---
        # The user said "I am fine" while a confirmed CRITICAL emergency is active.
        if emergency_signal.pending_resolution:
            logger.warning(
                "User %d stated affirmative phrase ('%s') during active CRITICAL emergency. "
                "Affirmation preserved with pending_resolution=True.",
                target_user.id,
                user_utterance,
            )
            # Update session metadata maintaining the active critical emergency
            context_data["active_emergency"] = emergency_signal.model_dump()
            session.context_metadata = json.dumps(context_data)
            session.current_state = ConversationState.ACTIVE_CONVERSATION

            # Persist messages
            user_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="emergency"),
            )
            spoken_resp = self._get_localized_emergency_response(
                pref_lang,
                emergency_signal.reasoning,
                is_pending_resolution=True,
            )
            assistant_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.ASSISTANT, content=spoken_resp, intent_detected="emergency"),
            )
            db.commit()

            return ChatTurnResponse(
                session_id=session.id,
                user_message_id=user_msg.id,
                user_utterance=user_utterance,
                assistant_message_id=assistant_msg.id,
                spoken_response=spoken_resp,
                intent=IntentType.EMERGENCY,
                proposed_action=ProposedAction(
                    action_type=ActionType.TRIGGER_EMERGENCY_ALERT,
                    parameters={
                        "severity": "critical",
                        "pending_resolution": True,
                        "user_affirmation": emergency_signal.user_affirmation,
                    },
                    requires_confirmation=False,
                ),
                emergency_signal=emergency_signal,
                session_state=session.current_state.value,
                active_emergency=True,
                pending_resolution=True,
            )

        # --- BRANCH B: User Confirmed Safe After Ordinary Uncertain Check ---
        if emergency_signal.is_safe_confirmed:
            logger.info("User %d confirmed safe. Standing down uncertain welfare check.", target_user.id)
            context_data.pop("active_emergency", None)
            session.context_metadata = json.dumps(context_data)
            session.current_state = ConversationState.ACTIVE_CONVERSATION

            user_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="greeting"),
            )
            spoken_resp = self._get_localized_safe_confirmation_response(pref_lang)
            assistant_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.ASSISTANT, content=spoken_resp, intent_detected="general_chat"),
            )
            db.commit()

            return ChatTurnResponse(
                session_id=session.id,
                user_message_id=user_msg.id,
                user_utterance=user_utterance,
                assistant_message_id=assistant_msg.id,
                spoken_response=spoken_resp,
                intent=IntentType.GENERAL_CHAT,
                proposed_action=ProposedAction(action_type=ActionType.NONE),
                emergency_signal=emergency_signal,
                session_state=session.current_state.value,
                active_emergency=False,
                pending_resolution=False,
            )

        # --- BRANCH C: New or Repeated CRITICAL / URGENT Emergency ---
        if emergency_signal.detected and emergency_signal.severity in (
            EmergencySeverity.CRITICAL,
            EmergencySeverity.URGENT,
        ):
            logger.error(
                "Emergency signal triggered for user %d: %s (%s). Bypassing LLM.",
                target_user.id,
                emergency_signal.severity.value,
                emergency_signal.trigger_phrase,
            )
            context_data["active_emergency"] = emergency_signal.model_dump()
            session.context_metadata = json.dumps(context_data)
            session.current_state = ConversationState.ACTIVE_CONVERSATION

            user_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="emergency"),
            )
            spoken_resp = self._get_localized_emergency_response(
                pref_lang,
                emergency_signal.reasoning,
                is_pending_resolution=False,
            )
            assistant_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.ASSISTANT, content=spoken_resp, intent_detected="emergency"),
            )
            db.commit()

            return ChatTurnResponse(
                session_id=session.id,
                user_message_id=user_msg.id,
                user_utterance=user_utterance,
                assistant_message_id=assistant_msg.id,
                spoken_response=spoken_resp,
                intent=IntentType.EMERGENCY,
                proposed_action=ProposedAction(
                    action_type=ActionType.TRIGGER_EMERGENCY_ALERT,
                    parameters={
                        "severity": emergency_signal.severity.value,
                        "trigger_phrase": emergency_signal.trigger_phrase,
                        "repeat_count": emergency_signal.repeat_count,
                    },
                    requires_confirmation=False,
                ),
                emergency_signal=emergency_signal,
                session_state=session.current_state.value,
                active_emergency=True,
                pending_resolution=False,
            )

        # --- BRANCH D: Normal Conversational Flow (LLM Evaluated) ---
        # 1. Persist user utterance
        user_msg = ConversationService.add_message(
            db,
            target_user.id,
            session.id,
            MessageCreate(role=MessageRole.USER, content=user_utterance),
        )

        # 2. Fetch synthesized user context (profile, operating mode, preferences, sliding history)
        context = ContextEngine.get_user_context(db, target_user.id, history_window=10)

        # 3. Assemble prompt
        system_prompt, user_prompt = PromptBuilder.build_prompt(context, user_utterance)

        # 4. Invoke Local LLM Client
        raw_llm_content = ""
        try:
            llm_resp = await self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            )
            raw_llm_content = llm_resp.content
        except LLMClientError as exc:
            logger.warning("LLM client failure for user %d: %s. Using fail-safe fallback.", target_user.id, exc)
            raw_llm_content = ""

        # 5. Defensive Response Parsing
        parsed_response = ResponseParser.parse(
            raw_llm_content,
            fallback_language=pref_lang.value,
        )

        # 6. Action Whitelist & Boundary Validation
        proposed_action = parsed_response.proposed_action
        try:
            proposed_action = ActionValidator.validate_action(proposed_action)
        except ActionValidationError as exc:
            logger.warning("Action validation rejected LLM proposal for user %d: %s", target_user.id, exc)
            proposed_action = ProposedAction(action_type=ActionType.NONE)

        # 7. Server-Side RBAC & Authorization Enforcement
        try:
            PermissionChecker.check_permission(target_user, proposed_action)
        except ActionPermissionDeniedError as exc:
            logger.warning("Permission denied for action proposal for user %d: %s", target_user.id, exc)
            proposed_action = ProposedAction(action_type=ActionType.NONE)

        # If uncertain distress was detected in Step 1, track it in session metadata
        if emergency_signal.detected and emergency_signal.severity == EmergencySeverity.UNCERTAIN:
            context_data["active_emergency"] = emergency_signal.model_dump()
            session.context_metadata = json.dumps(context_data)

        # 8. Persist assistant response
        assistant_msg = ConversationService.add_message(
            db,
            target_user.id,
            session.id,
            MessageCreate(
                role=MessageRole.ASSISTANT,
                content=parsed_response.spoken_response,
                intent_detected=parsed_response.intent.value,
            ),
        )

        # 9. Update session state
        session.current_state = ConversationState.ACTIVE_CONVERSATION
        db.commit()

        return ChatTurnResponse(
            session_id=session.id,
            user_message_id=user_msg.id,
            user_utterance=user_utterance,
            assistant_message_id=assistant_msg.id,
            spoken_response=parsed_response.spoken_response,
            intent=parsed_response.intent,
            proposed_action=proposed_action,
            emergency_signal=emergency_signal,
            session_state=session.current_state.value,
            active_emergency=bool("active_emergency" in context_data),
            pending_resolution=False,
        )
