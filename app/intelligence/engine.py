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
from app.intelligence.emergency_service import EmergencyService
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
    ParsedLLMResponse,
    ProposedAction,
)
from app.intelligence.validator import ActionValidationError, ActionValidator

logger = logging.getLogger("carebot.intelligence.engine")


class ConversationEngine:
    """Central orchestrator managing multi-turn conversational dialogue for CARE-BOT.

    INVIOLABLE SAFETY & SECURITY INVARIANTS:
    1. Deterministic Emergency Detection Precedes LLM: Emergency identification runs BEFORE any LLM call.
    2. Durable Emergency Persistence: An active CRITICAL emergency is stored in the database independently
       of conversation sessions. Starting a new session, changing operating mode, restarting the host app,
       or sending another message CANNOT silently clear it.
    3. Confirmed Emergencies Remain Active: A confirmed CRITICAL emergency cannot be cleared by
       verbal affirmation alone; it remains active until authorized caregiver resolution is completed.
    4. Honest Emergency Communication: The engine distinguishes between emergency detected, recorded locally,
       notification attempted, and notification delivered. It NEVER claims that a caregiver is being notified
       or has been notified unless confirmed by a verified notification delivery integration.
    5. Personal Information Protection: The engine and MockLLMClient NEVER invent medication names, schedules,
       reminders, appointments, or medical history. If verified data is not available, lack of access is clearly stated.
    6. LLM is Advisory Only: The LLM has zero database access, cannot execute actions, and cannot
       override server-side authorization boundaries.
    7. Atomic Persistence: Message persistence and state transitions execute within a single atomic database
       transaction with immediate rollback on error.
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

        # Standard emergency response (Truthful: acknowledges local detection without claiming unverified external notification)
        if language == SupportedLanguage.HINDI:
            return (
                f"आपातकालीन संकेत स्थानीय रूप से दर्ज किया गया: {reasoning}। कृपया शांत रहें और सुरक्षित स्थिति में रहें। "
                "(सिस्टम नोट: आपातकाल स्थानीय रूप से दर्ज किया गया है। बाह्य सूचना अभी नहीं भेजी गई है।)"
            )
        elif language == SupportedLanguage.BENGALI:
            return (
                f"জরুরি সংকেত স্থানীয়ভাবে নথিভুক্ত করা হয়েছে: {reasoning}। অনুগ্রহ করে শান্ত ও নিরাপদ অবস্থানে থাকুন। "
                "(সিস্টেম নোট: জরুরি অবস্থা স্থানীয়ভাবে রেকর্ড করা হয়েছে। বাহ্যিক বিজ্ঞপ্তি এখনও পাঠানো হয়নি।)"
            )
        return (
            f"Emergency signal registered locally: {reasoning}. Please remain calm and stay in a safe position. "
            "(System note: Emergency recorded locally. External notification has not been dispatched.)"
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

        # 2. Extract context metadata and persistent active emergency state
        context_data = {}
        if session.context_metadata:
            try:
                context_data = json.loads(session.context_metadata)
            except Exception:
                context_data = {}

        # Query durable active emergency stored independently of conversation sessions
        durable_emergency = EmergencyService.get_active_emergency(db, target_user.id)
        has_durable_active_emergency = durable_emergency is not None

        prior_emergency_signal: Optional[EmergencySignal] = None
        if durable_emergency:
            prior_emergency_signal = EmergencySignal(
                detected=True,
                severity=EmergencySeverity(durable_emergency.severity),
                trigger_phrase=durable_emergency.trigger_phrase,
                reasoning=durable_emergency.reasoning,
                confidence=1.0,
                is_safe_confirmed=False,
                requires_immediate_escalation=(durable_emergency.severity in ("critical", "urgent")),
                pending_resolution=(durable_emergency.status == "pending_resolution"),
                user_affirmation=durable_emergency.user_affirmation,
                recorded_locally=True,
                notification_attempted=False,
                notification_delivered=False,
            )
        elif "active_emergency" in context_data and isinstance(context_data["active_emergency"], dict):
            cached_severity = context_data["active_emergency"].get("severity")
            if cached_severity in ("critical", "urgent"):
                # Clean up previously resolved emergency from session context metadata
                context_data.pop("active_emergency", None)
                session.context_metadata = json.dumps(context_data)
                prior_emergency_signal = None
            else:
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
        if emergency_signal.pending_resolution or (has_durable_active_emergency and emergency_signal.is_safe_confirmed):
            logger.warning(
                "User %d stated affirmative phrase ('%s') during active CRITICAL emergency. "
                "Affirmation preserved with pending_resolution=True. Emergency remains active.",
                target_user.id,
                user_utterance,
            )
            # Emergency remains active in the database; update pending_resolution and affirmation
            emergency_signal = emergency_signal.model_copy(
                update={
                    "pending_resolution": True,
                    "user_affirmation": user_utterance,
                    "recorded_locally": True,
                    "notification_attempted": False,
                    "notification_delivered": False,
                }
            )

            try:
                EmergencyService.record_active_emergency(
                    db,
                    target_user.id,
                    severity=emergency_signal.severity,
                    trigger_phrase=emergency_signal.trigger_phrase,
                    reasoning=emergency_signal.reasoning,
                    user_affirmation=user_utterance,
                    pending_resolution=True,
                    commit=False,
                )
                context_data["active_emergency"] = emergency_signal.model_dump()
                session.context_metadata = json.dumps(context_data)
                session.current_state = ConversationState.ACTIVE_CONVERSATION

                # Persist messages atomically
                user_msg = ConversationService.add_message(
                    db,
                    target_user.id,
                    session.id,
                    MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="emergency"),
                    commit=False,
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
                    commit=False,
                )
                db.commit()
                db.refresh(user_msg)
                db.refresh(assistant_msg)
            except Exception:
                db.rollback()
                raise

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
                        "pending_resolution": True,
                        "user_affirmation": emergency_signal.user_affirmation,
                    },
                    requires_confirmation=False,
                ),
                emergency_signal=emergency_signal,
                session_state=session.current_state.value,
                active_emergency=True,
                pending_resolution=True,
                recorded_locally=True,
                notification_attempted=False,
                notification_delivered=False,
            )

        # --- BRANCH B: User Confirmed Safe After Ordinary Uncertain Check ---
        if emergency_signal.is_safe_confirmed and not has_durable_active_emergency:
            logger.info("User %d confirmed safe. Standing down uncertain welfare check.", target_user.id)
            context_data.pop("active_emergency", None)
            session.context_metadata = json.dumps(context_data)
            session.current_state = ConversationState.ACTIVE_CONVERSATION

            try:
                user_msg = ConversationService.add_message(
                    db,
                    target_user.id,
                    session.id,
                    MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="greeting"),
                    commit=False,
                )
                spoken_resp = self._get_localized_safe_confirmation_response(pref_lang)
                assistant_msg = ConversationService.add_message(
                    db,
                    target_user.id,
                    session.id,
                    MessageCreate(role=MessageRole.ASSISTANT, content=spoken_resp, intent_detected="general_chat"),
                    commit=False,
                )
                db.commit()
                db.refresh(user_msg)
                db.refresh(assistant_msg)
            except Exception:
                db.rollback()
                raise

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
                recorded_locally=False,
                notification_attempted=False,
                notification_delivered=False,
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
            # Update emergency signal to reflect local persistence & honest delivery state
            emergency_signal = emergency_signal.model_copy(
                update={
                    "recorded_locally": True,
                    "notification_attempted": False,
                    "notification_delivered": False,
                }
            )

            try:
                EmergencyService.record_active_emergency(
                    db,
                    target_user.id,
                    severity=emergency_signal.severity,
                    trigger_phrase=emergency_signal.trigger_phrase,
                    reasoning=emergency_signal.reasoning,
                    commit=False,
                )
                context_data["active_emergency"] = emergency_signal.model_dump()
                session.context_metadata = json.dumps(context_data)
                session.current_state = ConversationState.ACTIVE_CONVERSATION

                user_msg = ConversationService.add_message(
                    db,
                    target_user.id,
                    session.id,
                    MessageCreate(role=MessageRole.USER, content=user_utterance, intent_detected="emergency"),
                    commit=False,
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
                    commit=False,
                )
                db.commit()
                db.refresh(user_msg)
                db.refresh(assistant_msg)
            except Exception:
                db.rollback()
                raise

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
                recorded_locally=True,
                notification_attempted=False,
                notification_delivered=False,
            )

        # --- BRANCH D: Normal Conversational Flow (LLM Evaluated) ---
        # If the resident already has an unresolved active emergency in the database,
        # sending another non-emergency message does NOT silently clear the emergency!
        is_user_emergency_active = bool(has_durable_active_emergency or "active_emergency" in context_data)

        # 1. Fetch synthesized user context (profile, operating mode, preferences, sliding history)
        context = ContextEngine.get_user_context(db, target_user.id, history_window=10)

        # 2. Assemble prompt
        system_prompt, user_prompt = PromptBuilder.build_prompt(context, user_utterance)

        # 3. Invoke Local LLM Client
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

        # 4. Defensive Response Parsing
        parsed_response = ResponseParser.parse(
            raw_llm_content,
            fallback_language=pref_lang.value,
        )

        # 5. Unsupported query protection: Guard against hallucinating unverified personal data
        if parsed_response.intent == IntentType.REMINDER_QUERY:
            # Phase 4.2 has no connected reminder querying database
            # Ensure spoken response clearly states lack of access rather than inventing schedules
            if not any(
                kw in parsed_response.spoken_response.lower()
                for kw in ("cannot access", "do not have access", "not connected", "उपलब्ध नहीं", "তথ্য নেই")
            ):
                spoken_text = {
                    SupportedLanguage.HINDI: "मेरे पास इस समय आपके रिमाइंडर्स या शेड्यूल की जानकारी उपलब्ध नहीं है।",
                    SupportedLanguage.BENGALI: "এই মুহূর্তে আমার কাছে আপনার রিমাইন্ডার বা সময়সূচীর তথ্য উপলব্ধ নেই।",
                }.get(
                    pref_lang,
                    "I cannot access your reminders or schedule at this time because that service is not connected.",
                )
                parsed_response = ParsedLLMResponse(
                    intent=IntentType.REMINDER_QUERY,
                    confidence=parsed_response.confidence,
                    detected_language=parsed_response.detected_language,
                    spoken_response=spoken_text,
                    proposed_action=ProposedAction(action_type=ActionType.NONE),
                    raw_text=raw_llm_content,
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
            is_user_emergency_active = True

        # 8. Persist turn atomically in database
        try:
            user_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(role=MessageRole.USER, content=user_utterance),
                commit=False,
            )
            assistant_msg = ConversationService.add_message(
                db,
                target_user.id,
                session.id,
                MessageCreate(
                    role=MessageRole.ASSISTANT,
                    content=parsed_response.spoken_response,
                    intent_detected=parsed_response.intent.value,
                ),
                commit=False,
            )
            session.current_state = ConversationState.ACTIVE_CONVERSATION
            db.commit()
            db.refresh(user_msg)
            db.refresh(assistant_msg)
        except Exception:
            db.rollback()
            raise

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
            active_emergency=is_user_emergency_active,
            pending_resolution=(durable_emergency.status == "pending_resolution" if durable_emergency else False),
            recorded_locally=False,
            notification_attempted=False,
            notification_delivered=False,
        )
