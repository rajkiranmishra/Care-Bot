"""Prompt Builder constructing structured inference payloads for local LLMs."""

from datetime import datetime, timezone
import json
from typing import Optional, Tuple
from app.conversation.schemas import UserContextPayload
from app.database.models.user import OperatingMode, SupportedLanguage


JSON_SCHEMA_SPEC = """{
  "intent": "emergency | greeting | reminder_query | reminder_create | weather_query | health_checkin | general_chat | unknown",
  "confidence": 0.0 to 1.0,
  "detected_language": "en | hi | bn",
  "spoken_response": "The concise, natural response string to be read aloud to the user.",
  "proposed_action": {
    "action_type": "none | create_reminder | query_weather | trigger_emergency_alert | health_log",
    "parameters": {},
    "requires_confirmation": false
  }
}"""


class PromptBuilder:
    """Constructs prompt envelopes combining persona instructions, dialogue history, and JSON constraints.

    IMPORTANT ARCHITECTURAL & SECURITY NOTE:
    Prompt delimiters and role tagging are formatting conventions provided solely to guide
    the model's generation structure. In accordance with system security rules:
    PROMPT DELIMITERS ARE NOT A SECURITY MECHANISM.
    Untrusted input from users or model hallucinations can never bypass server-side
    action validation, parameter schema checking, or role authorization.
    """

    @classmethod
    def get_json_instruction(cls, language: SupportedLanguage) -> str:
        """Provide explicit JSON output constraint instruction."""
        return (
            "CRITICAL INSTRUCTION: You MUST respond ONLY with a single valid JSON object strictly matching "
            "this schema. Do not include markdown codeblocks (no ```json), do not include preambles, and do not "
            f"include postscripts.\nSCHEMA:\n{JSON_SCHEMA_SPEC}"
        )

    @classmethod
    def get_mode_exemplar(cls, mode: OperatingMode, language: SupportedLanguage) -> str:
        """Provide few-shot exemplars illustrating valid JSON for the specific operating mode."""
        if mode == OperatingMode.ELDERLY_CARE:
            if language == SupportedLanguage.HINDI:
                return (
                    'Example User: "मुझे शाम को 7 बजे अपनी दवा लेनी है"\n'
                    'Example Response: {"intent": "reminder_create", "confidence": 0.95, "detected_language": "hi", '
                    '"spoken_response": "जी, मैंने शाम 7 बजे के लिए आपकी दवा की याद दिलाना सेट कर दिया है।", '
                    '"proposed_action": {"action_type": "create_reminder", "parameters": {"title": "दवा लेना", "time": "19:00"}, "requires_confirmation": false}}'
                )
            elif language == SupportedLanguage.BENGALI:
                return (
                    'Example User: "আমাকে রাত ৮ টায় ওষুধ খাওয়ার কথা মনে করিয়ে দাও"\n'
                    'Example Response: {"intent": "reminder_create", "confidence": 0.95, "detected_language": "bn", '
                    '"spoken_response": "নিশ্চয়ই, রাত ৮টায় আপনার ওষুধ খাওয়ার রিমাইন্ডার সেট করে দিয়েছি।", '
                    '"proposed_action": {"action_type": "create_reminder", "parameters": {"title": "ওষুধ খাওয়া", "time": "20:00"}, "requires_confirmation": false}}'
                )
            return (
                'Example User: "Remind me to take my heart medicine at 8 pm"\n'
                'Example Response: {"intent": "reminder_create", "confidence": 0.98, "detected_language": "en", '
                '"spoken_response": "I have set a reminder for your heart medicine at 8:00 PM. Please take care.", '
                '"proposed_action": {"action_type": "create_reminder", "parameters": {"title": "heart medicine", "time": "20:00"}, "requires_confirmation": false}}'
            )
        else:
            # Working Professional Mode
            return (
                'Example User: "What is the weather today in Kolkata?"\n'
                'Example Response: {"intent": "weather_query", "confidence": 0.95, "detected_language": "en", '
                '"spoken_response": "Currently in Kolkata it is sunny and 28 degrees Celsius.", '
                '"proposed_action": {"action_type": "query_weather", "parameters": {"location": "Kolkata"}, "requires_confirmation": false}}'
            )

    @classmethod
    def build_prompt(
        cls,
        context: UserContextPayload,
        user_utterance: str,
        system_time: Optional[datetime] = None,
    ) -> Tuple[str, str]:
        """Assemble system prompt and user dialogue prompt.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        now = system_time or datetime.now(timezone.utc)
        time_str = now.strftime("%Y-%m-%d %H:%M:%S UTC (%A)")

        json_guidance = cls.get_json_instruction(context.preferred_language)
        few_shot = cls.get_mode_exemplar(context.operating_mode, context.preferred_language)

        system_prompt = (
            f"{context.system_instructions}\n\n"
            f"Current System Time: {time_str}\n\n"
            f"{json_guidance}\n\n"
            f"EXEMPLAR:\n{few_shot}"
        )

        # Build chronological dialogue history
        history_lines = []
        for msg in context.recent_messages[-8:]:
            role_tag = "Resident" if msg.role.value == "user" else "Assistant"
            history_lines.append(f"{role_tag}: {msg.content}")

        history_block = "\n".join(history_lines) if history_lines else "No previous dialogue in this session."

        user_prompt = (
            f"--- RECENT CONVERSATION HISTORY ---\n"
            f"{history_block}\n"
            f"--- CURRENT UTTERANCE ---\n"
            f"Resident: {user_utterance}\n"
            f"Assistant:"
        )

        return system_prompt, user_prompt
