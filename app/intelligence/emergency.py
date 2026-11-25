"""Deterministic Pre-LLM Emergency Signal Detection Layer.

Architectural Rules (Phase 4.1 Refinement):
1. The emergency detection layer MUST execute BEFORE LLM inference.
2. Separation of Detection from Notification: EmergencySignalDetector strictly classifies and
   identifies signals. It DOES NOT contact emergency services, send SMS/push notifications,
   or perform external side-effects. Those actions belong to a separately authorized service.
3. Silence Safety Policy:
   - Silence must NEVER be interpreted as proof that the user is safe.
   - Silence alone must NOT automatically establish a critical emergency.
   - Distinct states: NORMAL, UNCERTAIN, and CRITICAL.
   - Unanswered welfare checks follow a configurable escalation policy considering prior signals,
     user vulnerability, and timeout thresholds.
   - Never treat an absent response as confirmation of safety.
"""

import math
import re
from typing import List, Optional, Pattern, Tuple
from app.intelligence.schemas import (
    EmergencySeverity,
    EmergencySignal,
    WelfareEscalationPolicy,
)


# -----------------------------------------------------------------------------
# 1. Normalization & Pre-processing Helpers
# -----------------------------------------------------------------------------

def normalize_utterance(text: str) -> str:
    """Normalize text for resilient ASR phonetic and typo matching."""
    if not text:
        return ""
    # Lowercase
    cleaned = text.lower().strip()
    # Collapse 3 or more consecutive identical characters into a single character
    # (e.g. 'heeeeeelp' -> 'help', 'bachaaao' -> 'bachao', while 'fell' with 2 l's remains 'fell')
    cleaned = re.sub(r"(.)\1{2,}", r"\1", cleaned)
    # Remove punctuation except word characters, whitespace, and Indic Unicode blocks
    cleaned = re.sub(r"[^\w\s\u0900-\u097F\u0980-\u09FF]", " ", cleaned)
    # Collapse multiple spaces
    return " ".join(cleaned.split())


# -----------------------------------------------------------------------------
# 2. Pattern Dictionaries
# -----------------------------------------------------------------------------

# Explicit affirmative safety confirmations (User confirms they are safe/well)
AFFIRMATIVE_SAFE_PATTERNS: List[Pattern] = [
    re.compile(r"\b(i\s+am\s+(fine|okay|ok|safe|alright)|i\s*m\s+(fine|okay|ok|safe|alright))\b"),
    re.compile(r"\b(false\s+alarm|don\s*t\s+worry|no\s+emergency|no\s+problem|i\s+don\s*t\s+need\s+help)\b"),
    re.compile(r"(सब\s+ठीक\s+है|मैं\s+ठीक\s+हूँ|चिंता\s+मत\s+करो|कोई\s+बात\s+नहीं|sab\s+theek\s+hai|chinta\s+mat\s+karo|main\s+theek\s+hoon)"),
    re.compile(r"(সব\s+ঠিক\s+আছে|আমি\s+ঠিক\s+আছি|চিন্তা\s+করবেন\s+না|shob\s+thik\s+ache|chinta\s+korben\s+na|ami\s+thik\s+achi)"),
]

# A. Critical Emergency Patterns (Immediate physical danger, severe trauma, acute collapse)
CRITICAL_PATTERNS: List[Tuple[Pattern, str]] = [
    # English Falls, Inability to get up, Acute trauma
    (re.compile(r"\b(i\s+(have\s+)?fell\s+(down|over)|i\s+fall\s+down|fallen\s+down|slipped\s+and\s+fell)\b"), "English: Physical fall detected"),
    (re.compile(r"\b(can\s*not|can\s*t|unable\s+to)\s+(get\s+up|stand\s+up|move|breathe|breath)\b"), "English: Inability to rise or breathe"),
    (re.compile(r"\b(chest\s+pain|heart\s+attack|pain\s+in\s+(my\s+)?chest|having\s+a\s+stroke)\b"), "English: Acute cardiovascular distress"),
    (re.compile(r"\b(call\s+(an\s+)?ambulance|call\s+(911|112|108)|call\s+emergency)\b"), "English: Explicit emergency service dispatch request"),
    (re.compile(r"\b(bleeding\s+(badly|profusely|heavily)|passed\s+out|unconscious)\b"), "English: Acute bleeding or loss of consciousness"),

    # Hindi (Devanagari & Romanized/Hinglish)
    (re.compile(r"(मैं\s+गिर\s+(गया|गई)|गिर\s+गया\s+हूँ|उठ\s+नहीं\s+पा\s+रहा|उठ\s+नहीं\s+सकती)"), "Hindi: Physical fall or unable to rise (Devanagari)"),
    (re.compile(r"\b(main\s+gir\s+(gaya|gayi)|gir\s+gaya\s+hoon|uth\s+nah?i\s+pa\s+rah?a|uth\s+nah?i\s+sakt[ai])\b"), "Hindi: Physical fall or unable to rise (Romanized)"),
    (re.compile(r"((सीने|छाती)\s+में\s+.*?दर्द|सांस\s+नहीं\s+आ\s+रही|बचाओ\s+मुझे|मुझे\s+बचाओ|\bबचाओ\b)"), "Hindi: Acute chest pain or respiration distress (Devanagari)"),
    (re.compile(r"\b((seene|chhati)\s+mein\s+.*?dard|saans\s+nah?i\s+aa\s+rah?i|bachao\s+mujhe|mujhe\s+bachao|\bbachao\b)\b"), "Hindi: Acute chest pain or respiration distress (Romanized)"),
    (re.compile(r"(एंबुलेंस\s+बुलाओ|एम्बुलेंस\s+बुलाओ|डॉक्टर\s+को\s+बुलाओ)"), "Hindi: Emergency service call (Devanagari)"),
    (re.compile(r"\b(ambulance\s+bulao|doctor\s+ko\s+bulao)\b"), "Hindi: Emergency service call (Romanized)"),

    # Bengali (Bengali Script & Romanized)
    (re.compile(r"(আমি\s+পড়ে\s+গেছি|পড়ে\s+গেছি|পড়ে\s+গিয়েছি|উঠতে\s+পারছি\s+না)"), "Bengali: Physical fall or unable to rise (Bengali script)"),
    (re.compile(r"\b(a?mi\s+pore\s+(gechi|geyechi)|pore\s+gechi|utht?e\s+parchi\s+na)\b"), "Bengali: Physical fall or unable to rise (Romanized)"),
    (re.compile(r"(বুকে\s+.*?ব্যথা|শ্বাস\s+নিতে\s+পারছি\s+না|কে\s+আছো\s+বাঁচাও|বাঁচাও\s+আমাকে|\bবাঁচাও\b)"), "Bengali: Acute chest pain, respiratory arrest, or distress (Bengali script)"),
    (re.compile(r"\b(buke\s+.*?by?atha|shash\s+nite\s+parchi\s+na|banchao\s+amake|ke\s+acho\s+banchao|\bbanchao\b)\b"), "Bengali: Acute chest pain or respiratory distress (Romanized)"),
    (re.compile(r"(অ্যাম্বুলেন্স\s+ডাকো|ডাক্তার\s+ডাকো)"), "Bengali: Emergency service dispatch (Bengali script)"),
    (re.compile(r"\b(ambulance\s+dako|doctor\s+dako)\b"), "Bengali: Emergency service dispatch (Romanized)"),
]

# B. Urgent Emergency Patterns (High concern, physical distress, acute need of immediate assistance)
URGENT_PATTERNS: List[Tuple[Pattern, str]] = [
    # English
    (re.compile(r"\b(please\s+help\s+me|somebody\s+help\s+me|i\s+need\s+urgent\s+help|emergency\s+help)\b"), "English: Urgent distress plea"),
    (re.compile(r"\b(i\s+can\s*t\s+get\s+out\s+of\s+bed|i\s+am\s+trapped|i\s+am\s+stuck)\b"), "English: Trapped or immobilized"),

    # Hindi
    (re.compile(r"(मदद\s+करो\s+मेरी|कोई\s+बचाओ|जल्दी\s+मदद\s+चाहिए)"), "Hindi: Urgent plea for help (Devanagari)"),
    (re.compile(r"\b(madad\s+karo\s+meri|koi\s+bachao|jaldi\s+madad\s+chahiye|urgent\s+help\s+chahiye)\b"), "Hindi: Urgent plea for help (Romanized)"),

    # Bengali
    (re.compile(r"(সাহায্য\s+করুন|আমাকে\s+সাহায্য\s+করুন|তাড়াতাড়ি\s+এসো)"), "Bengali: Urgent plea for help (Bengali script)"),
    (re.compile(r"\b(shah?arjyo\s+korun|amake\s+shah?arjyo\s+korun|taratari\s+esho)\b"), "Bengali: Urgent plea for help (Romanized)"),
]

# C. Uncertain / Ambiguous Distress Patterns (Requires active, immediate voice confirmation)
UNCERTAIN_PATTERNS: List[Tuple[Pattern, str]] = [
    # English
    (re.compile(r"\b(i\s+feel\s+(very\s+)?(dizzy|faint|sick|terrible|awful|unwell))\b"), "English: Acute dizziness or malaise"),
    (re.compile(r"\b(everything\s+is\s+(still\s+)?spinning|something\s+is\s+(very\s+)?wrong|i\s+don\s*t\s+feel\s+safe)\b"), "English: Sudden disorientation or distress"),
    (re.compile(r"\b(head\s+hurts\s+badly|severe\s+headache|blacking\s+out)\b"), "English: Severe headache or fading vision"),

    # Hindi
    (re.compile(r"(चक्कर\s+आ\s+रहा\s+है|तबीयत\s+बहुत\s+खराब\s+है|कुछ\s+सही\s+नहीं\s+लग\s+रहा)"), "Hindi: Dizziness or acute illness (Devanagari)"),
    (re.compile(r"\b(chakkar\s+aa\s+rah?a\s+hai|tabiyat\s+bahut\s+kharab\s+hai|kuch\s+sahi\s+nah?i\s+lag\s+rah?a)\b"), "Hindi: Dizziness or acute illness (Romanized)"),

    # Bengali
    (re.compile(r"(মাথা\s+(ঘুরছে|ঘোরাচ্ছে)|শরীর\s+খুব\s+খারাপ\s+লাগছে|অসুস্থ\s+বোধ\s+করছি)"), "Bengali: Dizziness or acute illness (Bengali script)"),
    (re.compile(r"\b(matha\s+(ghurche|ghorache)|shorir\s+(khub\s+)?kharap|osustho\s+lagche)\b"), "Bengali: Dizziness or acute illness (Romanized)"),
]

# D. Contextual False-Positive Suppressors (Conversational, recreational, or task-oriented)
FALSE_POSITIVE_INDICATORS: List[Pattern] = [
    re.compile(r"\b(crossword|puzzle|riddle|homework|assignment|recipe|game|movie|song|video)\b"),
    re.compile(r"\b(find\s+my|looking\s+for\s+my)\s+(glasses|spectacles|keys|remote|phone|book|shoes|slippers)\b"),
    re.compile(r"(चश्मा\s+खोज|चाबी\s+खोज|रिमोट\s+खोज|chash?ma\s+khoj|chabi\s+khoj)"),
    re.compile(r"(চশমা\s+খুঁজ|চাবি\s+খুঁজ|chosh?ma\s+khuj|chabi\s+khuj)"),
]


# -----------------------------------------------------------------------------
# 3. Core Detector Implementation
# -----------------------------------------------------------------------------

class EmergencySignalDetector:
    """Deterministic, standalone emergency pattern and distress detector.

    ARCHITECTURAL BOUNDARY:
    This class is strictly a classifier and evaluator. It DOES NOT dispatch notifications,
    call third-party APIs, or trigger hardware alerts. Downstream domain services consume
    its EmergencySignal and handle authorized dispatch.
    """

    @classmethod
    def evaluate_utterance(
        cls,
        raw_text: str,
        prior_state: Optional[EmergencySignal] = None,
    ) -> EmergencySignal:
        """Analyze user utterance before invoking downstream local LLMs."""
        if not raw_text or not raw_text.strip():
            return EmergencySignal(
                detected=False,
                severity=EmergencySeverity.NORMAL,
                confidence=0.0,
                reasoning="Empty utterance",
                is_safe_confirmed=False,
            )

        normalized = normalize_utterance(raw_text)

        # 1. Check for explicit affirmative safety confirmations
        for pattern in AFFIRMATIVE_SAFE_PATTERNS:
            if pattern.search(normalized):
                return EmergencySignal(
                    detected=False,
                    severity=EmergencySeverity.NORMAL,
                    trigger_phrase=pattern.pattern,
                    confidence=0.99,
                    reasoning="User explicitly and affirmatively confirmed they are safe and okay.",
                    is_safe_confirmed=True,
                    requires_immediate_escalation=False,
                    requires_confirmation=False,
                )

        # 2. Check for recreational/task false-positive suppressors
        is_recreational_context = any(pattern.search(normalized) for pattern in FALSE_POSITIVE_INDICATORS)

        # 3. Check for CRITICAL patterns (Falls, chest pain, call ambulance, cant breathe)
        for pattern, rationale in CRITICAL_PATTERNS:
            match = pattern.search(normalized)
            if match:
                repeat_count = (prior_state.repeat_count + 1) if (prior_state and prior_state.detected) else 1
                return EmergencySignal(
                    detected=True,
                    severity=EmergencySeverity.CRITICAL,
                    trigger_phrase=match.group(0),
                    confidence=1.0 if repeat_count > 1 else 0.98,
                    reasoning=f"{rationale} (repeated x{repeat_count})" if repeat_count > 1 else rationale,
                    requires_immediate_escalation=True,
                    requires_confirmation=False,
                    repeat_count=repeat_count,
                    is_safe_confirmed=False,
                )

        # 4. Check for URGENT patterns (Help pleas, trapped in bed)
        for pattern, rationale in URGENT_PATTERNS:
            match = pattern.search(normalized)
            if match:
                if is_recreational_context:
                    return EmergencySignal(
                        detected=False,
                        severity=EmergencySeverity.NORMAL,
                        trigger_phrase=match.group(0),
                        confidence=0.2,
                        reasoning=f"False positive context detected for urgent trigger '{match.group(0)}'",
                        is_false_positive=True,
                        is_safe_confirmed=False,
                    )
                repeat_count = (prior_state.repeat_count + 1) if (prior_state and prior_state.detected) else 1
                return EmergencySignal(
                    detected=True,
                    severity=EmergencySeverity.URGENT,
                    trigger_phrase=match.group(0),
                    confidence=0.95 if repeat_count > 1 else 0.90,
                    reasoning=rationale,
                    requires_immediate_escalation=True,
                    requires_confirmation=False,
                    repeat_count=repeat_count,
                    is_safe_confirmed=False,
                )

        # 5. Check for UNCERTAIN / AMBIGUOUS patterns (Dizziness, disorientation, weakness)
        for pattern, rationale in UNCERTAIN_PATTERNS:
            match = pattern.search(normalized)
            if match:
                if is_recreational_context:
                    return EmergencySignal(
                        detected=False,
                        severity=EmergencySeverity.NORMAL,
                        trigger_phrase=match.group(0),
                        confidence=0.15,
                        reasoning=f"False positive context detected for ambiguous trigger '{match.group(0)}'",
                        is_false_positive=True,
                        is_safe_confirmed=False,
                    )
                repeat_count = (prior_state.repeat_count + 1) if (prior_state and prior_state.detected) else 1
                # If ambiguous distress is repeated consecutively, escalate severity to URGENT
                severity = EmergencySeverity.URGENT if repeat_count > 1 else EmergencySeverity.UNCERTAIN
                return EmergencySignal(
                    detected=True,
                    severity=severity,
                    trigger_phrase=match.group(0),
                    confidence=0.85 if repeat_count > 1 else 0.75,
                    reasoning=f"{rationale} (repeated x{repeat_count})" if repeat_count > 1 else rationale,
                    requires_immediate_escalation=(severity == EmergencySeverity.URGENT),
                    requires_confirmation=(severity == EmergencySeverity.UNCERTAIN),
                    repeat_count=repeat_count,
                    is_safe_confirmed=False,
                )

        # 6. Default: Normal utterance without emergency markers
        # Note: Absence of distress is NOT confirmation of safety (is_safe_confirmed=False)
        return EmergencySignal(
            detected=False,
            severity=EmergencySeverity.NORMAL,
            confidence=0.0,
            reasoning="No emergency signal detected",
            is_safe_confirmed=False,
        )

    @classmethod
    def evaluate_silence(
        cls,
        seconds_of_silence: Optional[float],
        prior_state: Optional[EmergencySignal] = None,
        is_awaiting_welfare_response: bool = False,
        policy: Optional[WelfareEscalationPolicy] = None,
        unanswered_checks: int = 0,
    ) -> EmergencySignal:
        """Evaluate silence using configurable escalation policy and prior context.

        Required Invariants:
        1. Silence must NEVER be interpreted as proof that the user is safe (is_safe_confirmed=False).
        2. Silence alone without preceding distress must NOT establish a critical emergency.
        3. Distinguishes NORMAL, UNCERTAIN, and CRITICAL states.
        4. Escalation follows configurable WelfareEscalationPolicy based on elapsed time,
           user vulnerability multiplier, and consecutive unanswered check thresholds.
        5. Defensively handles missing, negative, NaN, or non-numeric timestamps.
        """
        # Defensive timestamp sanitization
        try:
            if seconds_of_silence is None:
                seconds = 0.0
            else:
                seconds = float(seconds_of_silence)
                if math.isnan(seconds) or seconds < 0.0:
                    seconds = 0.0
        except (ValueError, TypeError):
            seconds = 0.0

        escalation_policy = policy or WelfareEscalationPolicy()
        effective_timeout = escalation_policy.welfare_timeout_seconds / escalation_policy.user_vulnerability_multiplier

        has_prior_distress = prior_state is not None and prior_state.detected and prior_state.severity not in (
            EmergencySeverity.NORMAL,
            EmergencySeverity.NONE,
        )

        # Scenario A: Ordinary silence without preceding distress or pending welfare check
        if not is_awaiting_welfare_response and not has_prior_distress:
            return EmergencySignal(
                detected=False,
                severity=EmergencySeverity.NORMAL,
                reasoning="Ordinary conversational silence without preceding distress. Silence is not treated as proof of safety.",
                is_safe_confirmed=False,
            )

        # Scenario B: Silence following a CONFIRMED emergency (CRITICAL or URGENT)
        # If the user previously experienced a fall or chest pain, silence DOES NOT downgrade the emergency.
        if has_prior_distress and prior_state.severity in (EmergencySeverity.CRITICAL, EmergencySeverity.URGENT):
            return EmergencySignal(
                detected=True,
                severity=prior_state.severity,
                trigger_phrase=prior_state.trigger_phrase,
                confidence=prior_state.confidence,
                reasoning=(
                    f"Persistent silence ({seconds:.1f}s) following confirmed {prior_state.severity.value} emergency. "
                    "Safety invariant: silence does not reset emergency status; escalation maintained."
                ),
                requires_immediate_escalation=True,
                requires_confirmation=False,
                repeat_count=prior_state.repeat_count,
                is_safe_confirmed=False,
                unanswered_checks=unanswered_checks,
            )

        # Scenario C: Silence following an UNCERTAIN distress trigger or pending welfare check
        # A welfare check was asked, or an ambiguous distress signal was previously recorded.
        if seconds < effective_timeout:
            # Within allowable response window: remain in UNCERTAIN state awaiting response
            return EmergencySignal(
                detected=True,
                severity=EmergencySeverity.UNCERTAIN,
                trigger_phrase=prior_state.trigger_phrase if prior_state else "[AWAITING_WELFARE_RESPONSE]",
                confidence=0.70,
                reasoning=f"Awaiting response to welfare check ({seconds:.1f}s elapsed of {effective_timeout:.1f}s timeout)",
                requires_immediate_escalation=False,
                requires_confirmation=True,
                is_safe_confirmed=False,
                unanswered_checks=unanswered_checks,
            )

        # Timeout has elapsed! Determine escalation based on unanswered count and prior distress
        next_unanswered = unanswered_checks + 1

        if next_unanswered < escalation_policy.unanswered_threshold:
            # First unanswered prompt: remain UNCERTAIN, prompt user with higher volume/urgency
            return EmergencySignal(
                detected=True,
                severity=EmergencySeverity.UNCERTAIN,
                trigger_phrase="[UNANSWERED_WELFARE_CHECK]",
                confidence=0.85,
                reasoning=(
                    f"Welfare check unanswered after {seconds:.1f}s (attempt {next_unanswered} of "
                    f"{escalation_policy.unanswered_threshold}). Prompting again before escalation."
                ),
                requires_immediate_escalation=False,
                requires_confirmation=True,
                is_safe_confirmed=False,
                unanswered_checks=next_unanswered,
            )

        # Unanswered threshold reached!
        # If there was a preceding distress signal (even ambiguous), escalate to CRITICAL
        if has_prior_distress:
            return EmergencySignal(
                detected=True,
                severity=EmergencySeverity.CRITICAL,
                trigger_phrase="[UNRESPONSIVE_AFTER_DISTRESS]",
                confidence=0.98,
                reasoning=(
                    f"User unresponsive following prior distress after {next_unanswered} welfare prompts "
                    f"({seconds:.1f}s elapsed). Escalated to CRITICAL in accordance with escalation policy."
                ),
                requires_immediate_escalation=True,
                requires_confirmation=False,
                is_safe_confirmed=False,
                unanswered_checks=next_unanswered,
            )

        # If there was NO prior distress signal, but welfare check was somehow unanswered
        # (e.g. routine scheduled check-in), respect policy.auto_escalate_to_critical:
        if escalation_policy.auto_escalate_to_critical:
            final_severity = EmergencySeverity.CRITICAL
            immediate_escalate = True
        else:
            final_severity = EmergencySeverity.URGENT
            immediate_escalate = True

        return EmergencySignal(
            detected=True,
            severity=final_severity,
            trigger_phrase="[UNRESPONSIVE_ROUTINE_CHECK]",
            confidence=0.80,
            reasoning=(
                f"Welfare check timed out without response across {next_unanswered} checks ({seconds:.1f}s elapsed). "
                f"Escalated to {final_severity.value} without assuming safety."
            ),
            requires_immediate_escalation=immediate_escalate,
            requires_confirmation=False,
            is_safe_confirmed=False,
            unanswered_checks=next_unanswered,
        )
