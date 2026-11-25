"""Deterministic Pre-LLM Emergency Signal Detection Layer.

Architectural Rule (Phase 4.1):
The emergency detection layer MUST execute BEFORE LLM inference.
It is deterministic, resilient to network/LLM downtime, and independent of model availability.

Design Principles:
1. Multilingual: Supports English, Hindi (Devanagari + Hinglish), and Bengali (Bengali script + Romanized).
2. False Positive Discrimination: Distinguishes actual medical/fall trauma from benign recreational
   uses (e.g. "help with crossword", "help find glasses", movie quotes).
3. Transcription Resilience: Normalizes phonetic elongations ("heeeelp", "bachaaao"), ASR drops, and typos.
4. Ambiguity / Uncertain Cases: Flags uncertain physical distress for immediate active confirmation.
5. Inviolable Silence Invariant: "Silence must never be interpreted as confirmation that the user is safe."
"""

import re
from typing import List, Optional, Pattern, Tuple
from app.intelligence.schemas import EmergencySeverity, EmergencySignal


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
# 2. Pattern Dictionaries: Critical, Urgent, and Uncertain
# -----------------------------------------------------------------------------

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
    (re.compile(r"\b(everything\s+is\s+spinning|something\s+is\s+(very\s+)?wrong|i\s+don\s*t\s+feel\s+safe)\b"), "English: Sudden disorientation or distress"),
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
    # Homework, puzzles, riddles, technology assistance
    re.compile(r"\b(crossword|puzzle|riddle|homework|assignment|recipe|game|movie|song|video)\b"),
    # Finding lost household items (glasses, keys, remote, book)
    re.compile(r"\b(find\s+my|looking\s+for\s+my)\s+(glasses|spectacles|keys|remote|phone|book|shoes|slippers)\b"),
    re.compile(r"(चश्मा\s+खोज|चाबी\s+खोज|रिमोट\s+खोज|chash?ma\s+khoj|chabi\s+khoj)"),
    re.compile(r"(চশমা\s+খুঁজ|চাবি\s+খুঁজ|chosh?ma\s+khuj|chabi\s+khuj)"),
    # Clear conversational negation: "I am okay, not in pain"
    re.compile(r"\b(i\s+am\s+(fine|okay|ok)|not\s+an\s+emergency|no\s+pain|don\s*t\s+worry|false\s+alarm)\b"),
    re.compile(r"(सब\s+ठीक\s+है|चिंता\s+मत\s+करो|कोई\s+बात\s+नहीं|sab\s+theek\s+hai|chinta\s+mat\s+karo)"),
    re.compile(r"(সব\s+ঠিক\s+আছে|চিন্তা\s+করবেন\s+না|shob\s+thik\s+ache|chinta\s+korben\s+na)"),
]


# -----------------------------------------------------------------------------
# 3. Core Detector Implementation
# -----------------------------------------------------------------------------

class EmergencySignalDetector:
    """Deterministic, standalone emergency pattern and distress detector."""

    @classmethod
    def evaluate_utterance(cls, raw_text: str) -> EmergencySignal:
        """Analyze user utterance before invoking downstream local LLMs."""
        if not raw_text or not raw_text.strip():
            return EmergencySignal(
                detected=False,
                severity=EmergencySeverity.NONE,
                confidence=0.0,
                reasoning="Empty utterance",
            )

        normalized = normalize_utterance(raw_text)

        # 1. Check for false-positive suppressors (e.g. crosswords, finding glasses)
        is_recreational_context = any(pattern.search(normalized) for pattern in FALSE_POSITIVE_INDICATORS)

        # 2. Check for CRITICAL patterns (Falls, chest pain, call ambulance, cant breathe)
        for pattern, rationale in CRITICAL_PATTERNS:
            match = pattern.search(normalized)
            if match:
                if is_recreational_context and re.search(r"\b(not\s+an\s+emergency|false\s+alarm|just\s+a\s+joke)\b", normalized):
                    return EmergencySignal(
                        detected=False,
                        severity=EmergencySeverity.NONE,
                        trigger_phrase=match.group(0),
                        confidence=0.1,
                        reasoning=f"False positive rejected: explicit non-emergency negation with trigger '{match.group(0)}'",
                        is_false_positive=True,
                    )

                return EmergencySignal(
                    detected=True,
                    severity=EmergencySeverity.CRITICAL,
                    trigger_phrase=match.group(0),
                    confidence=0.98,
                    reasoning=rationale,
                    requires_immediate_escalation=True,
                    requires_confirmation=False,
                )

        # 3. Check for URGENT patterns (Help pleas, trapped in bed)
        for pattern, rationale in URGENT_PATTERNS:
            match = pattern.search(normalized)
            if match:
                if is_recreational_context:
                    return EmergencySignal(
                        detected=False,
                        severity=EmergencySeverity.NONE,
                        trigger_phrase=match.group(0),
                        confidence=0.2,
                        reasoning=f"False positive context detected for urgent trigger '{match.group(0)}'",
                        is_false_positive=True,
                    )
                return EmergencySignal(
                    detected=True,
                    severity=EmergencySeverity.URGENT,
                    trigger_phrase=match.group(0),
                    confidence=0.90,
                    reasoning=rationale,
                    requires_immediate_escalation=True,
                    requires_confirmation=False,
                )

        # 4. Check for UNCERTAIN / AMBIGUOUS patterns (Dizziness, disorientation, weakness)
        for pattern, rationale in UNCERTAIN_PATTERNS:
            match = pattern.search(normalized)
            if match:
                if is_recreational_context:
                    return EmergencySignal(
                        detected=False,
                        severity=EmergencySeverity.NONE,
                        trigger_phrase=match.group(0),
                        confidence=0.15,
                        reasoning=f"False positive context detected for ambiguous trigger '{match.group(0)}'",
                        is_false_positive=True,
                    )
                return EmergencySignal(
                    detected=True,
                    severity=EmergencySeverity.UNCERTAIN,
                    trigger_phrase=match.group(0),
                    confidence=0.75,
                    reasoning=rationale,
                    requires_immediate_escalation=False,
                    requires_confirmation=True,
                )

        # 5. Default: No emergency signal detected
        return EmergencySignal(
            detected=False,
            severity=EmergencySeverity.NONE,
            confidence=0.0,
            reasoning="No emergency signal detected",
        )

    @classmethod
    def evaluate_silence(
        cls,
        seconds_of_silence: float,
        prior_state: Optional[EmergencySignal] = None,
        is_awaiting_welfare_response: bool = False,
    ) -> EmergencySignal:
        """Inviolable Safety Rule: Silence must NEVER be interpreted as confirmation of user safety.

        If a user has expressed uncertain or potential distress and then goes silent, or if
        an automated welfare check goes unanswered, this method escalates to CRITICAL.
        """
        if not is_awaiting_welfare_response and (not prior_state or not prior_state.detected):
            return EmergencySignal(
                detected=False,
                severity=EmergencySeverity.NONE,
                reasoning="Standard conversational silence without prior distress trigger",
            )

        if seconds_of_silence >= 10.0:
            return EmergencySignal(
                detected=True,
                severity=EmergencySeverity.CRITICAL,
                trigger_phrase="[UNRESPONSIVE_SILENCE]",
                confidence=0.95,
                reasoning=(
                    f"Silence violation ({seconds_of_silence:.1f}s): User became unresponsive following a "
                    "distress trigger or welfare check. In accordance with safety rules, silence is "
                    "never treated as safety and has been escalated to CRITICAL emergency status."
                ),
                requires_immediate_escalation=True,
                requires_confirmation=False,
            )

        return EmergencySignal(
            detected=True,
            severity=EmergencySeverity.UNCERTAIN,
            trigger_phrase="[AWAITING_RESPONSE_SILENCE]",
            confidence=0.70,
            reasoning=f"Awaiting response to welfare check ({seconds_of_silence:.1f}s elapsed)",
            requires_immediate_escalation=False,
            requires_confirmation=True,
        )
