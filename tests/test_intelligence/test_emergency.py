"""Unit and integration tests for deterministic emergency signal detection."""

import pytest
from app.intelligence.emergency import EmergencySignalDetector, normalize_utterance
from app.intelligence.schemas import EmergencySeverity


def test_normalize_utterance_phonetic_collapse():
    """Verify normalization collapses excessive repetitions and cleans punctuation."""
    assert normalize_utterance("Heeeeeelp!!!") == "help"
    assert normalize_utterance("bachaaaaao!!") == "bachao"
    assert normalize_utterance("  I   FELL   DOWN  ") == "i fell down"


def test_english_critical_emergencies():
    """Verify critical English emergency phrases are identified deterministically."""
    phrases = [
        "I fell down and can't get up",
        "Help me, I have fallen down",
        "I am having severe chest pain",
        "Please call an ambulance immediately",
        "I can't breathe",
        "He is bleeding heavily and passed out",
    ]
    for phrase in phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.detected is True, f"Failed for '{phrase}'"
        assert sig.severity == EmergencySeverity.CRITICAL, f"Wrong severity for '{phrase}': {sig.severity}"
        assert sig.requires_immediate_escalation is True
        assert sig.requires_confirmation is False


def test_hindi_critical_emergencies_devanagari_and_romanized():
    """Verify Hindi emergency detection in both Devanagari and Romanized forms."""
    phrases = [
        "मैं गिर गया हूँ और उठ नहीं पा रहा",
        "सीने में बहुत तेज़ दर्द हो रहा है",
        "मुझे सांस नहीं आ रही, बचाओ",
        "जल्दी एम्बुलेंस बुलाओ",
        "main gir gaya hoon aur uth nahi pa raha",
        "seene mein dard hai bahut",
        "mujhe bachao",
        "ambulance bulao jaldi",
    ]
    for phrase in phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.detected is True, f"Failed for '{phrase}'"
        assert sig.severity == EmergencySeverity.CRITICAL, f"Wrong severity for '{phrase}': {sig.severity}"
        assert sig.requires_immediate_escalation is True


def test_bengali_critical_emergencies_script_and_romanized():
    """Verify Bengali emergency detection in both Bengali script and Romanized forms."""
    phrases = [
        "আমি পড়ে গেছি, উঠতে পারছি না",
        "বুকে প্রচণ্ড ব্যথা করছে",
        "শ্বাস নিতে পারছি না, বাঁচাও",
        "অ্যাম্বুলেন্স ডাকো তাড়াতাড়ি",
        "ami pore gechi uthte parchi na",
        "buke prochondo byatha korche",
        "ke acho banchao amake",
        "ambulance dako taratari",
    ]
    for phrase in phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.detected is True, f"Failed for '{phrase}'"
        assert sig.severity == EmergencySeverity.CRITICAL, f"Wrong severity for '{phrase}': {sig.severity}"
        assert sig.requires_immediate_escalation is True


def test_transcription_error_and_elongation_resilience():
    """Verify detection resilience against ASR elongations and dropouts."""
    degraded_phrases = [
        "heeeelp me please i fell down",
        "bachaaao mujhe",
        "buke byatha uthte parchi naaa",
        "cant breath please call ambulance",
    ]
    for phrase in degraded_phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.detected is True, f"Failed for degraded phrase '{phrase}'"
        assert sig.severity == EmergencySeverity.CRITICAL


def test_false_positive_recreational_and_task_discrimination():
    """Verify recreational and benign idioms with 'help' are suppressed."""
    benign_phrases = [
        "Can you help me with my crossword puzzle?",
        "Help me find my glasses please",
        "Help me solve this math riddle",
        "Can you help me look for my remote?",
        "chashma khoj rahe hain madad karo",
        "choshma khujchi ektu sahajyo korun",
        "I am fine, not an emergency at all",
        "Don't worry, sab theek hai",
    ]
    for phrase in benign_phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert (
            sig.severity == EmergencySeverity.NONE or sig.is_false_positive is True
        ), f"False positive triggered for '{phrase}': {sig}"


def test_uncertain_distress_cases():
    """Verify uncertain malaise/dizziness triggers active confirmation rather than false silence."""
    uncertain_phrases = [
        "I feel very dizzy and faint",
        "Everything is spinning around me",
        "chakkar aa raha hai bohot zyada",
        "matha ghorache khub shorir kharap",
    ]
    for phrase in uncertain_phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.detected is True, f"Failed for uncertain phrase '{phrase}'"
        assert sig.severity == EmergencySeverity.UNCERTAIN
        assert sig.requires_confirmation is True
        assert sig.requires_immediate_escalation is False


def test_silence_safety_invariant_rule():
    """Inviolable Safety Rule: Silence must NEVER be interpreted as confirmation of user safety."""
    # 1. Normal conversational silence when no emergency was active
    idle_silence = EmergencySignalDetector.evaluate_silence(seconds_of_silence=15.0, is_awaiting_welfare_response=False)
    assert idle_silence.detected is False

    # 2. Silence following an uncertain distress signal or unanswered welfare check
    # Brief pause (< 10s): remains in uncertain state awaiting response
    brief_silence = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=4.0,
        is_awaiting_welfare_response=True,
    )
    assert brief_silence.detected is True
    assert brief_silence.severity == EmergencySeverity.UNCERTAIN
    assert brief_silence.requires_confirmation is True

    # Prolonged silence (>= 10s): MUST escalate to CRITICAL
    critical_silence = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=10.5,
        is_awaiting_welfare_response=True,
    )
    assert critical_silence.detected is True
    assert critical_silence.severity == EmergencySeverity.CRITICAL
    assert critical_silence.requires_immediate_escalation is True
    assert "silence is never treated as safety" in critical_silence.reasoning.lower()
