"""Unit and regression tests for deterministic emergency signal detection and silence policy."""

import math
import pytest
from app.intelligence.emergency import EmergencySignalDetector, normalize_utterance
from app.intelligence.schemas import (
    EmergencySeverity,
    EmergencySignal,
    WelfareEscalationPolicy,
)


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
    ]
    for phrase in benign_phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase)
        assert sig.severity in (EmergencySeverity.NORMAL, EmergencySeverity.NONE)
        assert sig.is_false_positive is True or sig.detected is False


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


# -----------------------------------------------------------------------------
# Regression Tests for Silence Invariant and Emergency Safety Refinements
# -----------------------------------------------------------------------------

def test_ordinary_silence_without_preceding_distress():
    """Silence alone must never establish a critical emergency, nor be treated as proof of safety."""
    sig = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=20.0,
        prior_state=None,
        is_awaiting_welfare_response=False,
    )
    assert sig.detected is False
    assert sig.severity == EmergencySeverity.NORMAL
    # Crucial invariant: absent response is NOT confirmation of safety
    assert sig.is_safe_confirmed is False
    assert sig.requires_immediate_escalation is False


def test_silence_following_potential_distress_signal():
    """An unanswered welfare check following potential distress follows configurable escalation."""
    policy = WelfareEscalationPolicy(welfare_timeout_seconds=20.0, unanswered_threshold=2)

    prior_uncertain = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.UNCERTAIN,
        trigger_phrase="dizzy",
        confidence=0.75,
        requires_confirmation=True,
    )

    # 1. Within allowable timeout (< 20s): remains UNCERTAIN, awaiting confirmation
    within_window = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=8.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=0,
    )
    assert within_window.severity == EmergencySeverity.UNCERTAIN
    assert within_window.requires_confirmation is True
    assert within_window.requires_immediate_escalation is False
    assert within_window.is_safe_confirmed is False

    # 2. Timeout reached on first check (unanswered_checks=0 -> attempt 1): remains UNCERTAIN for repeat prompt
    first_timeout = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=22.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=0,
    )
    assert first_timeout.severity == EmergencySeverity.UNCERTAIN
    assert first_timeout.unanswered_checks == 1
    assert first_timeout.requires_confirmation is True
    assert first_timeout.requires_immediate_escalation is False

    # 3. Second check also timed out (unanswered_checks=1 -> attempt 2 >= threshold): escalates to CRITICAL
    second_timeout = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=25.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=1,
    )
    assert second_timeout.severity == EmergencySeverity.CRITICAL
    assert second_timeout.requires_immediate_escalation is True
    assert second_timeout.requires_confirmation is False
    assert second_timeout.unanswered_checks == 2


def test_silence_following_confirmed_emergency():
    """Silence following a confirmed emergency (e.g. fall) must NEVER reset or downgrade status."""
    prior_fall = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.CRITICAL,
        trigger_phrase="i fell down",
        confidence=0.98,
        requires_immediate_escalation=True,
    )

    sig = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=45.0,
        prior_state=prior_fall,
        is_awaiting_welfare_response=True,
    )
    # Status maintained as CRITICAL; never cleared to safe
    assert sig.detected is True
    assert sig.severity == EmergencySeverity.CRITICAL
    assert sig.requires_immediate_escalation is True
    assert sig.is_safe_confirmed is False


def test_user_responding_that_they_are_safe():
    """User explicitly responding that they are safe resets distress and sets is_safe_confirmed=True."""
    safe_phrases = [
        "I am fine, don't worry",
        "I am okay, it was a false alarm",
        "Sab theek hai, koi baat nahi",
        "Main theek hoon",
        "Shob thik ache, ami thik achi",
    ]
    prior_distress = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.UNCERTAIN,
        requires_confirmation=True,
    )

    for phrase in safe_phrases:
        sig = EmergencySignalDetector.evaluate_utterance(phrase, prior_state=prior_distress)
        assert sig.detected is False
        assert sig.severity == EmergencySeverity.NORMAL
        assert sig.is_safe_confirmed is True
        assert sig.requires_immediate_escalation is False
        assert sig.requires_confirmation is False


def test_repeated_emergency_signals():
    """Consecutive emergency signals increment repeat_count and elevate confidence/severity."""
    # First fall signal
    sig1 = EmergencySignalDetector.evaluate_utterance("I fell down")
    assert sig1.repeat_count == 1
    assert sig1.severity == EmergencySeverity.CRITICAL

    # Repeated fall signal within dialogue
    sig2 = EmergencySignalDetector.evaluate_utterance("Help, I fell down!", prior_state=sig1)
    assert sig2.repeat_count == 2
    assert sig2.confidence == 1.0
    assert sig2.severity == EmergencySeverity.CRITICAL

    # Repeated uncertain malaise escalates to URGENT
    sig_u1 = EmergencySignalDetector.evaluate_utterance("I feel dizzy")
    assert sig_u1.severity == EmergencySeverity.UNCERTAIN

    sig_u2 = EmergencySignalDetector.evaluate_utterance("Everything is still spinning", prior_state=sig_u1)
    assert sig_u2.repeat_count == 2
    assert sig_u2.severity == EmergencySeverity.URGENT
    assert sig_u2.requires_immediate_escalation is True


def test_missing_or_invalid_timestamps():
    """evaluate_silence must fail safely when timestamps are missing, negative, NaN, or invalid."""
    invalid_timestamps = [None, -10.0, float("nan"), "invalid_str", {}]

    for val in invalid_timestamps:
        sig = EmergencySignalDetector.evaluate_silence(
            seconds_of_silence=val,  # type: ignore
            prior_state=None,
            is_awaiting_welfare_response=False,
        )
        assert sig.detected is False
        assert sig.severity == EmergencySeverity.NORMAL
        assert sig.is_safe_confirmed is False


def test_unavailable_notification_service():
    """Verify detector is decoupled from notification side-effects and functions without any notification service."""
    # EmergencySignalDetector must perform pure classification without attempting network calls or external side-effects
    signal = EmergencySignalDetector.evaluate_utterance("Please call an ambulance, I cannot breathe")
    assert signal.detected is True
    assert signal.severity == EmergencySeverity.CRITICAL
    assert signal.requires_immediate_escalation is True

    # The detector does not initiate any external notification: it only marks requires_immediate_escalation=True
    # for consumption by downstream authorized services.
    assert hasattr(signal, "requires_immediate_escalation")
