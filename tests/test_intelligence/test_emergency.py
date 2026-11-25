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
# Regression Tests for Silence Invariant, Escalation Policy & Safety Checks
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
    policy = WelfareEscalationPolicy(
        welfare_timeout_seconds=20.0,
        unanswered_threshold=2,
        auto_escalate_to_critical=False,
    )

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

    # 2. Timeout reached on first check (attempt 1 < threshold=2): remains UNCERTAIN for repeat prompt
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

    # 3. Second check also timed out (attempt 2 >= threshold=2): escalates to URGENT (auto_escalate_to_critical=False)
    second_timeout = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=25.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=1,
    )
    assert second_timeout.severity == EmergencySeverity.URGENT
    assert second_timeout.requires_immediate_escalation is True
    assert second_timeout.requires_confirmation is False
    assert second_timeout.unanswered_checks == 2


def test_escalation_configuration_auto_escalate_to_critical_toggle():
    """Test explicit behavior of auto_escalate_to_critical toggle (False -> URGENT vs True -> CRITICAL)."""
    prior_uncertain = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.UNCERTAIN,
        trigger_phrase="not feeling well",
        requires_confirmation=True,
    )

    # Policy A: auto_escalate_to_critical = False
    policy_urgent = WelfareEscalationPolicy(
        welfare_timeout_seconds=15.0,
        unanswered_threshold=1,
        auto_escalate_to_critical=False,
    )
    sig_urgent = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=20.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy_urgent,
        unanswered_checks=0,
    )
    assert sig_urgent.severity == EmergencySeverity.URGENT
    assert sig_urgent.requires_immediate_escalation is True
    assert "auto_escalate_to_critical=False" in sig_urgent.reasoning

    # Policy B: auto_escalate_to_critical = True
    policy_critical = WelfareEscalationPolicy(
        welfare_timeout_seconds=15.0,
        unanswered_threshold=1,
        auto_escalate_to_critical=True,
    )
    sig_critical = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=20.0,
        prior_state=prior_uncertain,
        is_awaiting_welfare_response=True,
        policy=policy_critical,
        unanswered_checks=0,
    )
    assert sig_critical.severity == EmergencySeverity.CRITICAL
    assert sig_critical.requires_immediate_escalation is True
    assert "auto_escalate_to_critical=True" in sig_critical.reasoning


def test_escalation_configuration_unanswered_threshold_and_multiplier():
    """Verify policy respects higher unanswered_threshold (e.g. 3) and user_vulnerability_multiplier."""
    # Multiplier 2.0 halves the effective timeout from 30.0s to 15.0s
    policy = WelfareEscalationPolicy(
        welfare_timeout_seconds=30.0,
        unanswered_threshold=3,
        user_vulnerability_multiplier=2.0,
        auto_escalate_to_critical=True,
    )
    prior_state = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.UNCERTAIN,
        requires_confirmation=True,
    )

    # 12 seconds: < effective timeout of 15s -> still awaiting
    s1 = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=12.0,
        prior_state=prior_state,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=0,
    )
    assert s1.severity == EmergencySeverity.UNCERTAIN
    assert s1.unanswered_checks == 0

    # 16 seconds: > effective timeout (15s), but attempt 1 < threshold 3 -> remains UNCERTAIN
    s2 = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=16.0,
        prior_state=prior_state,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=0,
    )
    assert s2.severity == EmergencySeverity.UNCERTAIN
    assert s2.unanswered_checks == 1

    # Attempt 2 < threshold 3 -> remains UNCERTAIN
    s3 = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=16.0,
        prior_state=prior_state,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=1,
    )
    assert s3.severity == EmergencySeverity.UNCERTAIN
    assert s3.unanswered_checks == 2

    # Attempt 3 >= threshold 3 -> escalates to CRITICAL
    s4 = EmergencySignalDetector.evaluate_silence(
        seconds_of_silence=16.0,
        prior_state=prior_state,
        is_awaiting_welfare_response=True,
        policy=policy,
        unanswered_checks=2,
    )
    assert s4.severity == EmergencySeverity.CRITICAL
    assert s4.unanswered_checks == 3


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


def test_affirmative_response_after_uncertain_distress_clears_safely():
    """User responding that they are safe after an UNCERTAIN check clears to NORMAL."""
    prior_uncertain = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.UNCERTAIN,
        requires_confirmation=True,
    )

    sig = EmergencySignalDetector.evaluate_utterance(
        "I am fine, don't worry, it was a false alarm",
        prior_state=prior_uncertain,
    )
    assert sig.detected is False
    assert sig.severity == EmergencySeverity.NORMAL
    assert sig.is_safe_confirmed is True
    assert sig.pending_resolution is False
    assert sig.user_affirmation == "I am fine, don't worry, it was a false alarm"


def test_affirmative_response_after_confirmed_critical_emergency():
    """CRITICAL SAFETY INVARIANT: User saying 'I am fine' must NOT automatically clear a confirmed CRITICAL emergency!

    An elderly person who has fallen may speak reflexively while in danger.
    Confirmed emergencies require an authorized resolution procedure before clearing.
    """
    confirmed_fall = EmergencySignal(
        detected=True,
        severity=EmergencySeverity.CRITICAL,
        trigger_phrase="I fell down and can't get up",
        confidence=0.98,
        requires_immediate_escalation=True,
    )

    # User subsequently says "I am fine"
    sig = EmergencySignalDetector.evaluate_utterance(
        "I am fine, I'm okay",
        prior_state=confirmed_fall,
    )

    # Critical emergency is NOT cleared!
    assert sig.detected is True
    assert sig.severity == EmergencySeverity.CRITICAL
    assert sig.requires_immediate_escalation is True
    assert sig.is_safe_confirmed is False
    assert sig.pending_resolution is True
    assert sig.user_affirmation == "I am fine, I'm okay"
    assert "cannot clear an active CRITICAL emergency" in sig.reasoning


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
    signal = EmergencySignalDetector.evaluate_utterance("Please call an ambulance, I cannot breathe")
    assert signal.detected is True
    assert signal.severity == EmergencySeverity.CRITICAL
    assert signal.requires_immediate_escalation is True
    assert hasattr(signal, "requires_immediate_escalation")
