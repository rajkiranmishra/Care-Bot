"""Tests for password hashing and security verification."""

import pytest
from app.core.security import hash_password, verify_password


def test_password_hashing_and_verification():
    """Verify PBKDF2 salted hash generates unique hashes and verifies correctly."""
    password = "CareBotSecretPassword123!"
    hashed_1 = hash_password(password)
    hashed_2 = hash_password(password)

    # Hashes must be unique due to random 16-byte salt
    assert hashed_1 != hashed_2
    assert hashed_1.startswith("pbkdf2_sha256$100000$")

    # Both must verify successfully
    assert verify_password(password, hashed_1) is True
    assert verify_password(password, hashed_2) is True


def test_wrong_password_fails_verification():
    """Verify incorrect passwords fail verification."""
    password = "CorrectPassword123"
    hashed = hash_password(password)

    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "invalid_hash_format") is False


def test_empty_password_raises_error():
    """Verify empty password raises ValueError."""
    with pytest.raises(ValueError):
        hash_password("")
