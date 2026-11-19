"""Tests for password hashing, token creation, validation, and expiration."""

import time
import pytest
from app.core.security import (
    ExpiredTokenError,
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)

TEST_SECRET = "carebot-test-cryptographic-secret-key-32-chars!"


def test_password_hashing_and_verification():
    """Verify PBKDF2 salted hash generates unique hashes with 600,000 iterations and verifies correctly."""
    password = "CareBotSecretPassword123!"
    hashed_1 = hash_password(password)
    hashed_2 = hash_password(password)

    # Hashes must be unique due to random 16-byte salt
    assert hashed_1 != hashed_2
    assert hashed_1.startswith("pbkdf2_sha256$600000$")

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


def test_create_and_decode_access_token():
    """Verify standard access token round-trip decode."""
    claims = {"sub": "42", "username": "alice", "role": "resident"}
    token = create_access_token(claims, secret_key=TEST_SECRET, expires_in_seconds=300)

    decoded = decode_access_token(token, secret_key=TEST_SECRET)
    assert decoded["sub"] == "42"
    assert decoded["username"] == "alice"
    assert decoded["role"] == "resident"
    assert decoded["exp"] > time.time()


def test_tampered_token_rejected():
    """Verify cryptographic signature verification detects tampering."""
    token = create_access_token({"sub": "1"}, secret_key=TEST_SECRET, expires_in_seconds=300)
    parts = token.split(".")

    # Tamper with signature
    tampered_sig = parts[0] + "." + parts[1] + ".invalidSignatureB64"
    with pytest.raises(InvalidTokenError):
        decode_access_token(tampered_sig, secret_key=TEST_SECRET)

    # Decode with wrong secret
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key="wrong-secret-key-32-chars-long-padding!")


def test_expired_token_rejected():
    """Verify expired token raises ExpiredTokenError."""
    token = create_access_token({"sub": "1"}, secret_key=TEST_SECRET, expires_in_seconds=-10)
    with pytest.raises(ExpiredTokenError):
        decode_access_token(token, secret_key=TEST_SECRET)


def test_secret_key_minimum_length_enforced():
    """Verify creating or decoding tokens with short secrets (< 32 chars) raises ValueError."""
    short_secret = "too-short-secret"
    with pytest.raises(ValueError, match="at least 32 characters"):
        create_access_token({"sub": "1"}, secret_key=short_secret)

    valid_token = create_access_token({"sub": "1"}, secret_key=TEST_SECRET)
    with pytest.raises(ValueError, match="at least 32 characters"):
        decode_access_token(valid_token, secret_key=short_secret)


def test_unsupported_algorithm_rejected():
    """Verify algorithm confusion attacks (e.g. alg: none) are rejected."""
    import base64
    import json

    # Create token with "alg": "none"
    header = {"alg": "none", "typ": "JWT"}
    payload = {"sub": "1", "exp": time.time() + 300, "iss": "carebot-auth", "aud": "carebot-client"}
    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    p_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    none_token = f"{h_b64}.{p_b64}."

    with pytest.raises(InvalidTokenError, match="Unsupported algorithm"):
        decode_access_token(none_token, secret_key=TEST_SECRET)


def test_issuer_and_audience_verification():
    """Verify issuer and audience claims are checked during decode."""
    token = create_access_token(
        {"sub": "1"},
        secret_key=TEST_SECRET,
        issuer="carebot-auth",
        audience="carebot-client",
    )

    # Valid decode
    decoded = decode_access_token(token, secret_key=TEST_SECRET)
    assert decoded["iss"] == "carebot-auth"
    assert decoded["aud"] == "carebot-client"

    # Issuer mismatch
    with pytest.raises(InvalidTokenError, match="issuer mismatch"):
        decode_access_token(token, secret_key=TEST_SECRET, expected_issuer="foreign-issuer")

    # Audience mismatch
    with pytest.raises(InvalidTokenError, match="audience mismatch"):
        decode_access_token(token, secret_key=TEST_SECRET, expected_audience="wrong-audience")
