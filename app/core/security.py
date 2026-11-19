"""Security utilities, OWASP-compliant password hashing, and signed access token management."""

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Dict, Optional

# OWASP Recommended Iteration Count for PBKDF2-HMAC-SHA256
PBKDF2_ITERATIONS = 600_000


class TokenSecurityError(Exception):
    """Base exception for authentication token errors."""
    pass


class InvalidTokenError(TokenSecurityError):
    """Token format, signature, or structure is invalid."""
    pass


class ExpiredTokenError(TokenSecurityError):
    """Token expiration timestamp has elapsed."""
    pass


def hash_password(password: str) -> str:
    """Hash password securely using PBKDF2-HMAC-SHA256 with 600,000 rounds and a 16-byte random salt.

    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    if not password:
        raise ValueError("Password cannot be empty")

    salt = secrets.token_bytes(16)
    derived_key = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${derived_key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a stored PBKDF2 hash using constant-time comparison."""
    if not plain_password or not hashed_password:
        return False

    try:
        parts = hashed_password.split("$")
        if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
            return False

        iterations = int(parts[1])
        salt = bytes.fromhex(parts[2])
        expected_hash = bytes.fromhex(parts[3])

        candidate_key = hashlib.pbkdf2_hmac(
            hash_name="sha256",
            password=plain_password.encode("utf-8"),
            salt=salt,
            iterations=iterations,
        )
        return hmac.compare_digest(candidate_key, expected_hash)
    except (ValueError, IndexError, TypeError):
        return False


def _b64_encode(data: bytes) -> str:
    """Encode bytes to URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64_decode(data_str: str) -> bytes:
    """Decode URL-safe base64 with appropriate padding."""
    padding = 4 - (len(data_str) % 4)
    if padding != 4:
        data_str += "=" * padding
    return base64.urlsafe_b64decode(data_str.encode("utf-8"))


def create_access_token(
    claims: Dict[str, Any],
    secret_key: str,
    expires_in_seconds: int = 3600,
) -> str:
    """Generate a cryptographically signed HMAC-SHA256 access token (RFC 7519 compliant)."""
    now = int(time.time())
    payload = {
        **claims,
        "iat": now,
        "exp": now + expires_in_seconds,
    }

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str, secret_key: str) -> Dict[str, Any]:
    """Validate signature and expiration of an access token, returning the claims payload."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise InvalidTokenError("Token structure must contain exactly three segments")

        header_b64, payload_b64, sig_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
        actual_sig = _b64_decode(sig_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise InvalidTokenError("Cryptographic signature mismatch")

        payload = json.loads(_b64_decode(payload_b64).decode("utf-8"))

        # Verify expiration
        exp = payload.get("exp")
        if not exp or not isinstance(exp, (int, float)):
            raise InvalidTokenError("Token missing valid expiration claim")

        if time.time() > exp:
            raise ExpiredTokenError("Token has expired")

        return payload
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise InvalidTokenError(f"Malformed token payload: {exc}") from exc
