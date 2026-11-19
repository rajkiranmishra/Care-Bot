"""Security utilities, password hashing, and authentication abstractions."""

import hashlib
import hmac
import os
import secrets
from typing import Tuple


def hash_password(password: str) -> str:
    """Hash a password securely using PBKDF2-HMAC-SHA256 with a cryptographically random salt.

    Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
    """
    if not password:
        raise ValueError("Password cannot be empty")

    salt = secrets.token_bytes(16)
    iterations = 100_000
    derived_key = hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=password.encode("utf-8"),
        salt=salt,
        iterations=iterations,
    )
    return f"pbkdf2_sha256${iterations}${salt.hex()}${derived_key.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a PBKDF2-HMAC-SHA256 stored hash string."""
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
