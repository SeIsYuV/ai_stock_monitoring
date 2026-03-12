from __future__ import annotations

"""Minimal password hashing helpers for the single-admin login flow."""

from hashlib import sha256
import hmac


def hash_password(password: str) -> str:
    """Hash the plain password before storing it in SQLite."""
    return sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a login password using constant-time comparison."""
    candidate_hash = hash_password(password)
    return hmac.compare_digest(candidate_hash, password_hash)
