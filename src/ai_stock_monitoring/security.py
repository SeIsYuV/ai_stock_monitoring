from __future__ import annotations

from hashlib import sha256
import hmac


def hash_password(password: str) -> str:
    return sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    candidate_hash = hash_password(password)
    return hmac.compare_digest(candidate_hash, password_hash)
