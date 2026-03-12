from __future__ import annotations

"""Password hashing helpers for the single-admin login flow.

实现目标：
- 默认使用带随机盐的 PBKDF2，避免把管理员密码直接存成可快速撞库的摘要
- 兼容历史版本里已经写入 SQLite 的旧 SHA-256 哈希，避免升级后无法登录
- 在用户下一次成功登录后，可把旧哈希自动升级成更安全的新格式
"""

from base64 import urlsafe_b64encode
from hashlib import pbkdf2_hmac, sha256
import hmac
import secrets


_PBKDF2_ALGORITHM = "sha256"
_PBKDF2_ITERATIONS = 390_000
_PBKDF2_PREFIX = "pbkdf2_sha256"


def _pbkdf2_digest(password: str, salt: str, iterations: int) -> str:
    digest = pbkdf2_hmac(
        _PBKDF2_ALGORITHM,
        password.encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    )
    return urlsafe_b64encode(digest).decode("ascii")


def hash_password(password: str) -> str:
    """Hash the plain password before storing it in SQLite."""

    salt = secrets.token_urlsafe(16)
    digest = _pbkdf2_digest(password, salt, _PBKDF2_ITERATIONS)
    return f"{_PBKDF2_PREFIX}${_PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a login password using constant-time comparison.

    历史兼容：如果数据库里还是旧版纯 SHA-256 哈希，也允许登录，
    后续可以在成功登录后自动重写为更安全的 PBKDF2 格式。
    """

    if password_hash.startswith(f"{_PBKDF2_PREFIX}$"):
        try:
            _, iterations_text, salt, stored_digest = password_hash.split("$", 3)
            candidate_digest = _pbkdf2_digest(password, salt, int(iterations_text))
        except (TypeError, ValueError):
            return False
        return hmac.compare_digest(candidate_digest, stored_digest)

    candidate_hash = sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate_hash, password_hash)


def password_hash_needs_rehash(password_hash: str) -> bool:
    """Return whether the stored hash should be upgraded to the latest format."""

    return not password_hash.startswith(f"{_PBKDF2_PREFIX}$")
