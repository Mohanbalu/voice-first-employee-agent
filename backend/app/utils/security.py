"""Security Utilities — Password Hashing and JWT Token Management.

Production-quality password security with Argon2id and PyJWT:
- Never stores plaintext passwords.
- Salted, memory-hard Argon2id key derivation.
- Signed, expiring JSON Web Tokens (HS256).
- Strict secret masking and zero leakage in logs or exceptions.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
import jwt

logger = logging.getLogger("app.security")

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum 32-byte secret for HS256 HMAC-SHA256
_DEFAULT_DEV_SECRET = "hcl-voice-agent-enterprise-secret-key-production-32-chars!!"
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or _DEFAULT_DEV_SECRET
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Argon2id password hasher with RFC 9106 recommended parameters
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)


def hash_password(password: str) -> str:
    """Hashes a plaintext password using Argon2id.

    Args:
        password: Raw password string.

    Returns:
        Salted Argon2id hash string.
    """
    if not password:
        raise ValueError("Password cannot be empty")
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plaintext password against an Argon2id hash.

    Args:
        plain_password: Raw password provided during login.
        hashed_password: Stored hash from the database.

    Returns:
        True if password matches, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return _hasher.verify(hashed_password, plain_password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception as exc:
        logger.error("Unexpected error during password verification: %s", type(exc).__name__)
        return False


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Creates and signs a JWT access token.

    Args:
        data: Dict of claims (sub, username, role, tenant_id, name).
        expires_delta: Optional custom lifetime. Default: ACCESS_TOKEN_EXPIRE_MINUTES.

    Returns:
        Encoded JWT string.
    """
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))

    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "iss": "hcl-voice-agent",
    })

    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decodes and validates a JWT access token.

    Args:
        token: Bearer JWT token string.

    Returns:
        Dict of claims from token payload.

    Raises:
        jwt.PyJWTError on invalid signature, expiration, or malformed structure.
    """
    return jwt.decode(
        token,
        JWT_SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        issuer="hcl-voice-agent",
        options={"require": ["exp", "sub", "tenant_id", "role"]},
    )
