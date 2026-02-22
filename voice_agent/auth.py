"""
JWT authentication helpers for the voice agent WebSocket.

Verifies Supabase-issued JWTs and extracts the user ID.
"""

import logging

import jwt

from voice_agent.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def verify_token(token: str) -> str | None:
    """Decode a Supabase JWT and return the user ID (``sub`` claim).

    Returns ``None`` if the token is invalid, expired, or missing
    the ``sub`` claim.
    """
    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
        user_id = payload.get("sub")
        if not user_id:
            logger.warning("JWT missing 'sub' claim")
            return None
        return user_id
    except jwt.ExpiredSignatureError:
        logger.warning("JWT has expired")
        return None
    except jwt.InvalidTokenError as exc:
        logger.warning("Invalid JWT: %s", exc)
        return None
