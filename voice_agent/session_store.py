"""
Supabase-backed session store for voice agent sessions.

Provides CRUD operations for the ``voice_sessions`` table.
The resumption handle is the Gemini-issued token that allows
reconnecting to an existing Live API session (valid for 2 hrs).

"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

TABLE = "voice_sessions"

class SessionStore:
    def __init__(self, supabase_client):
        self.supabase = supabase_client

    async def create_session(self, user_id: str, deck_id: str) -> dict[str, Any]:
        """Insert a new session row and return it."""
        result = (
            await self.supabase.table(TABLE)
            .insert({"uid": user_id, "deck_id": deck_id, "status": "active"})
            .execute()
        )
        row = result.data[0]
        logger.info("Created voice session %s for user %s", row["id"], user_id)
        return row

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Fetch a session by ID.  Returns ``None`` if not found."""
        result = (
            await self.supabase.table(TABLE)
            .select("*")
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )
        return result.data

    async def update_handle(self, session_id: str, handle: str) -> None:
        """Persist the latest Gemini resumption token."""
        await self.supabase.table(TABLE).update(
            {"resume_handle": handle, "updated_at": "now()"}
        ).eq("id", session_id).execute()
        logger.debug("Updated resume handle for session %s", session_id)

    async def update_status(self, session_id: str, status: str) -> None:
        """Set session status (active / paused / completed)."""
        await self.supabase.table(TABLE).update(
            {"status": status, "updated_at": "now()"}
        ).eq("id", session_id).execute()

    async def increment_cards_reviewed(self, session_id: str) -> None:
        """Bump the cards_reviewed counter by 1.

        Uses a raw RPC or read-modify-write since Supabase Python
        client doesn't support atomic increments directly.
        """
        row = await self.get_session(session_id)
        if row:
            new_count = (row.get("cards_reviewed") or 0) + 1
            await self.supabase.table(TABLE).update(
                {"cards_reviewed": new_count, "updated_at": "now()"}
            ).eq("id", session_id).execute()

    async def close_session(self, session_id: str, stats: dict[str, Any] | None = None) -> None:
        """Mark a session as completed, optionally storing final stats."""
        update_data: dict[str, Any] = {"status": "completed", "updated_at": "now()"}
        if stats:
            update_data.update(stats)
        await self.supabase.table(TABLE).update(update_data).eq("id", session_id).execute()
        logger.info("Closed voice session %s", session_id)
