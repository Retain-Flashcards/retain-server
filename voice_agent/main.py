"""
FastAPI application with WebSocket endpoint for the live voice agent.

Acts as a proxy between a client app and the Gemini Live API,
forwarding audio (binary frames) and control messages (JSON text frames).
"""

import json
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState

from voice_agent.config import get_settings, Settings
from voice_agent.gemini_session import GeminiSessionManager
from voice_agent.session_store import SessionStore
from voice_agent.auth import verify_token
from voice_agent.supabase import supabase_client, supabase_service_client
from voice_agent.card_manager import CardManager
from voice_agent.background import BackgroundTaskManager
from voice_agent.embeddings import EmbeddingService

# ── Logging setup ───────────────────────────────────────────────

settings = get_settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice Agent",
    description="Live voice AI agent proxy powered by the Gemini Live API",
    version="0.1.0",
)

# Allow the test client (opened from file:// or another port) to call HTTP endpoints
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health() -> dict:
    """Health-check endpoint."""
    return {"status": "ok", "model": settings.gemini_model}


@app.get("/auth-config")
async def auth_config() -> dict:
    """Return public Supabase config for client-side auth."""
    return {
        "supabase_url": settings.supabase_url,
        "supabase_anon_key": settings.supabase_key,
    }


# ── Embedding webhook ───────────────────────────────────────────

_embedding_service: EmbeddingService | None = None


def _get_embedding_service() -> EmbeddingService:
    """Lazily initialise the shared EmbeddingService."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service


@app.post("/embed-note")
async def embed_note(request: Request) -> JSONResponse:
    """Webhook endpoint to embed a single note on creation.

    Secured by a shared secret in the Authorization header.
    Expects a Supabase webhook payload: { "record": { ... } }
    """
    # ── Authenticate ────────────────────────────────────────────
    auth_header = request.headers.get("authorization", "")
    expected = f"Bearer {settings.embed_webhook_secret}"
    if not settings.embed_webhook_secret or auth_header != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # ── Parse payload ──────────────────────────────────────────
    body = await request.json()
    record = body.get("record")
    if not record:
        raise HTTPException(status_code=400, detail="Missing 'record' in payload")

    note_id = record.get("id")
    front = record.get("front_content", "") or ""
    back = record.get("back_content", "") or ""
    text = f"{front}\n\n{back}".strip()

    if not note_id or not text:
        raise HTTPException(status_code=400, detail="Note must have an id and content")

    # ── Embed & persist ────────────────────────────────────────
    svc = _get_embedding_service()
    embedding = svc.embed_note(text)

    supabase = await supabase_service_client()
    await supabase.table("notes").update(
        {"embedding": embedding}
    ).eq("id", note_id).execute()

    logger.info("Embedded note %s", note_id)
    return JSONResponse({"status": "ok", "note_id": note_id})


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    deck_id: Optional[str] = Query(default=None),
) -> None:
    """Bidirectional audio proxy between a client and Gemini Live.

    Query params
    ------------
    token : str
        Supabase JWT access token (required).  The ``sub`` claim
        is used as the user ID.
    deck_id : str, optional
        The ID of the deck to use for creating a new session.
    session_id : str, optional
        Pass an existing session ID to resume.  The DB row must
        contain a valid ``resume_handle``.
    """
    await websocket.accept()
    logger.info("Client WebSocket connected")

    # ── Authenticate ────────────────────────────────────────────
    if not token:
        await websocket.send_json({
            "type": "error",
            "message": "Missing 'token' query parameter",
        })
        await websocket.close(code=1008)
        return

    user_id = verify_token(token)
    if user_id is None:
        await websocket.send_json({
            "type": "error",
            "message": "Invalid or expired token",
        })
        await websocket.close(code=1008)
        return

    # Either session_id or deck_id is required
    if not session_id and not deck_id:
        await websocket.send_json({
            "type": "error",
            "message": "Missing 'session_id' or 'deck_id' query parameter",
        })
        await websocket.close(code=1008)
        return

    # Initialize supabase, session store, and Card Manager
    supabase = await supabase_client(token)

    # This feature is only for paid users
    user_record = await supabase.table('users').select('plan').eq('id', user_id).single().execute()
    if not user_record.data or user_record.data['plan'] != 'retain-pro':
        await websocket.send_json({
            "type": "error",
            "message": "This feature is only for paid users",
        })
        await websocket.close(code=1008)
        return

    session_store = SessionStore(supabase)
    bg = BackgroundTaskManager()


    logger.info("Authenticated user %s", user_id)

    session: GeminiSessionManager | None = None
    resume_handle: str | None = None

    try:

        # ── Load or create DB session row ───────────────────────
        if session_id:
            row = await session_store.get_session(session_id)
            if row is None:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Session {session_id} not found",
                })
                await websocket.close()
                return
            # Verify the session belongs to this user
            if row.get("uid") != user_id:
                await websocket.send_json({
                    "type": "error",
                    "message": "Session does not belong to this user",
                })
                await websocket.close(code=1008)
                return
            resume_handle = row.get("resume_handle")
            await session_store.update_status(session_id, "active")

            # Fetch deck ID from the session row
            deck_id = row.get('deck_id')

            logger.info("Resuming session %s (has_handle=%s)", session_id, bool(resume_handle))
        else:
            row = await session_store.create_session(user_id, deck_id)
            session_id = row["id"]
            logger.info("Created new session %s for user %s", session_id, user_id)

        # Set up the card manager
        card_manager = CardManager(settings, bg, supabase, deck_id, session_store, session_id)


        # ── Connect to Gemini ───────────────────────────────────
        session = GeminiSessionManager(websocket, settings, session_store, card_manager, bg)
        await session.connect(
            resume_handle=resume_handle,
            session_id=session_id,
        )

        # Tell the client about the audio format and session ID
        await websocket.send_json({
            "type": "session_config",
            "session_id": session_id,
            "input_sample_rate": settings.input_sample_rate,
            "output_sample_rate": settings.output_sample_rate,
            "chunk_size": settings.chunk_size,
            "audio_encoding": "pcm_s16le",
            "channels": 1,
        })

        # Start the background loop that reads Gemini responses
        session.start_receive_loop()

        # ── Client message loop ─────────────────────────────────
        while True:
            message = await websocket.receive()

            # Binary frame → raw audio
            if "bytes" in message and message["bytes"]:
                await session.send_audio(message["bytes"])

            # Text frame → JSON control message
            elif "text" in message and message["text"]:
                await _handle_control_message(session, message["text"])

    except (WebSocketDisconnect, RuntimeError):
        logger.info("Client disconnected")

    except Exception:
        logger.exception("WebSocket handler error")
        # Try to notify the client before closing
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Session terminated due to an unexpected error",
                })
            except Exception:
                pass

    finally:
        if session is not None:
            await session.disconnect()
        # Mark session completed if it wasn't paused
        if session_id and (session is None or session.state.value != "paused"):
            await session_store.close_session(session_id)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
        logger.info("WebSocket connection closed")


async def _handle_control_message(
    session: GeminiSessionManager, raw: str
) -> None:
    """Parse and dispatch a JSON control message from the client."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Received non-JSON text frame: %.100s", raw)
        await session._send_error("Invalid JSON in control message")
        return

    msg_type = data.get("type")

    if msg_type == "pause":
        await session.pause()
    elif msg_type == "resume":
        await session.resume()
    elif msg_type == "pause_and_disconnect":
        await session.pause()
        # The client wants to leave — close the WS cleanly.
        # The finally block in websocket_endpoint will NOT mark
        # the session completed because state is "paused".
        raise WebSocketDisconnect()
    else:
        logger.warning("Unknown control message type: %s", msg_type)
        await session._send_error(f"Unknown message type: {msg_type}")


# ── Entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "voice_agent.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
