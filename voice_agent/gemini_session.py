"""
Gemini Live API session manager.

Owns the bidirectional connection to Gemini, the pause/resume state
machine, the function-calling loop, and the per-session CardManager.
"""

import asyncio
import enum
import json
import logging
from typing import Any

import websockets.exceptions

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

from voice_agent.background import BackgroundTaskManager
from voice_agent.card_manager import CardManager
from voice_agent.config import Settings
from voice_agent.session_store import SessionStore
from voice_agent.tools import TOOL_DECLARATIONS, execute_tool

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    ACTIVE = "active"
    PAUSED = "paused"


class GeminiSessionManager:
    """Manages a single Gemini Live session for one WebSocket client.

    Parameters
    ----------
    websocket : WebSocket
        The client-facing FastAPI WebSocket.
    settings : Settings
        Application configuration.
    """

    def __init__(self, websocket: WebSocket, settings: Settings, session_store: SessionStore, card_manager: CardManager, bg: BackgroundTaskManager) -> None:
        self.ws = websocket
        self.settings = settings
        self.state = SessionState.ACTIVE
        self.session_id: str | None = None     # DB session row ID

        self._client = genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location
        )
        self._connect_cm: Any | None = None    # the context manager from .connect()
        self._session: Any | None = None        # the AsyncSession inside the CM
        self._closed = False                    # set True once Gemini WS is gone
        self._resume_handle: str | None = None  # latest resumption token
        self._reconnect_lock = asyncio.Lock()    # prevent concurrent reconnects
        self._bg = bg
        self._card_manager = card_manager
        self._receive_task: asyncio.Task[None] | None = None
        self._keepalive_task: asyncio.Task[None] | None = None
        self._session_store = session_store
        
        # Track AI context for aggressive reconnection recovery
        self._current_turn_text: list[str] = []
        self._last_ai_turn: str = ""

    # ── Lifecycle ───────────────────────────────────────────────

    async def connect(
        self,
        *,
        resume_handle: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Open the Gemini Live session.

        Parameters
        ----------
        resume_handle : str | None
            A Gemini resumption token from a previous session.
            Pass ``None`` to start a fresh session.
        session_id : str | None
            DB session row ID.  Passed through so the receive loop
            can persist resumption tokens.
        """
        self.session_id = session_id
        self._resume_handle = resume_handle

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.settings.gemini_voice,
                    )
                )
            ),
            enable_affective_dialog=True,
            output_audio_transcription=types.AudioTranscriptionConfig(),
            
            
            # Dynamically inject context recovery note if we are reconnecting
            system_instruction=(
                self.settings.system_instruction + (
                    f"\n\n[SYSTEM RECOVERY NOTE]: The connection just dropped right after you said the following to the user: "
                    f"'{self._last_ai_turn}'. DO NOT REPEAT YOURSELF. The user just heard you say that. Pick up the conversation "
                    f"naturally from exactly where that sentence left off."
                ) if self._last_ai_turn else self.settings.system_instruction
            ),
            
            tools=[types.Tool(function_declarations=TOOL_DECLARATIONS)],
            # ── Session resumption ──────────────────────────────
            session_resumption=types.SessionResumptionConfig(
                handle=resume_handle,  # None → new session
            ),
            # ── Context window compression (unlimited length) ──
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
        )

        self._connect_cm = self._client.aio.live.connect(
            model=self.settings.gemini_model,
            config=config,
        )
        self._session = await self._connect_cm.__aenter__()
        self._closed = False


        is_resumed = "resumed" if resume_handle else "new"
        logger.info(
            "Gemini Live session connected (model=%s, %s)",
            self.settings.gemini_model,
            is_resumed,
        )

        # Initialise the card manager in the background so the session
        # can start accepting audio immediately (only if not already initialized).
        if not self._card_manager.current_topic:
            self._bg.schedule(self._card_manager.initialize(), name="cache-init")

    async def disconnect(self) -> None:
        """Gracefully tear down the session and all background work."""
        self._closed = True

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass

        if self._keepalive_task and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass

        await self._bg.cancel_all()

        if self._connect_cm is not None:
            try:
                await self._connect_cm.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error closing Gemini session")
            self._connect_cm = None
            self._session = None

        logger.info("Gemini session disconnected")

    # ── Auto-reconnect ──────────────────────────────────────────

    async def _reconnect(self) -> None:
        """Tear down the old Gemini session and reconnect using the resume handle."""
        async with self._reconnect_lock:
            if not self._closed:
                return  # Another coroutine already reconnected

            logger.info("Reconnecting to Gemini (has_handle=%s)", bool(self._resume_handle))

            # Tear down old connection
            if self._receive_task and not self._receive_task.done():
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass

            if self._keepalive_task and not self._keepalive_task.done():
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass

            if self._connect_cm is not None:
                try:
                    await self._connect_cm.__aexit__(None, None, None)
                except Exception:
                    pass
                self._connect_cm = None
                self._session = None

            # Reconnect
            await self.connect(
                resume_handle=self._resume_handle,
                session_id=self.session_id,
            )
            self.start_receive_loop()

            await self._send_json({
                "type": "gemini_reconnected",
                "message": "AI session reconnected.",
            })
            logger.info("Gemini reconnected successfully")

    # ── Audio send path ─────────────────────────────────────────

    async def send_audio(self, data: bytes) -> None:
        """Forward a raw PCM audio chunk to Gemini.

        Silently drops the chunk if the session is paused.
        If the Gemini connection has dropped, auto-reconnects first.
        """
        if self.state is SessionState.PAUSED:
            return

        # Auto-reconnect if Gemini dropped
        if self._closed or self._session is None:
            try:
                await self._reconnect()
            except Exception:
                logger.exception("Failed to reconnect to Gemini")
                await self._send_error("Failed to reconnect to the AI model")
                return

        try:
            await self._session.send_realtime_input(
                audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000"),
            )
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosed) as e:
            logger.info("Gemini connection closed normally (%s %s) — will reconnect on next audio", e.code, e.reason)
            self._closed = True
        except Exception:
            logger.exception("Failed to send audio to Gemini")
            await self._send_error("Failed to send audio to the AI model")

    # ── Receive loop ────────────────────────────────────────────

    def start_receive_loop(self) -> None:
        """Start the background task that reads Gemini responses."""
        self._receive_task = asyncio.create_task(
            self._receive_loop(), name="gemini-receive"
        )
        self._keepalive_task = asyncio.create_task(
            self._keepalive_loop(), name="gemini-keepalive"
        )

    async def _keepalive_loop(self) -> None:
        """Send a silent audio frame every 3 seconds to prevent idle timeout."""
        try:
            while not self._closed:
                await asyncio.sleep(3)
                # If connected and active, send a ping to keep Gemini from dropping us
                if self._session is not None and self.state is SessionState.ACTIVE:
                    try:
                        # 0.1s of blank 16kHz 16-bit mono audio
                        await self._session.send_realtime_input(
                            audio=types.Blob(data=b'\x00' * 3200, mime_type="audio/pcm;rate=16000")
                        )
                    except Exception:
                        pass
        except asyncio.CancelledError:
            pass

    async def _receive_loop(self) -> None:
        """Continuously read from Gemini and forward to the client WS."""
        if self._session is None:
            return

        try:
            while not self._closed:
                turn = self._session.receive()
                logger.info('Receiving from Gemini')
                async for response in turn:
                    await self._handle_response(response)
                # Brief yield to avoid busy-waiting between turns
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosed) as e:
            # Gemini closed its WebSocket — mark as closed so send_audio
            # will trigger a reconnect when the user speaks again.
            logger.info("Gemini connection closed normally (%s %s) — will reconnect on next audio", e.code, e.reason)
            self._closed = True
            await self._send_json({
                "type": "gemini_disconnected",
                "message": "AI session paused. It will resume when you start talking.",
            })
        except WebSocketDisconnect:
            logger.info("Client disconnected during receive loop")
        except Exception:
            logger.exception("Receive loop encountered an error")
            await self._send_error("Session encountered an unexpected error")

    async def _handle_response(self, response: Any) -> None:
        """Route a single Gemini response to the appropriate handler."""

        # ── Session resumption token update ──────────────────────
        if response.session_resumption_update:
            update = response.session_resumption_update
            if update.resumable and update.new_handle:
                self._resume_handle = update.new_handle
                logger.debug("Received new resumption handle")
                # Persist to DB in the background
                if self.session_id:
                    self._bg.schedule(
                        self._session_store.update_handle(self.session_id, update.new_handle),
                        name="persist-handle",
                    )
            return

        # ── GoAway — server is about to close the connection ────
        if response.go_away is not None:
            time_left = str(response.go_away.time_left) if response.go_away.time_left else "unknown"
            logger.warning("GoAway received — %s remaining", time_left)
            await self._send_json({
                "type": "go_away",
                "time_left": time_left,
                "resume_handle": self._resume_handle,
            })
            return

        # ── Tool calls ──────────────────────────────────────────
        if response.tool_call:
            await self._handle_tool_calls(response.tool_call)
            return

        # ── Tool call cancellation ──────────────────────────────
        if response.tool_call_cancellation:
            cancelled_ids = [
                fc_id for fc_id in response.tool_call_cancellation.ids
            ]
            logger.info("Tool calls cancelled: %s", cancelled_ids)
            return

        # ── Server content ──────────────────────────────────────
        if response.server_content:
            sc = response.server_content

            # Interruption by VAD
            if sc.interrupted:
                logger.debug("Gemini generation interrupted by VAD")
                await self._send_json({"type": "interrupted"})
                return

            # Audio transcription (model spoken text)
            if sc.output_transcription and sc.output_transcription.text:
                self._current_turn_text.append(sc.output_transcription.text)

            # Model turn with audio data
            if sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and isinstance(part.inline_data.data, bytes):
                        # Drop audio if paused
                        if self.state is SessionState.PAUSED:
                            continue
                        try:
                            await self.ws.send_bytes(part.inline_data.data)
                        except WebSocketDisconnect:
                            raise
                        except Exception:
                            logger.exception("Failed to send audio to client")

            # Turn complete
            if sc.turn_complete:
                
                # Snapshot the accumulated text for context recovery
                if self._current_turn_text:
                    self._last_ai_turn = "".join(self._current_turn_text).strip()
                    logger.info("Snapshotted AI turn for recovery: %s", self._last_ai_turn)
                    self._current_turn_text.clear()
                else:
                    logger.debug("Model turn complete")
                    
                await self._send_json({"type": "turn_complete"})

    # ── Tool call handling ──────────────────────────────────────

    async def _handle_tool_calls(self, tool_call: Any) -> None:
        """Execute tool calls and feed responses back to Gemini."""
        function_responses: list[types.FunctionResponse] = []

        for fc in tool_call.function_calls:
            logger.info("Tool call: %s(%s)", fc.name, fc.args)

            # Notify client (informational)
            await self._send_json({
                "type": "tool_call",
                "name": fc.name,
                "args": fc.args or {},
            })

            # Execute
            result = await execute_tool(fc.name, fc.args or {}, self._card_manager)

            # Notify client (informational)
            await self._send_json({
                "type": "tool_result",
                "name": fc.name,
                "result": result,
            })

            # Prevent sending full card data back to Gemini to save tokens
            model_result = dict(result) if isinstance(result, dict) else result
            if isinstance(model_result, dict) and "card" in model_result:
                del model_result["card"]

            function_responses.append(
                types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": model_result},
                )
            )

        # Send all responses back to Gemini
        if function_responses and self._session is not None:
            try:
                await self._session.send_tool_response(
                    function_responses=function_responses,
                )
            except Exception:
                logger.exception("Failed to send tool responses to Gemini")
                await self._send_error("Failed to process tool results")

    # ── Pause / Resume ──────────────────────────────────────────

    async def pause(self) -> None:
        """Pause the session — stop forwarding audio in both directions."""
        if self.state is SessionState.PAUSED:
            # Idempotent
            await self._send_json({"type": "paused"})
            return

        self.state = SessionState.PAUSED

        if self._session is not None:
            try:
                await self._session.send_realtime_input(audio_stream_end=True)
            except Exception:
                logger.exception("Failed to send audio_stream_end on pause")

        # Persist paused status
        if self.session_id:
            self._bg.schedule(
                self._session_store.update_status(self.session_id, "paused"),
                name="persist-pause",
            )

        await self._send_json({
            "type": "paused",
            "session_id": self.session_id,
            "resume_handle": self._resume_handle,
        })
        logger.info("Session paused")

    async def resume(self) -> None:
        """Resume the session — re-enable audio forwarding."""
        if self.state is SessionState.ACTIVE:
            # Idempotent
            await self._send_json({"type": "resumed"})
            return

        self.state = SessionState.ACTIVE

        # Persist active status
        if self.session_id:
            self._bg.schedule(
                self._session_store.update_status(self.session_id, "active"),
                name="persist-resume",
            )

        await self._send_json({"type": "resumed"})
        logger.info("Session resumed")

    # ── Helpers ─────────────────────────────────────────────────

    async def _send_json(self, payload: dict[str, Any]) -> None:
        """Send a JSON text frame to the client, swallowing errors."""
        try:
            await self.ws.send_json(payload)
        except WebSocketDisconnect:
            raise
        except Exception:
            logger.exception("Failed to send JSON to client: %s", payload.get("type"))

    async def _send_error(self, message: str) -> None:
        """Send a client-safe error message."""
        await self._send_json({"type": "error", "message": message})
