# Voice Agent Client Specification

This document details how the frontend client should interact with the Gemini Live Python server backend over WebSockets.

## 1. Connection

The client connects to the server via a WebSocket proxy that handles authentication, session management, and bidirectional communication with the Gemini Live API.

**Endpoint:**
```
ws://<server-host>/ws
```

### Query Parameters

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `token` | `string` | **Yes** | A valid Supabase JWT access token. The `sub` claim is used as the user ID. |
| `deck_id` | `string` | No* | The ID of the deck to use for creating a new session. |
| `session_id` | `string` | No* | The ID of an existing session to resume. |

*\*Note: Either `deck_id` or `session_id` must be provided.*

---

## 2. Client-to-Server Messages

The client sends two types of messages to the server over the WebSocket connection:

### A. Raw Audio (Binary Frames)
User microphone input should be captured, processed into raw PCM audio chunks, and sent directly as **Binary Frames** (`ArrayBuffer`).
- **Format:** `pcm_s16le` (16-bit PCM, little-endian)
- **Sample Rate:** `16000 Hz` (Provided in the `session_config` event)
- **Channels:** `1` (Mono)

*Note: The server will silently drop incoming audio frames if the session is currently paused.*

### B. Control Messages (Text Frames - JSON)
The client can send JSON-formatted control messages as Text Frames to manage the session state.

#### 1. Pause Session
Pauses the Gemini AI generation and stops forwarding audio in both directions.
```json
{
  "type": "pause"
}
```

#### 2. Resume Session
Resumes an actively paused session.
```json
{
  "type": "resume"
}
```

#### 3. Pause & Disconnect
Pauses the session and cleanly closes the WebSocket connection, allowing for future resumption using the `session_id`.
```json
{
  "type": "pause_and_disconnect"
}
```

---

## 3. Server-to-Client Messages

The server sends messages back to the client as either Audio or JSON.

### A. AI Audio (Binary Frames)
When the Gemini model generates spoken output, the server forwards raw PCM audio chunks via **Binary Frames**. These should be scheduled for gapless playback on the client.
- **Format:** `pcm_s16le`
- **Sample Rate:** `24000 Hz` (Provided in the `session_config` event)
- **Channels:** `1`

### B. Events & Control (Text Frames - JSON)
The server sends various JSON events to inform the client of session state, configurations, interruptions, and tool execution.

#### 1. Session Configuration (`session_config`)
Sent immediately upon a successful connection. Provides essential settings for audio encoding.
```json
{
  "type": "session_config",
  "session_id": "uuid-string",
  "input_sample_rate": 16000,
  "output_sample_rate": 24000,
  "chunk_size": 1024,
  "audio_encoding": "pcm_s16le",
  "channels": 1
}
```

#### 2. AI Speaking Status (`turn_complete`)
Sent when the AI has finished its current spoken turn.
```json
{
  "type": "turn_complete"
}
```

#### 3. Interrupted by User (`interrupted`)
Sent when Voice Activity Detection (VAD) detects the user speaking while the AI is talking. The client should immediately stop/flush all currently playing AI audio.
```json
{
  "type": "interrupted"
}
```

#### 4. Session Paused (`paused`)
Sent confirming the session has been suspended.
```json
{
  "type": "paused",
  "session_id": "uuid-string",
  "resume_handle": "string | null"
}
```

#### 5. Session Resumed (`resumed`)
Sent confirming the session is active again.
```json
{
  "type": "resumed"
}
```

#### 6. Gemini Disconnected / Reconnected
Sent if the upstream connection to Gemini temporarily drops (and is awaiting auto-reconnection), or successfully reconnects.
```json
{
  "type": "gemini_disconnected",
  "message": "AI session paused. It will resume when you start talking."
}

{
  "type": "gemini_reconnected",
  "message": "AI session reconnected."
}
```

#### 7. Graceful Teardown Warning (`go_away`)
Sent if the upstream server needs to close the connection soon. The client should prepare to reconnect using the `session_id`.
```json
{
  "type": "go_away",
  "time_left": "duration-string",
  "resume_handle": "string"
}
```

#### 8. Usage Update (`usage_update`)
Sent periodically (e.g., every 3 seconds) to inform the client of the user's current voice session usage and remaining time.
```json
{
  "type": "usage_update",
  "total_minute_budget": 60,
  "minutes_used": 15,
  "minutes_left": 45,
  "next_refresh_date": "2026-03-01T00:00:00Z"
}
```

#### 9. Errors (`error`)
Sent for critical failures or misconfigurations. The WebSocket connection may be closed following an error.
```json
{
  "type": "error",
  "message": "Human-readable error description"
}
```

---

## 4. Tool Execution Events

The Gemini model is equipped with a set of tools to interact with the flashcard backend. When the model invokes a tool, the server executes it and emits helpful events to keep the client UI in sync.

#### 1. Tool Call Started (`tool_call`)
Emitted right before executing a tool on the backend. This can be used to show loading spinners on the UI.
```json
{
  "type": "tool_call",
  "name": "tool_function_name",
  "args": {
    "arg_1": "value"
  }
}
```

#### 2. Tool Result Completed (`tool_result`)
Emitted immediately after a tool finishes execution. The result contains payload data that you can use to render cards or update state on the client.
```json
{
  "type": "tool_result",
  "name": "tool_function_name",
  "result": { ...dynamic result payload... }
}
```

### Available Tools and `tool_result` Payloads

Below are the tools callable by Gemini, and the exact structure of the `result` object that the client will receive in the `tool_result` event:

#### A. Fetching Cards for Current Topic (`check_top_5_cards_current_topic`)
Fetches the next batch of (up to 5) due flashcards for the current topic.
**`result` shape:**
```json
{
  "topic": "Current Topic Name",
  "cards": [
    { 
      "id": 123, 
      "content": "Front \\n\\n Back format of the card" 
    }
  ]
}
```

#### B. Changing Topics (`search_cards_for_new_topic`)
Fetches a new batch of 5 cards when the user asks to switch to a different subject/topic.
**Arguments:** `topic` (string)
**`result` shape:**
```json
{
  "topic": "New Topic Searched",
  "cards": [
    { 
      "id": 123, 
      "content": "Front \\n\\n Back format of the card" 
    }
  ]
}
```

#### C. Submitting a Card Review (`submit_review`)
Grading a flashcard's difficulty based on the user's audio response. Evicts the card from the active server cache.
**Arguments:** `card_id` (integer), `difficulty` (string: "correct", "struggled", or "incorrect")
**`result` shape:**
```json
{
  "status": "submitted_in_background",
  "card_id": 123,
  "difficulty": "correct",
  "card": {
    "num_id": 123,
    "content": "...",
    "deck_id": "..."
  }
}
```
*(Note: `card` object is only included if the card was successfully found in the cache.)*

#### D. Skipping a Card (`skip_card_permanently`)
Permanently skips/suspends a card due to being unusable or duplicate.
**Arguments:** `card_id` (integer)
**`result` shape:**
```json
{
  "status": "skipped_in_background",
  "card_id": 123,
  "card": {
    "num_id": 123,
    ...
  }
}
```
