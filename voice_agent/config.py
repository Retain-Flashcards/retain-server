"""
Configuration loader for the Voice Agent server.

All settings are loaded from environment variables or a .env file
using Pydantic BaseSettings. Required variables will raise a
validation error at startup if missing.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""
    model_config = SettingsConfigDict(
        extra="allow"
    )

    # ── Google Cloud / Vertex AI ────────────────────────────────
    gcp_project_id: str = Field(
        ...,
        description="Google Cloud Project ID for Vertex AI (required)",
    )
    gcp_location: str = Field(
        default="us-central1",
        description="Google Cloud region for Vertex AI",
    )
    gemini_model: str = Field(
        default="gemini-live-2.5-flash-native-audio",
        description="Gemini model identifier for the Live API",
    )
    gemini_voice: str = Field(
        default="Charon",
        description="Prebuilt voice name for speech output",
    )
    google_application_credentials: Optional[str] = Field(default=None, description='Path to service account key (optional)')

    # ── System prompt ───────────────────────────────────────────
    system_instruction: str = Field(
        default=(
            "You are a flashcard tutor guiding the user to actively recall concepts. "
            "Many cards have blanks like {{c<n>::word::hint}}. Do NOT read aloud the full card context before they answer. Probe the user conversationally.\n\n"
            "CRITICAL RULES:\n"
            "1. ACCURACY FIRST: You MUST hear the user EXPLICITLY RECALL EVERY SINGLE blanked word or key concept before submitting a review. NEVER assume they know a detail just because they got another part right.\n"
            "2. NO SPOILERS: NEVER reveal the exact answers or give obvious hints before the user guesses. Give the user a few turns/chances to come up with the answer before grading them.\n"
            "3. REVIEWS: Use the `submit_review` tool **ONLY AFTER** you've been able to **FULLY EVALUATE** how well the user knows a card. Grades: `correct` (easy recall), `struggled` (needed multiple hints/turns), `incorrect` (had to give up or say \"I don't know\" for ANY of the items in the card).\n"
            "  a. Try and get the user to say each key item in the card. This might take a few turns as they get one or two and then have to think and get you the others.\n"
            "  b. Note about grades: Correct = won't see again for potentially a long time, they're good on it. Struggled = they're okay to not see it again today, but should see it sooner than 'correct'. Incorrect = they should see it again TODAY so they can relearn it."
            "4. HANDLING 'I DONT KNOW': If the user simply gives up and says 'I don't know' or 'I'm not sure' ABOUT EVEN A SINGLE IMPORTANT POINT ON THE CARD, mark the card INCORRECT. However, if they say 'I'm not sure what you mean' or 'rephrase that', you should try to explain/hint instead of penalizing them.\n"
            "5. QUEUE MANAGEMENT: Check `check_top_5_cards_current_topic` frequently. If you see cards you've already discussed, you forgot to grade them! Grade them immediately.\n"
            "  a. This is a dynamic selection from a potentially massive list. You have no idea how many cards are left, you are done ONLY when you run check_top_5_cards_current_topic and you get 0 results.\n\n"
            "  b. If you check top 5 and get no results, let the user know the session is DONE"
            "6. TOOL CALLING: Do not add any parameters besides those stated in the tool definitions"
            "7. IMAGES: The user can't see card images. Do your best to give questions based on the text, if there isn't meaningful content without the image, you may skip that card."
            ""
            "IMPORTANT: Don't submit reviews until you've EXPLICITLY covered ALL its key points with the user and have a strong sense of their knowledge of it."
            "Start by enthusiastically asking if they're ready to review."
        ),
        description="System instruction / persona prompt for the model",
    )

    # ── Audio ───────────────────────────────────────────────────
    input_sample_rate: int = Field(
        default=16000,
        description="Expected input audio sample rate in Hz (16-bit PCM mono)",
    )
    output_sample_rate: int = Field(
        default=24000,
        description="Gemini output audio sample rate in Hz (16-bit PCM mono)",
    )
    chunk_size: int = Field(
        default=4096,
        description="Bytes per audio chunk forwarded over WebSocket",
    )

    # ── Database ──────────────────────────────────────────────────
    supabase_url: str = Field(
        ...,
        description="Supabase URL (required)",
    )
    supabase_key: str = Field(
        ...,
        description="Supabase key (required)",
    )
    supabase_jwt_secret: str = Field(
        ...,
        description="Supabase JWT secret for token verification (required)",
    )
    supabase_service_role_key: str = Field(
        default="",
        description="Supabase service role key for server-side writes (required for embed endpoint)",
    )

    # ── Embedding webhook ───────────────────────────────────────
    embed_webhook_secret: str = Field(
        default="",
        description="Shared secret for the /embed-note webhook endpoint",
    )

    # ── Server ──────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Uvicorn bind host")
    port: int = Field(default=8000, description="Uvicorn bind port")
    log_level: str = Field(default="info", description="Logging level")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


def get_settings() -> Settings:
    """
    Instantiate and return the application settings.

    Raises ``pydantic.ValidationError`` if any required variable
    (e.g. ``GEMINI_API_KEY``) is missing.
    """
    return Settings()
