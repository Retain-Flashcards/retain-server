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
import os

load_dotenv()


def _load_prompt() -> str:
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


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
        default_factory=_load_prompt,
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
