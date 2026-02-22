# ── Voice Agent Dockerfile ──────────────────────────────────────
# Deploys the voice_agent FastAPI/WebSocket server.
#
# Build:
#   docker build -t voice-agent .
#
# Run:
#   docker run -p 8000:8000 \
#     --env-file .env \
#     -v /path/to/google-service-account-key.json:/app/google-service-account-key.json:ro \
#     voice-agent
# ────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching
COPY voice_agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the voice_agent package
COPY voice_agent/ ./voice_agent/

EXPOSE 8000

# Run via the module entry point
CMD ["python", "-m", "voice_agent.main"]
