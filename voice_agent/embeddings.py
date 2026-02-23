"""
Shared embedding service for note and query embeddings.

Uses Vertex AI via google.genai.Client with task-specific embedding types:
- RETRIEVAL_DOCUMENT for note content
- RETRIEVAL_QUERY for search queries
"""

import re
import logging

import numpy as np
from google.genai import Client, types

from voice_agent.config import Settings

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Strip cloze markup, HTML tags, images, and non-text parts from card text."""
    # Unwrap cloze deletions, keeping the answer text
    text = re.sub(r'{{c(\d)::(.+?)(?:(?:::)([^:]+)?)?}}', r'\2', text)
    # Remove markdown image references  ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove all HTML tags (img, video, audio, br, div, etc.)
    text = re.sub(r'<[^>]+/?>', '', text)
    # Remove any stray base64 data URIs
    text = re.sub(r'data:[\w/\-]+;base64,[A-Za-z0-9+/=]+', '', text)
    # Strip markdown bold/italic markers but keep text
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', text)
    # Strip markdown heading markers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def transform_embedding(embedding_values: list[float]) -> list[float]:
    """L2-normalise an embedding vector."""
    np_embedding = np.array(embedding_values)
    norm = np.linalg.norm(np_embedding)
    if norm == 0:
        return np_embedding.tolist()
    return (np_embedding / norm).tolist()


class EmbeddingService:
    """Thin wrapper around the Vertex AI embedding model."""

    MODEL = "gemini-embedding-001"
    DIMENSIONS = 768

    def __init__(self, settings: Settings) -> None:
        self._client = Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gcp_location,
        )

    # ── Single-item helpers ─────────────────────────────────────

    def embed_note(self, text: str) -> list[float]:
        """Embed note content using RETRIEVAL_DOCUMENT task type."""
        cleaned = clean_text(text)
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=cleaned,
            config=types.EmbedContentConfig(
                output_dimensionality=self.DIMENSIONS,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return transform_embedding(result.embeddings[0].values)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query using RETRIEVAL_QUERY task type."""
        cleaned = clean_text(text)
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=cleaned,
            config=types.EmbedContentConfig(
                output_dimensionality=self.DIMENSIONS,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return transform_embedding(result.embeddings[0].values)

    # ── Batch helper ────────────────────────────────────────────

    def embed_notes_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of note texts in a single API call using RETRIEVAL_DOCUMENT."""
        cleaned = [clean_text(t) for t in texts]
        result = self._client.models.embed_content(
            model=self.MODEL,
            contents=cleaned,
            config=types.EmbedContentConfig(
                output_dimensionality=self.DIMENSIONS,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return [transform_embedding(e.values) for e in result.embeddings]
