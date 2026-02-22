"""
Mass-embed all notes that have no embedding yet.

Connects to Supabase with the service role key, fetches notes with
embedding IS NULL across all decks, batches them, and upserts the
resulting embeddings back.

Usage:
    python mass_embed_notes.py
    python mass_embed_notes.py --batch-size 50 --delay 2.0
"""

import argparse
import logging
import time

from dotenv import load_dotenv
from supabase import create_client

from voice_agent.config import get_settings
from voice_agent.embeddings import EmbeddingService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
DEFAULT_DELAY = 1.0  # seconds between batches


def fetch_notes(supabase, batch_size: int) -> list[dict]:
    """Fetch a page of notes that still need embeddings."""
    response = (
        supabase.table("notes")
        .select("id, front_content, back_content")
        .is_("embedding", "null")
        .limit(batch_size)
        .execute()
    )
    return response.data or []


def upsert_embeddings(supabase, records: list[dict]) -> None:
    """Write embeddings back to the notes table."""
    supabase.table("notes").upsert(records).execute()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mass-embed all notes missing embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Notes per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to wait between batches (default: {DEFAULT_DELAY})",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Use the service role key so we can read/write all notes
    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    embedding_service = EmbeddingService(settings)

    batch_num = 0
    total_embedded = 0

    while True:
        notes = fetch_notes(supabase, args.batch_size)
        if not notes:
            break

        batch_num += 1
        logger.info("Batch %d: %d notes", batch_num, len(notes))

        # Build the text list for batch embedding
        texts = []
        for note in notes:
            front = note.get("front_content", "") or ""
            back = note.get("back_content", "") or ""
            texts.append(f"{front}\n\n{back}".strip())

        # Generate embeddings in one API call
        embeddings = embedding_service.embed_notes_batch(texts)

        # Build upsert records
        records = [
            {"id": notes[i]["id"], "embedding": embeddings[i]}
            for i in range(len(notes))
        ]
        upsert_embeddings(supabase, records)

        total_embedded += len(notes)
        logger.info(
            "Batch %d complete — %d notes embedded so far", batch_num, total_embedded
        )

        # Rate-limit to be kind to the API
        if len(notes) == args.batch_size:
            time.sleep(args.delay)
        else:
            break  # Last partial batch — no more notes

    logger.info("Done. Total notes embedded: %d", total_embedded)


if __name__ == "__main__":
    main()
