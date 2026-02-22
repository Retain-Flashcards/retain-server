from supabase import create_client, Client
from google.genai import Client, types
import os
import re
import numpy as np
import time
from typing import TypedDict, Optional

GOOGLE_API_KEY=''

TEST_DECK = ''

client = Client(api_key=GOOGLE_API_KEY)
supabase = create_client('', '')

PAGE_SIZE = 20

def clean_text(text: str) -> str:
    text = re.sub(r'{{c(\d)::(.+?)(?:(?:::)([^:]+)?)?}}', r'\2', text).strip()
    text = re.sub(r'<img.*src=[\'\"].*[\'\"].*>', '', text).strip()
    return text

def fetch_notes():
    response = supabase.table('notes').select('*').eq('deck_id', TEST_DECK).is_('embedding', None).limit(PAGE_SIZE).execute()

    if not response.data:
        return []
    
    return response.data


def insert_embeddings(data_list: list[dict]):
    supabase.table('notes').upsert(data_list).execute()


def transform_embedding(embedding_values):
    np_embedding = np.array(embedding_values)
    return (np_embedding / np.linalg.norm(np_embedding)).tolist()


def create_embeddings(notes: list[dict]) -> list[dict]:
    text_list = [clean_text(note['front_content'] + '\n\n' + note['back_content']) for note in notes]

    embeddings = client.models.embed_content(
        model='gemini-embedding-001', 
        contents=text_list,
        config=types.EmbedContentConfig(
            output_dimensionality=768,
            task_type='RETRIEVAL_DOCUMENT'
        )
    )

    return [{**notes[i], 'embedding': transform_embedding(embeddings.embeddings[i].values)} for i in range(len(notes))]


notes_list = fetch_notes()
n = 1

while len(notes_list) > 0:
    print(f'Batch {n}: {len(notes_list)} notes')
    embeddings = create_embeddings(notes_list)
    insert_embeddings(embeddings)
    notes_list = fetch_notes()
    time.sleep(10)
    n += 1