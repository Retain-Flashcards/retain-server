"""
Card manager with prefetch / serve / cache pattern.

Fetches 10 cards per query, serves 5 immediately, caches the other 5.
When the ready buffer depletes, the prefetch buffer is promoted and a
background refill is triggered.
"""

import asyncio
import logging
import random
from collections import deque
from typing import Any
import re
from enum import StrEnum
from datetime import datetime, timezone

from voice_agent.supabase import AsyncClient
from voice_agent.background import BackgroundTaskManager
from voice_agent.config import Settings
from voice_agent.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class CardDifficulty(StrEnum):
    AGAIN = 'again'
    HARD = 'hard'
    GOOD = 'good'


class CardManager:

    def __init__(
        self,
        settings: Settings,
        bg: BackgroundTaskManager,
        supabase: AsyncClient,
        deck_id: str,
        session_store: Any = None,
        session_id: str | None = None
    ) -> None:
        self.settings = settings
        self.bg = bg
        self.supabase = supabase
        self.session_store = session_store
        self.session_id = session_id
        self.current_topic: str = ""
        self.current_topic_embedding = None
        self.card_id_counter = 1
        self.deck_id = deck_id
        self.n_cards_to_serve = 5
        self.card_cache: list[dict[str, Any]] = []
        self.card_ids_to_num_ids: dict[str, int] = {}
        self.num_ids_to_card_ids: dict[int, str] = {}
        self._embedding_service = EmbeddingService(settings)
        self._skipped_cards: set[str] = set()

    # ── Lifecycle ───────────────────────────────────────────────

    async def initialize(self) -> None:
        """Fetch the first due card, derive a topic, run the initial query."""
        first_card = await self._fetch_first_due_card()

        topic = None
        if first_card:
            topic = first_card.get('front_content', None)
            if first_card.get('back_content') is not None:
                topic += '\n\n' + first_card.get('back_content')
        if topic is None:
            topic = 'general'

        logger.info("CardManager initialised with topic: %s", topic)
        await self.set_topic(topic)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'<img.*src=[\'\"].*[\'\"].*>', '', text).strip()
        text = re.sub(r'!\[.+?\]\(.+?\)', '<image>', text).strip()
        return text

    def get_text_from_card(self, card: dict[str, Any]) -> str | None:
        front_content = card.get('note_front_content', None)
        back_content = card.get('note_back_content', None)

        text = ''
        if front_content is not None:
            text += front_content
        if back_content is not None:
            text += back_content

        if text == '':
            return None

        return self._clean_text(text)

    async def _fetch_first_due_card(self) -> dict[str, Any] | None:
        card = await self.supabase.table('randomized_reviews').select('*').eq('deck_id', self.deck_id).execute()
        return card.data[0] if card.data and len(card.data) > 0 else None

    async def _query_cards_by_topic(self, embedding, limit: int = 10) -> list[dict[str, Any]]:
        cards = await self.supabase.rpc('search_voice_session_cards', {
            'query_embedding': embedding, 
            'match_threshold': 0.5,
            'match_count': limit,
            'with_deck_id': self.deck_id
        }).execute()
        logger.info("Found %d cards for topic", len(cards.data))

        # Assign unique numeric IDs
        for card in cards.data:
            card['num_id'] = self.card_ids_to_num_ids.get(card['id'])
            if card['num_id'] is None:
                card['num_id'] = self.card_id_counter
                self.card_id_counter += 1
                self.card_ids_to_num_ids[card['id']] = card['num_id']
                self.num_ids_to_card_ids[card['num_id']] = card['id']
            
            card['content'] = self.get_text_from_card(card)

        return cards.data

    def merge_cards_to_cache(self, cards: list[dict[str, Any]]):
        """Since both sets of cards are subsets of the same ordered list,
        and the new cards are more up-to-date,
        deduplication is ensured if we append the new cards at the point of the
        first matching card ID in the cache, or the end of the cache
        """
        if len(cards) == 0:
            return
        search_card = cards[0]

        for i, card in enumerate(self.card_cache):
            if card['id'] in self._skipped_cards:
                continue
            if card['id'] == search_card['id']:
                self.card_cache = self.card_cache[:i] + cards
                return
        
        self.card_cache.extend(cards)

    async def _update_cache(self) -> list[dict[str, Any]]:
        new_cards = await self._query_cards_by_topic(self.current_topic_embedding)
        self.merge_cards_to_cache(new_cards)
        return self.card_cache
        
    
    async def get_top(self, n: int) -> list[dict[str, Any]]:
        # If needed, get the cards before returning
        if len(self.card_cache) < n:
            await self._update_cache()

        # Refill cache if we can't guarantee instant response on next query
        elif len(self.card_cache) < n * 2:
            self.bg.schedule(self._update_cache(), name='refill_card_cache')

        # Return the top n cards
        return self.card_cache[:n]
            

    async def set_topic(self, topic: str) -> None:
        """Change the current topic, clear cache, and refill."""
        self.current_topic = topic
        self.current_topic_embedding = self._embedding_service.embed_query(topic)
        self.card_cache.clear()

        await self._update_cache()
    
    async def _review_card(self, card_id: int, difficulty: CardDifficulty) -> None:
        """Review a card and update its review schedule."""
        
        # Step 1: Translate integer card_id to corresponding UUID
        str_card_id = self.num_ids_to_card_ids[card_id]

        # Now, submit the review
        await self.supabase.rpc('submit_card_review', {
            'p_card_id': str_card_id,
            'p_category': difficulty.lower(),
            'p_local_timestamp': datetime.now(timezone.utc).isoformat(),
            'p_should_bury_related': False
        }).execute()

        if self.session_store and self.session_id:
            await self.session_store.increment_cards_reviewed(self.session_id)

        # Once this is complete, we remove it from our cache
        self.card_cache = [card for card in self.card_cache if card['id'] != str_card_id]
        
    
    def get_card(self, card_id: int) -> dict[str, Any] | None:
        str_id = self.num_ids_to_card_ids.get(card_id)
        if not str_id:
            return None
        for card in self.card_cache:
            if card['id'] == str_id:
                return card
        return None

    def review_card(self, card_id: int, difficulty: CardDifficulty) -> None:
        self.bg.schedule(self._review_card(card_id, difficulty), name='review_card')

    def skip_card(self, card_id: int) -> None:
        str_id = self.num_ids_to_card_ids[card_id]
        self._skipped_cards.add(str_id)
        self.card_cache = [card for card in self.card_cache if card['id'] != str_id]
