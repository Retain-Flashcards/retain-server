"""
Tool definitions and execution dispatcher for the voice agent.

Each tool is an async function that receives a ``CardCache`` instance.
The Gemini model only sees the parameters defined in ``TOOL_DECLARATIONS``;
the cache is injected server-side by ``execute_tool``.
"""

import logging
from typing import Any, Callable, Awaitable
from voice_agent.card_manager import CardManager

logger = logging.getLogger(__name__)


# ── Tool implementations ────────────────────────────────────────

async def check_top_5_cards_current_topic(card_manager: CardManager) -> dict:
    """Return the next 5 cards for the current topic from the cache."""
    cards = await card_manager.get_top(5)
    return {
        "topic": card_manager.current_topic,
        "cards": [{ 'id': card['num_id'], 'content': card['content'] } for card in cards]
    }


async def search_cards_for_new_topic(card_manager: CardManager, topic: str) -> dict:
    """Switch to a new topic, refetch cards, and return the first 5."""
    await card_manager.set_topic(topic)
    cards = await card_manager.get_top(5)
    return {
        "topic": topic,
        "cards": [{ 'id': card['num_id'], 'content': card['content'] } for card in cards]
    }


async def submit_review(card_manager: CardManager, card_id: int, difficulty: str) -> dict:
    """Record a review for a card and evict it from the cache.

    Parameters
    ----------
    card_id : int
        The ID of the card being reviewed.
    difficulty : str
        How difficult the user found the card (e.g. "good", "hard", "again").
    """
    # Map the model's difficulty to the backend's expected difficulty
    difficulty_map = {
        "correct": "good",
        "struggled": "hard",
        "incorrect": "again"
    }
    mapped_difficulty = difficulty_map.get(difficulty.lower(), "good")
    card = card_manager.get_card(card_id)
    card_manager.review_card(card_id, mapped_difficulty)
    logger.info("Review submitted — card_id=%d difficulty=%s (mapped to %s)", card_id, difficulty, mapped_difficulty)
    
    result = {
        "status": "submitted_in_background",
        "card_id": card_id,
        "difficulty": difficulty
    }
    if card:
        result["card"] = card
    return result

async def skip_card_permanently(card_manager: CardManager, card_id: int) -> dict:
    """Skip a card and evict it from the cache."""
    card = card_manager.get_card(card_id)
    card_manager.skip_card(card_id)
    logger.info("Card skipped — card_id=%d", card_id)
    
    result = {
        "status": "skipped_in_background",
        "card_id": card_id
    }
    if card:
        result["card"] = card
    return result


# ── Gemini function declarations ────────────────────────────────
# These define what the model "sees" — no mention of the cache param.

TOOL_DECLARATIONS: list[dict[str, Any]] = [
    {
        "name": "check_top_5_cards_current_topic",
        "description": (
            "Retrieve the next BATCH of up to 5 due flashcards for the user, continuing the topics being explored. "
            "WARNING: This is a batched retrieval from a potentially massive queue. Receiving 5 cards does NOT mean there are only 5 left. "
            "NEVER tell the user they are 'almost done'. You are NOT done with the session until this tool returns exactly 0 cards."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "search_cards_for_new_topic",
        "description": (
            "Search for flashcards on a new topic. Replaces the current "
            "topic AND returns the first 5 matching cards - generally should be prompted by the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The new topic to search for.",
                },
            },
            "required": ["topic"],
        },
    },
    {
        "name": "submit_review",
        "description": (
            "Submit the user's review for a specific flashcard. Call this "
            "after the user has demonstrated they know the blanked out items of the card."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "integer",
                    "description": "The ID of the card being reviewed.",
                },
                "difficulty": {
                    "type": "string",
                    "description": (
                        "How difficult the card was for the user. "
                        "- correct = the user solidly knew the material and easily recalled ALL of it without your help (they won't see this for days)\n"
                        "- struggled = the user got it right, but they had to think for a while or needed a tiny hint to jumpstart their memory (they won't see this for days)\n"
                        "- incorrect = the user said 'I don't know', missed ANY key piece of information, or you had to give them the answer. Even if they got 90% right, if they missed a key blank it is INCORRECT (they will see this again today)"
                    ),
                    "enum": ["correct", "struggled", "incorrect"],
                },
            },
            "required": ["card_id", "difficulty"],
        },
    },
    {
        "name": "skip_card_permanently",
        "description": (
            "Mark a card as skipped so it doesn't return to your reviews. "
            "Use rarely, only if a card is completely unusable or is a duplicate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {
                    "type": "integer",
                    "description": "The ID of the card being skipped.",
                },
            },
            "required": ["card_id"],
        },
    }
]

# ── Registry & dispatcher ───────────────────────────────────────

TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict]]] = {
    "check_top_5_cards_current_topic": check_top_5_cards_current_topic,
    "search_cards_for_new_topic": search_cards_for_new_topic,
    "submit_review": submit_review,
    "skip_card_permanently": skip_card_permanently,
}

# TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict]]] = {}

async def execute_tool(name: str, args: dict[str, Any], card_manager: CardManager) -> dict[str, Any]:
    """Dispatch a tool call by name, injecting the card cache.

    Returns a result dict on success, or an error dict if the tool
    is unknown or raises an exception. Never leaks tracebacks.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        logger.warning("Unknown tool requested: %s", name)
        return {"error": f"Unknown tool: {name}"}

    logger.info("Executing tool '%s' with args: %s", name, args)

    try:
        return await fn(card_manager=card_manager, **args)
    except TypeError as exc:
        # Most likely a missing / extra argument
        logger.exception("Tool '%s' called with bad arguments: %s", name, args)
        return {"error": f"Invalid arguments for tool '{name}'"}
    except Exception:
        logger.exception("Tool '%s' failed", name)
        return {"error": f"Tool '{name}' encountered an internal error"}
