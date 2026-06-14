"""Summary-Augmented Chunking: generate an enriched, embedding-optimized text per game.

Instead of embedding the terse Description, we embed a richer contextual paragraph
that surfaces name, platform, genre, publisher, era, and salient facts. This is what
gets stored as the leaf text for the RAPTOR+SAC index.
"""

from typing import List

from udaplay.config import CHAT_MODEL, get_openai_client

SAC_SYSTEM = (
    "You write concise, factual context paragraphs that maximize semantic searchability "
    "for a video-game catalog. Use only the provided facts. 2-4 sentences."
)


def build_sac_prompt(game: dict) -> str:
    return (
        "Summarize this game as a single dense paragraph for semantic retrieval. "
        "Mention the game, its platform, genre, publisher, release era, and what it is known for.\n"
        f"Name: {game.get('Name')}\n"
        f"Platform: {game.get('Platform')}\n"
        f"Genre: {game.get('Genre')}\n"
        f"Publisher: {game.get('Publisher')}\n"
        f"YearOfRelease: {game.get('YearOfRelease')}\n"
        f"Description: {game.get('Description')}\n"
    )


def enrich(game: dict, client=None, model: str = CHAT_MODEL) -> str:
    client = client or get_openai_client()
    resp = client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[
            {"role": "system", "content": SAC_SYSTEM},
            {"role": "user", "content": build_sac_prompt(game)},
        ],
    )
    return resp.choices[0].message.content.strip()


def enrich_many(games: List[dict], client=None, model: str = CHAT_MODEL) -> List[str]:
    client = client or get_openai_client()
    return [enrich(g, client=client, model=model) for g in games]
