"""Builds the three persistent ChromaDB collections for the ablation.

- udaplay_baseline    : flat, raw text per game
- udaplay_raptor      : baseline leaves + RAPTOR summary nodes
- udaplay_raptor_sac  : SAC-enriched leaves + RAPTOR summary nodes
"""

import json
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from udaplay import raptor, sac
from udaplay.config import (CHROMA_OPENAI_KEY_ENV, CHROMA_PATH, DATA_DIR,
                            EMBED_MODEL, OPENAI_BASE_URL)

INDEX_BASELINE = "udaplay_baseline"
INDEX_RAPTOR = "udaplay_raptor"
INDEX_RAPTOR_SAC = "udaplay_raptor_sac"
LONGTERM = "udaplay_longterm"


def baseline_text(game: dict) -> str:
    return (f"[{game.get('Platform')}] {game.get('Name')} "
            f"({game.get('YearOfRelease')}) - {game.get('Description')}")


def clean_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool: drop None, JSON-encode lists/dicts."""
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = "Unknown"
        elif isinstance(v, (list, dict)):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def load_games(data_dir: str = DATA_DIR) -> List[dict]:
    games = []
    for f in sorted(Path(data_dir).glob("*.json")):
        games.append(json.loads(f.read_text()))
    return games


def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_embedding_function():
    # api_base is REQUIRED so embeddings route through Vocareum (Chroma ignores OPENAI_BASE_URL env).
    return embedding_functions.OpenAIEmbeddingFunction(
        model_name=EMBED_MODEL,
        api_base=OPENAI_BASE_URL,
        api_key_env_var=CHROMA_OPENAI_KEY_ENV,
    )


def _get_or_reset_collection(client, name: str, force: bool):
    if force:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, embedding_function=get_embedding_function())


def _add_nodes(collection, nodes, batch_size: int = 100) -> None:
    """Add nodes in batches.

    Chroma's embedding function embeds every document in a single add() call as one
    API request; the full ~1000-doc corpus exceeds the provider's per-request token
    cap (300k on Vocareum). Batching keeps each embedding request well under it.
    """
    if not nodes:
        return
    for start in range(0, len(nodes), batch_size):
        batch = nodes[start:start + batch_size]
        collection.add(
            ids=[n.id for n in batch],
            documents=[n.text for n in batch],
            metadatas=[clean_metadata(n.metadata) for n in batch],
        )


def _leaf_nodes(texts: List[str], games: List[dict]):
    return [raptor.Node(id=f"leaf-{i}", text=texts[i], level=0, type="leaf",
                        children=[], metadata={**games[i], "type": "leaf", "level": 0})
            for i in range(len(games))]


def build_baseline(force: bool = False) -> int:
    client = get_chroma_client()
    coll = _get_or_reset_collection(client, INDEX_BASELINE, force)
    if coll.count() > 0 and not force:
        return coll.count()
    games = load_games()
    nodes = _leaf_nodes([baseline_text(g) for g in games], games)
    _add_nodes(coll, nodes)
    return coll.count()


def build_raptor(force: bool = False) -> int:
    client = get_chroma_client()
    coll = _get_or_reset_collection(client, INDEX_RAPTOR, force)
    if coll.count() > 0 and not force:
        return coll.count()
    games = load_games()
    leaf_texts = [baseline_text(g) for g in games]
    leaf_meta = [{**g} for g in games]
    nodes = raptor.build_tree(leaf_texts, leaf_meta)
    _add_nodes(coll, nodes)
    return coll.count()


def build_raptor_sac(force: bool = False) -> int:
    client = get_chroma_client()
    coll = _get_or_reset_collection(client, INDEX_RAPTOR_SAC, force)
    if coll.count() > 0 and not force:
        return coll.count()
    games = load_games()
    leaf_texts = sac.enrich_many(games)
    leaf_meta = [{**g} for g in games]
    nodes = raptor.build_tree(leaf_texts, leaf_meta)
    _add_nodes(coll, nodes)
    return coll.count()


def build_all(force: bool = False) -> dict:
    return {
        INDEX_BASELINE: build_baseline(force),
        INDEX_RAPTOR: build_raptor(force),
        INDEX_RAPTOR_SAC: build_raptor_sac(force),
    }


def get_collection(name: str):
    return get_chroma_client().get_collection(
        name=name, embedding_function=get_embedding_function())
