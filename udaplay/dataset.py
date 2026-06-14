"""RAWG-backed dataset builder.

Produces ~1000 game records in the starter schema:
{Name, Platform, Genre, Publisher, Description, YearOfRelease}.

Key-optional: with no RAWG_API_KEY, fetching is skipped and existing
data/games/*.json are used. Never raises on a missing key.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

from udaplay.config import CACHE_DIR, CHAT_MODEL, DATA_DIR, get_openai_client

RAWG_BASE = "https://api.rawg.io/api"


def year_from_released(released: Optional[str]) -> Optional[int]:
    if not released:
        return None
    try:
        return int(str(released)[:4])
    except (ValueError, TypeError):
        return None


def _first_name(items, key="name") -> Optional[str]:
    if not items:
        return None
    first = items[0]
    if isinstance(first, dict):
        # platforms come nested as {"platform": {"name": ...}}
        if "platform" in first and isinstance(first["platform"], dict):
            return first["platform"].get(key)
        return first.get(key)
    return None


def normalize_game(list_item: dict, detail: Optional[dict] = None) -> dict:
    detail = detail or {}
    name = list_item.get("name") or detail.get("name") or "Unknown"
    year = year_from_released(list_item.get("released") or detail.get("released"))
    platform = _first_name(list_item.get("platforms")) or "Unknown"
    genre = _first_name(list_item.get("genres")) or "Unknown"
    publisher = _first_name(detail.get("publishers")) or "Unknown"
    description = (detail.get("description_raw") or "").strip() or None
    return {
        "Name": name,
        "Platform": platform,
        "Genre": genre,
        "Publisher": publisher,
        "Description": description,  # None -> LLM fills it later
        "YearOfRelease": year,
    }


class RateLimiter:
    """Enforce a minimum interval between successive calls (conservative throttle)."""

    def __init__(self, min_interval: float = 0.2):
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        delta = now - self._last
        if self._last and delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class RawgClient:
    """Minimal cached RAWG API client."""

    def __init__(self, api_key: str, cache_dir: str = CACHE_DIR, min_interval: float = 0.2):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(min_interval)

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        key = json.dumps({"e": endpoint, "p": params}, sort_keys=True)
        digest = hashlib.sha256(key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _get(self, endpoint: str, params: dict) -> dict:
        params = {**params, "key": self.api_key}
        cache_params = {k: v for k, v in params.items() if k != "key"}
        path = self._cache_path(endpoint, cache_params)
        if path.exists():
            return json.loads(path.read_text())
        self.limiter.wait()
        resp = requests.get(f"{RAWG_BASE}/{endpoint}", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        path.write_text(json.dumps(data))
        return data

    def list_games(self, page: int, page_size: int = 40, dates: str = "2020-01-01,2025-12-31") -> dict:
        return self._get("games", {"page": page, "page_size": page_size,
                                   "dates": dates, "ordering": "-added"})

    def game_detail(self, game_id: int) -> dict:
        return self._get(f"games/{game_id}", {})


def fill_missing_description(game: dict, client=None, model: str = CHAT_MODEL) -> dict:
    """If Description is missing, generate a concise factual blurb with the LLM.

    Publisher gaps are left as 'Unknown' (we do not invent publishers)."""
    if game.get("Description"):
        return game
    client = client or get_openai_client()
    prompt = (
        "Write a single concise, factual sentence describing this video game for a catalog. "
        "Do not invent details beyond what is given.\n"
        f"Name: {game.get('Name')}\nPlatform: {game.get('Platform')}\n"
        f"Genre: {game.get('Genre')}\nPublisher: {game.get('Publisher')}\n"
        f"Year: {game.get('YearOfRelease')}"
    )
    resp = client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    game["Description"] = resp.choices[0].message.content.strip()
    return game


def _count_json(out_dir: Path) -> int:
    return len(list(out_dir.glob("*.json")))


def build_dataset(target_count: int = 1000, out_dir: str = DATA_DIR,
                  force: bool = False, fill_descriptions: bool = True) -> int:
    """Build the dataset to ~target_count games. Returns the number of JSON files present.

    Key-optional: without RAWG_API_KEY, logs a notice and returns the existing count.
    Idempotent: skips games already on disk (by RAWG id filename) unless force=True.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("RAWG_API_KEY")
    if not api_key:
        print(f"[dataset] No RAWG_API_KEY set; using {_count_json(out)} existing file(s).")
        return _count_json(out)

    client = RawgClient(api_key=api_key)
    llm = get_openai_client() if fill_descriptions else None

    existing = {p.stem for p in out.glob("*.json")}
    written = len(existing)
    page = 1
    while written < target_count:
        listing = client.list_games(page=page)
        results = listing.get("results", [])
        if not results:
            break
        for item in results:
            if written >= target_count:
                break
            gid = item.get("id")
            stem = f"rawg-{gid}"
            target = out / f"{stem}.json"
            if target.exists() and not force:
                continue
            detail = client.game_detail(gid) if gid else {}
            game = normalize_game(item, detail)
            if fill_descriptions and not game["Description"]:
                game = fill_missing_description(game, client=llm)
            target.write_text(json.dumps(game, ensure_ascii=False, indent=2))
            written += 1
        if not listing.get("next"):
            break
        page += 1

    print(f"[dataset] {_count_json(out)} game(s) in {out}.")
    return _count_json(out)
