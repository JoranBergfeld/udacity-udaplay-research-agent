"""Ablation harness: compare Baseline vs RAPTOR vs RAPTOR+SAC on a fixed query set."""

from typing import List

import pandas as pd

from udaplay import agent
from udaplay import index_builder as ib

# Each item: a query and the substrings we expect to appear in a retrieved game Name.
QUERY_SET = [
    {"query": "When was Gran Turismo released on PlayStation 1?", "expected": ["Gran Turismo"]},
    {"query": "Open-world game set in San Andreas", "expected": ["San Andreas"]},
    {"query": "Classic Super Nintendo Mario platformer", "expected": ["Super Mario World"]},
    {"query": "Latest Halo game with Master Chief", "expected": ["Halo Infinite"]},
    {"query": "Realistic racing simulator on Sony console", "expected": ["Gran Turismo"]},
    {"query": "Rockstar open-world crime game", "expected": ["Grand Theft Auto"]},
    {"query": "Nintendo platformer with dinosaurs", "expected": ["Super Mario World"]},
    {"query": "First-person shooter on Xbox Series", "expected": ["Halo"]},
    {"query": "Story of Carl CJ Johnson", "expected": ["San Andreas"]},
    {"query": "Save Princess Toadstool from Bowser", "expected": ["Mario"]},
    {"query": "Racing game with many cars and tracks", "expected": ["Gran Turismo"]},
    {"query": "Action adventure on PlayStation 2", "expected": ["San Andreas"]},
]


def hit_at_k(results: List[dict], expected: List[str]) -> bool:
    names = " ".join((r.get("Name") or "") for r in results).lower()
    return any(e.lower() in names for e in expected)


def mean_score(results: List[dict]) -> float:
    scores = [r.get("score", 0.0) for r in results]
    return sum(scores) / len(scores) if scores else 0.0


INDEXES = [ib.INDEX_BASELINE, ib.INDEX_RAPTOR, ib.INDEX_RAPTOR_SAC]


def run_ablation(query_set=None, retriever=None, n_results: int = 5) -> pd.DataFrame:
    """Run each query against each index; return a per-index metrics DataFrame."""
    query_set = query_set or QUERY_SET
    retriever = retriever or agent.retrieve_game

    rows = {}
    for index in INDEXES:
        hits, scores = [], []
        for item in query_set:
            results = retriever(item["query"], index=index, n_results=n_results)
            hits.append(1.0 if hit_at_k(results, item["expected"]) else 0.0)
            scores.append(mean_score(results))
        rows[index] = {
            "hit_rate": sum(hits) / len(hits),
            "mean_top_score": sum(scores) / len(scores),
            "n_queries": len(query_set),
        }
    return pd.DataFrame(rows).T[["hit_rate", "mean_top_score", "n_queries"]]
