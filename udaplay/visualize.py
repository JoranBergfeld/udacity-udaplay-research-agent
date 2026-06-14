"""Visualizations: RAPTOR tree (Graphviz DOT), retrieval trace, and ablation charts."""

from typing import List

import matplotlib
matplotlib.use("Agg")  # safe in notebooks and headless runs
import matplotlib.pyplot as plt
import pandas as pd

from udaplay.raptor import Node


def tree_to_dot(nodes: List[Node]) -> str:
    """Return a Graphviz DOT string of the RAPTOR tree (child -> parent edges)."""
    lines = ["digraph RAPTOR {", "  rankdir=BT;"]
    for n in nodes:
        shape = "box" if n.type == "summary" else "ellipse"
        label = (n.text[:30] + "...") if len(n.text) > 33 else n.text
        label = label.replace('"', "'")
        lines.append(f'  "{n.id}" [shape={shape}, label="{n.id}\\n{label}"];')
    for n in nodes:
        for child in n.children:
            lines.append(f'  "{child}" -> "{n.id}";')
    lines.append("}")
    return "\n".join(lines)


def draw_tree(nodes: List[Node], save_path: str = None):
    """Render the tree with graphviz if available; otherwise return the DOT string."""
    dot = tree_to_dot(nodes)
    try:
        import graphviz
        src = graphviz.Source(dot)
        if save_path:
            src.render(save_path, format="png", cleanup=True)
        return src
    except Exception:
        if save_path:
            with open(save_path, "w") as f:
                f.write(dot)
        return dot


def draw_retrieval_trace(query: str, results: List[dict], save_path: str = None):
    """Bar chart of retrieved nodes by score, colored by tree level."""
    names = [f"{r.get('Name') or r.get('type')} (L{r.get('level')})" for r in results]
    scores = [r.get("score", 0.0) for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names[::-1], scores[::-1])
    ax.set_title(f"Retrieval trace: {query[:50]}")
    ax.set_xlabel("similarity score")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return ax


def draw_ablation(df: pd.DataFrame, save_path: str = None):
    """Grouped bar chart comparing indexes on hit_rate and mean_top_score."""
    cols = [c for c in ["hit_rate", "mean_top_score"] if c in df.columns]
    ax = df[cols].plot(kind="bar", figsize=(8, 5), rot=20)
    ax.set_title("Ablation: Baseline vs RAPTOR vs RAPTOR+SAC")
    ax.set_ylabel("score")
    ax.figure.tight_layout()
    if save_path:
        ax.figure.savefig(save_path)
    return ax
