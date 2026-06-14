"""RAPTOR: recursive clustering + abstractive summarization into a retrieval tree.

Faithful to the paper: embed leaves -> UMAP dimensionality reduction -> Gaussian
Mixture (soft) clustering with BIC model selection, global then local -> LLM
summary per cluster -> recurse. The flattened node list (collapsed tree) is what
gets indexed, so retrieval can hit leaves and summaries at any level.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class Node:
    id: str
    text: str
    level: int
    type: str  # "leaf" | "summary"
    children: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def reduce_dim(embeddings: np.ndarray, n_neighbors: int = 15,
               n_components: int = 10, metric: str = "cosine") -> np.ndarray:
    """UMAP reduction with guards for small n (UMAP needs n_neighbors < n_samples)."""
    import umap

    n = embeddings.shape[0]
    if n <= 2:
        return embeddings
    n_neighbors = min(n_neighbors, n - 1)
    n_components = min(n_components, max(2, n - 2))
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                        metric=metric, random_state=42)
    return reducer.fit_transform(embeddings)


def best_gmm(data: np.ndarray, max_clusters: int = 50, random_state: int = 42):
    """Fit GMMs for k in [1, max_clusters], return the one minimizing BIC."""
    n = data.shape[0]
    max_clusters = max(1, min(max_clusters, n))
    bics = []
    for k in range(1, max_clusters + 1):
        gm = GaussianMixture(n_components=k, random_state=random_state).fit(data)
        bics.append(gm.bic(data))
    best_k = int(np.argmin(bics)) + 1
    gm = GaussianMixture(n_components=best_k, random_state=random_state).fit(data)
    return gm, best_k


def soft_labels(gm: GaussianMixture, data: np.ndarray, threshold: float = 0.1) -> List[List[int]]:
    """Soft assignment: a point belongs to every cluster with prob > threshold.

    Falls back to the argmax cluster so no point is orphaned."""
    probs = gm.predict_proba(data)
    out = []
    for p in probs:
        members = list(np.where(p > threshold)[0])
        if not members:
            members = [int(np.argmax(p))]
        out.append([int(c) for c in members])
    return out


def perform_clustering(embeddings: np.ndarray, dim: int = 10, threshold: float = 0.1,
                       max_clusters: int = 50, local_threshold: int = 50) -> List[List[int]]:
    """Cluster row vectors, returning a list of clusters (each a list of row indices).

    Global pass over all points; a large global cluster (more than ``local_threshold``
    members) is refined with a second local pass. Soft membership means a point may
    appear in more than one cluster. ``local_threshold`` keeps the tree from
    over-fragmenting: only clusters too big to summarize in one shot get re-split."""
    n = embeddings.shape[0]
    if n <= 2:
        return [list(range(n))]

    reduced = reduce_dim(embeddings, n_components=dim)
    gm, k = best_gmm(reduced, max_clusters=max_clusters)
    global_labels = soft_labels(gm, reduced, threshold)

    # Collect global cluster -> member indices
    global_clusters: dict[int, List[int]] = {}
    for idx, labels in enumerate(global_labels):
        for c in labels:
            global_clusters.setdefault(c, []).append(idx)

    final: List[List[int]] = []
    for members in global_clusters.values():
        if len(members) <= local_threshold:
            final.append(members)
            continue
        # Local refinement within a large global cluster
        sub = embeddings[members]
        sub_reduced = reduce_dim(sub, n_components=min(dim, sub.shape[0] - 2))
        sub_gm, _ = best_gmm(sub_reduced, max_clusters=max_clusters)
        sub_labels = soft_labels(sub_gm, sub_reduced, threshold)
        local: dict[int, List[int]] = {}
        for local_idx, labels in enumerate(sub_labels):
            for c in labels:
                local.setdefault(c, []).append(members[local_idx])
        final.extend(local.values())

    return [m for m in final if m]


def _default_embedder(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Embed via the OpenAI embeddings endpoint (respects OPENAI_BASE_URL).

    Batched so each request stays under the provider's per-request token cap
    (300k on Vocareum); the full corpus in one call would exceed it."""
    from udaplay.config import EMBED_MODEL, get_openai_client

    client = get_openai_client()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return np.array(vectors, dtype=float)


def _default_summarizer(texts: List[str]) -> str:
    from udaplay.config import CHAT_MODEL, get_openai_client

    client = get_openai_client()
    joined = "\n\n---\n\n".join(texts)
    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0.2,
        messages=[
            {"role": "system", "content": "You summarize clusters of video-game records "
                                           "into one dense, factual paragraph capturing the "
                                           "common themes (platforms, genres, eras, publishers)."},
            {"role": "user", "content": f"Summarize these records:\n\n{joined}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def build_tree(leaf_texts: List[str], leaf_metadatas: List[dict],
               embedder: Optional[Callable[[List[str]], np.ndarray]] = None,
               summarizer: Optional[Callable[[List[str]], str]] = None,
               clusterer: Optional[Callable[..., List[List[int]]]] = None,
               max_levels: int = 3, dim: int = 10, threshold: float = 0.1) -> List[Node]:
    """Build the collapsed RAPTOR tree and return all nodes (leaves + summaries)."""
    embedder = embedder or _default_embedder
    summarizer = summarizer or _default_summarizer
    clusterer = clusterer or (lambda emb, **kw: perform_clustering(emb, dim=dim, threshold=threshold))

    nodes: List[Node] = []
    current: List[Node] = []
    for i, (text, meta) in enumerate(zip(leaf_texts, leaf_metadatas)):
        node = Node(id=f"leaf-{i}", text=text, level=0, type="leaf",
                    children=[], metadata={**meta, "type": "leaf", "level": 0})
        nodes.append(node)
        current.append(node)

    level = 1
    while level <= max_levels and len(current) > 1:
        emb = embedder([n.text for n in current])
        clusters = clusterer(emb, dim=dim, threshold=threshold)
        clusters = [c for c in clusters if c]
        if len(clusters) >= len(current):
            break  # no compression achieved; stop

        new_nodes: List[Node] = []
        for ci, idxs in enumerate(clusters):
            members = [current[i] for i in idxs]
            summary = summarizer([m.text for m in members])
            node = Node(id=f"summary-l{level}-c{ci}", text=summary, level=level,
                        type="summary", children=[m.id for m in members],
                        metadata={"type": "summary", "level": level})
            nodes.append(node)
            new_nodes.append(node)

        current = new_nodes
        level += 1

    return nodes
