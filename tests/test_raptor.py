import numpy as np

from udaplay import raptor


def test_node_dataclass():
    n = raptor.Node(id="leaf-0", text="t", level=0, type="leaf", children=[], metadata={})
    assert n.id == "leaf-0" and n.type == "leaf" and n.level == 0


def test_best_gmm_picks_two_clusters():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=-5, scale=0.1, size=(20, 2))
    b = rng.normal(loc=5, scale=0.1, size=(20, 2))
    data = np.vstack([a, b])
    gm, k = raptor.best_gmm(data, max_clusters=5, random_state=0)
    assert k == 2


def test_soft_labels_assigns_every_point():
    rng = np.random.default_rng(0)
    data = np.vstack([rng.normal(-5, 0.1, (10, 2)), rng.normal(5, 0.1, (10, 2))])
    gm, k = raptor.best_gmm(data, max_clusters=5, random_state=0)
    labels = raptor.soft_labels(gm, data, threshold=0.1)
    assert len(labels) == 20
    assert all(len(lbl) >= 1 for lbl in labels)  # no orphan points


def test_perform_clustering_groups_two_blobs():
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(-5, 0.05, (15, 8)), rng.normal(5, 0.05, (15, 8))])
    clusters = raptor.perform_clustering(emb, dim=4, threshold=0.1, max_clusters=6)
    # every index assigned, and clearly fewer clusters than points (compression)
    assigned = {i for grp in clusters for i in grp}
    assert assigned == set(range(30))
    assert 1 <= len(clusters) <= 8
    assert len(clusters) < 30


def test_build_tree_with_injected_clusterer():
    leaf_texts = [f"text-{i}" for i in range(8)]
    leaf_meta = [{"Name": f"G{i}"} for i in range(8)]

    # Fake embedder: deterministic, shape (n, 4)
    def fake_embedder(texts):
        return np.array([[len(t), i, 0.0, 1.0] for i, t in enumerate(texts)], dtype=float)

    # Fake summarizer: join member texts
    def fake_summarizer(texts):
        return "SUMMARY(" + "|".join(texts) + ")"

    # Deterministic clusterer: split current level into 2 halves
    def half_clusterer(emb, **kwargs):
        n = emb.shape[0]
        mid = n // 2
        return [list(range(mid)), list(range(mid, n))]

    nodes = raptor.build_tree(leaf_texts, leaf_meta,
                              embedder=fake_embedder, summarizer=fake_summarizer,
                              clusterer=half_clusterer, max_levels=3)

    leaves = [n for n in nodes if n.type == "leaf"]
    summaries = [n for n in nodes if n.type == "summary"]
    assert len(leaves) == 8
    assert len(summaries) >= 1
    # children of a summary must reference real node ids
    all_ids = {n.id for n in nodes}
    for s in summaries:
        assert s.children and all(c in all_ids for c in s.children)
    # levels increase
    assert max(n.level for n in nodes) >= 1
