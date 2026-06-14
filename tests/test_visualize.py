import pandas as pd

from udaplay import visualize
from udaplay.raptor import Node


def test_tree_to_dot_contains_nodes():
    nodes = [Node(id="leaf-0", text="a", level=0, type="leaf", children=[], metadata={}),
             Node(id="summary-l1-c0", text="s", level=1, type="summary",
                  children=["leaf-0"], metadata={})]
    dot = visualize.tree_to_dot(nodes)
    assert "leaf-0" in dot and "summary-l1-c0" in dot
    assert '"leaf-0" -> "summary-l1-c0"' in dot


def test_draw_ablation_returns_axes(tmp_path):
    df = pd.DataFrame({"hit_rate": [0.5, 0.7, 0.9], "mean_top_score": [0.4, 0.6, 0.8]},
                      index=["baseline", "raptor", "raptor_sac"])
    out = tmp_path / "ablation.png"
    ax = visualize.draw_ablation(df, save_path=str(out))
    assert out.exists()
    assert ax is not None
