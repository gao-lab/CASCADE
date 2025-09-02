import networkx as nx
import numpy as np
import pandas as pd
from pytest import raises

from cascade.plot import (
    interactive_heatmap,
    motion_pictures,
    pair_grid,
    plot_adj,
    plot_adj_confusion,
    plot_colored_curves,
    plot_design_error_curve,
    plot_design_scores,
    set_figure_params,
)


def test_set_figure_params():
    set_figure_params()


def test_plot_adj(dag):
    adj = nx.to_pandas_adjacency(dag)
    plot_adj(adj, mask=adj > 0)
    plot_adj(adj, mask=adj > 0, cluster=True)


def test_plot_adj_confusion(dag):
    true = nx.to_pandas_adjacency(dag, weight=None, dtype=bool)
    pred = nx.to_pandas_adjacency(dag)
    plot_adj_confusion(true, pred > 0)
    with raises(TypeError):
        plot_adj_confusion(pred, pred > 0)
    with raises(TypeError):
        plot_adj_confusion(true, pred)
    with raises(ValueError):
        plot_adj_confusion(true, pred > 0, center=0)


def test_plot_colored_curves():
    data = pd.DataFrame({"x": np.arange(10), "y": np.arange(10), "hue": np.arange(10)})
    plot_colored_curves(x="x", y="y", hue="hue", data=data)


def test_plot_design_scores():
    scores = pd.DataFrame({"score": [5, 4, 3, 2, 1]}, index=["A", "B", "C", "D", "E"])
    plot_design_scores(scores)


def test_plot_design_error_curve():
    curve = pd.DataFrame(
        {"score": [5, 4, 3, 2, 1], "mse_est": [0.1, 0.2, 0.3, 0.4, 0.5]},
        index=["A", "B", "C", "D", "E"],
    )
    curve["mse_est_mean"] = curve["mse_est"]
    curve["mse_est_lower"] = curve["mse_est"] - 0.05
    curve["mse_est_upper"] = curve["mse_est"] + 0.05
    plot_design_error_curve(curve, cutoff=3)


def test_motion_pictures(tmp_path):
    motion_pictures("tests/test.tfevents", tmp_path / "test.mp4", "adj", True)
    motion_pictures("tests/test.tfevents", tmp_path / "test.avi", "adj", True)


def test_pair_grid(adata):
    pair_grid(adata.to_df().iloc[:, :3])
    pair_grid(adata.to_df().iloc[:, :4], weight="D")


def test_interactive_heatmap(adata):
    interactive_heatmap(adata.to_df(), row_clust=None, col_clust=None)
    interactive_heatmap(
        adata.to_df(),
        row_clust=adata.obs["interv"],
        col_clust=adata.var["group"],
        highlights={"A": "red", "B": "green", "123": "blue", "321": "yellow"},
    )
