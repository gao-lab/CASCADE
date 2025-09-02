r"""
Plotting utilities
"""

from functools import reduce
from math import ceil
from operator import add
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from loguru import logger
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap, ListedColormap, Normalize
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import pearsonr
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .data import EPS
from .typing import Kws
from .utils import hclust


def set_figure_params() -> None:
    r"""
    Set global parameters for publication-quality figures.
    """
    sc.set_figure_params(
        scanpy=True,
        dpi_save=600,
        vector_friendly=True,
        format="pdf",
        facecolor=(1.0, 1.0, 1.0, 0.0),
        transparent=False,
    )
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.axisbelow"] = True
    rcParams["grid.linestyle"] = (0, (10, 5))
    rcParams["grid.color"] = "#EEEEEE"
    rcParams["legend.frameon"] = False
    rcParams["savefig.bbox"] = "tight"


def plot_adj(
    adj: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    cmap: str | Colormap = "bwr",
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cluster: bool = False,
    cluster_kws: dict | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    r"""
    Plot adjacency matrix

    Parameters
    ----------
    adj
        Adjacency matrix
    mask
        Boolean mask matrix
    cmap
        Color map
    row_labels
        Row labels to show
    col_labels
        Column labels to show
    cluster
        Whether to cluster rows and columns
    cluster_kws
        Keyword arguments for :func:`~scipy.cluster.hierarchy.linkage`,
        only relevant if `cluster` is True
    ax
        Existing axes object
    **kwargs
        Additional arguments are passed to :func:`~seaborn.heatmap`

    Returns
    -------
    Axes object
    """
    if mask is not None:
        mask = mask.reindex_like(adj).astype(bool)  # NaNs become True and masked
    if cluster:
        adj_nonzero = adj != 0
        row_mask = adj_nonzero.sum(axis=1) > 0
        col_mask = adj_nonzero.sum(axis=0) > 0
        if not row_mask.all():
            n_remove = row_mask.size - row_mask.sum()
            logger.warning(f"Ignoring {n_remove} rows with all zeros.")
        if not col_mask.all():
            n_remove = col_mask.size - col_mask.sum()
            logger.warning(f"Ignoring {n_remove} columns with all zeros.")
        adj = adj.loc[row_mask, col_mask]
        mask = mask.loc[row_mask, col_mask]
        cluster_kws = {
            "method": "weighted",
            "metric": "cosine",
            "optimal_ordering": True,
            **(cluster_kws or {}),
        }
        row_order = leaves_list(linkage(adj, **cluster_kws))
        col_order = leaves_list(linkage(adj.T, **cluster_kws))
        adj = adj.iloc[row_order, col_order]
        mask = mask.iloc[row_order, col_order]

    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad("lightgrey")
    ax = sns.heatmap(
        adj,
        mask=mask,
        linewidths=0.0,
        rasterized=True,
        cmap=cmap,
        ax=ax,
        **kwargs,
    )
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    row_labels = np.asarray(row_labels or [])
    col_labels = np.asarray(col_labels or [])
    xticks = adj.columns.get_indexer(col_labels) + 0.5
    yticks = adj.index.get_indexer(row_labels) + 0.5
    xsort, ysort = np.argsort(xticks), np.argsort(yticks)
    xticks, xticklabels = xticks[xsort], col_labels[xsort]
    yticks, yticklabels = yticks[ysort], row_labels[ysort]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.grid(visible=False)
    return ax


def plot_adj_confusion(
    true: pd.DataFrame,
    pred: pd.DataFrame,
    mask: pd.DataFrame | None = None,
    palette: list[str] | None = None,
    cbar: bool = True,
    **kwargs,
) -> Axes:
    r"""
    Plot confusion categories of adjacency prediction

    Parameters
    ----------
    true
        True adjacency matrix (must be boolean)
    pred
        Predicted adjacency matrix (must be boolean)
    mask
        Boolean mask matrix
    palette
        Color palette
    cbar
        Whether to include a color bar
    **kwargs
        Additional arguments are passed to :func:`plot_adj`

    Returns
    -------
    Axes object


    .. note::

        Nodes are ordered according to ``pred``.
    """
    if unsupported_args := {"center", "vmin", "vmax"} & kwargs.keys():
        raise ValueError(f"Unsupported keyword arguments: {unsupported_args}")
    if (true.dtypes != bool).any():
        raise TypeError("True adjacency matrix must be boolean")
    if (pred.dtypes != bool).any():
        raise TypeError("Predicted adjacency matrix must be boolean")
    true = true.loc[pred.index, pred.columns]  # Necessary to support partial pred

    confusion = pd.DataFrame(index=pred.index, columns=pred.columns, dtype=float)
    confusion[~true & ~pred] = 0  # True negative
    confusion[true & ~pred] = 1  # False negative
    confusion[~true & pred] = 2  # False positive
    confusion[true & pred] = 3  # True positive
    cmap = ListedColormap(palette or ["#63a99e", "#ffbb78", "#d62728", "#265158"])

    ax = plot_adj(
        confusion,
        mask=mask,
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        cbar=cbar,
        **kwargs,
    )

    if cbar:
        # Adapted from https://stackoverflow.com/a/57994641
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["TN", "FN", "FP", "TP"])
    return ax


def plot_colored_curves(
    x: str,
    y: str,
    hue: str,
    data: pd.DataFrame,
    group: str | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    r"""
    Plot colored curves where each segment is separately colored

    Parameters
    ----------
    x
        X-axis variable
    y
        Y-axis variable
    hue
        Color variable
    data
        Data frame
    group
        Grouping variable for plotting multiple curves
    cmap
        Color map
    vmin
        Minimum value for color map
    vmax
        Maximum value for color map
    ax
        Existing axes object

    Returns
    -------
    Axes object
    """
    ax = plt.gca() if ax is None else ax
    cmap = plt.colormaps.get_cmap(cmap or "viridis")
    norm = Normalize(vmin=vmin, vmax=vmax)
    gb = data.groupby(group) if group is not None else [(None, data)]

    segments = reduce(
        add,
        [
            [
                [(x1, y1), (x2, y2)]
                for x1, y1, x2, y2 in zip(d[x], d[y], d[x][1:], d[y][1:])
            ]
            for _, d in gb
        ],
    )
    colors = reduce(add, [[cmap(norm(c)) for c in d[hue]] for _, d in gb])
    ax.add_collection(LineCollection(segments, colors=colors))

    x_range = data[x].max() - data[x].min()
    y_range = data[y].max() - data[y].min()
    ax.margins(x_range * 0.05, y_range * 0.05)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax)
    return ax


def motion_pictures(
    tb_file: str,
    vid_file: str,
    tag: str,
    mono: bool,
    width: int = 480,
    height: int = 480,
    fps: float = 30,
) -> None:
    r"""
    Create video from images in tensorboard log

    Parameters
    ----------
    tb_file
        Path to input tensorboard log file
    vid_file
        Path to output video file
    tag
        Image tag in the TensorBoard log
    mono
        Whether the image is monochrome
    width
        Video width
    height
        Video height
    fps
        Frames per second
    """
    try:
        import cv2
    except ImportError:  # pragma: no cover
        raise ImportError("OpenCV is required for this function")

    accumulator = EventAccumulator(tb_file, size_guidance={"images": 0})
    accumulator.Reload()

    vid_file = Path(vid_file)
    vid_file.parent.mkdir(parents=True, exist_ok=True)
    codec_map = {".mp4": "mp4v"}
    fourcc = cv2.VideoWriter_fourcc(*codec_map.get(vid_file.suffix, "xvid"))
    video = cv2.VideoWriter(vid_file.as_posix(), fourcc, fps, (width, height), not mono)

    for event in accumulator.Images(tag):
        frame = cv2.imdecode(
            np.frombuffer(event.encoded_image_string, dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE if mono else cv2.IMREAD_COLOR,
        )
        height_rep = ceil(height / frame.shape[0])
        width_rep = ceil(width / frame.shape[1])
        frame = frame.repeat(height_rep, axis=0).repeat(width_rep, axis=1)
        frame = cv2.resize(frame, (width, height))
        video.write(frame)
    video.release()


def plot_design_scores(
    design: pd.DataFrame,
    n_label: int = 5,
    n_scatter: int | None = None,
    cutoff: float | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    r"""
    Visualize design scores

    Parameters
    ----------
    design
        Design scores (from :meth:`cascade.model.CASCADE.design` or
        :meth:`cascade.model.CASCADE.design_error_curve`)
    n_label
        Number of top interventions to label with text
    n_scatter
        Number of top interventions to plot (plot all by default)
    cutoff
        Score cutoff from :meth:`cascade.model.CASCADE.design_error_curve`
    ax
        Existing axes object
    **kwargs
        Additional keyword arguments are passed to :func:`seaborn.scatterplot`

    Returns
    -------
    Axes object
    """
    design = design.sort_values("score", ascending=False)
    if n_scatter:
        design = design.head(n_scatter)
    design = design.copy()
    design["rank"] = np.arange(design.shape[0]) + 1
    ax = sns.scatterplot(
        data=design.sort_values("rank", ascending=False),
        x="rank",
        y="score",
        hue="rank",
        palette="BuPu_r",
        rasterized=True,
        edgecolor=None,
        legend=False,
        ax=ax,
        **kwargs,
    )
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymin + (ymax - ymin) * 1.1)

    texts = [
        ax.text(row["rank"], row["score"], index, fontsize="small")
        for index, row in design.head(n_label).iterrows()
    ]
    adjust_text(
        texts, arrowprops={"arrowstyle": "-"}, min_arrow_len=5, force_text=(1.0, 1.0)
    )
    for t in texts:
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    if cutoff is not None:
        ax.axhline(y=cutoff, c="darkred", ls="--")
    if "" in design.index:
        ax.axhline(y=design.loc["", "score"], c="lightgrey", ls="--", zorder=0)
    ax.set_xlabel("Design rank")
    ax.set_ylabel("Design score")
    return ax


def plot_design_error_curve(
    curve: pd.DataFrame,
    cutoff: float | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    r"""
    Plot design error curve

    Parameters
    ----------
    curve
        Design error curve from :meth:`cascade.model.CASCADE.design_error_curve`
    cutoff
        Score cutoff from :meth:`cascade.model.CASCADE.design_error_curve`
    ax
        Existing axes object
    **kwargs
        Additional keyword arguments are passed to :func:`seaborn.scatterplot`

    Returns
    -------
    Axes object
    """
    curve = curve.dropna()
    ax = plt.gca() if ax is None else ax
    ax.fill_between(
        curve["score"],
        curve["mse_est_lower"],
        curve["mse_est_upper"],
        color="lightgrey",
        alpha=0.3,
    )
    ax = sns.lineplot(curve, x="score", y="mse_est_mean", c="black", ax=ax)
    ax = sns.scatterplot(
        curve,
        x="score",
        y="mse_est",
        edgecolor=None,
        rasterized=True,
        s=15,
        ax=ax,
        **kwargs,
    )

    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    # xrange = xmax - xmin
    # yrange = ymax - ymin
    # ax.set_xlim(xmin, xmax + 0.2 * xrange)
    # ax.set_ylim(ymin - 0.1 * yrange, ymax)

    texts = [
        ax.text(row["score"], row["mse_est"], index, fontsize="small")
        for index, row in curve.query(f"score > {cutoff}").iterrows()
    ]
    adjust_text(
        texts, arrowprops={"arrowstyle": "-"}, min_arrow_len=5, force_text=(1.0, 1.0)
    )
    for t in texts:
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
    if cutoff is not None:
        ax.axvspan(xmin=cutoff, xmax=curve["score"].max(), color="darkred", alpha=0.1)
        ax.axvline(x=cutoff, c="darkred", ls="--")
    ax.set_xlabel("Design score")
    ax.set_ylabel("MSE estimate")
    return ax


def _add_diagonal(*args, **kwargs):
    ax = plt.gca()
    ax.axline((0, 0), slope=1, c="darkred", ls="--")


def _annotate_corr_mse(x, y, color=None, label=None, hue=None, weight=None, **kwargs):
    ax = plt.gca()
    r, _ = pearsonr(x, y)
    mse = np.square(x - y).mean()
    if weight is not None:
        weight = weight.size * weight / weight.sum()
        x_center = x - (x * weight).mean()
        y_center = y - (y * weight).mean()
        var_x = (np.square(x_center) * weight).mean()
        var_y = (np.square(y_center) * weight).mean()
        cov = (x_center * y_center * weight).mean()
        r_weighted = cov / (np.sqrt(var_x * var_y) + EPS)
        mse_weighted = (np.square(x - y) * weight).mean()
        ax.annotate(
            f"r = {r:.3f}",
            xy=(0.5, 0.75),
            xycoords="axes fraction",
            ha="center",
        )
        ax.annotate(
            f"MSE = {mse:.3f}",
            xy=(0.5, 0.61),
            xycoords="axes fraction",
            ha="center",
        )
        ax.annotate(
            f"r (weighted) = {r_weighted:.3f}",
            xy=(0.5, 0.39),
            xycoords="axes fraction",
            ha="center",
        )
        ax.annotate(
            f"MSE (weighted) = {mse_weighted:.3f}",
            xy=(0.5, 0.25),
            xycoords="axes fraction",
            ha="center",
        )
    else:
        ax.annotate(
            f"r = {r:.3f}",
            xy=(0.5, 0.57),
            xycoords="axes fraction",
            ha="center",
        )
        ax.annotate(
            f"MSE = {mse:.3f}",
            xy=(0.5, 0.43),
            xycoords="axes fraction",
            ha="center",
        )
    ax.set_axis_off()


def pair_grid(
    data: pd.DataFrame,
    weight: str | None = None,
    hist_kws: Kws = None,
    scatter_kws: Kws = None,
    **kwargs,
) -> sns.PairGrid:
    r"""
    Make a paired grid scatter plot, with histograms on the diagonal and
    correlation and mean squared error annotations on the upper triangle

    Parameters
    ----------
    data
        Data frame
    weight
        Weight for computing correlation and mean squared error
    hist_kws
        Keyword arguments for :func:`~seaborn.histplot`
    scatter_kws
        Keyword arguments for :func:`~seaborn.scatterplot`
    **kwargs
        Additional arguments are passed to :class:`~seaborn.PairGrid`

    Returns
    -------
    PairGrid object
    """
    g = sns.PairGrid(data, **kwargs)
    g.map_diag(sns.histplot, **(hist_kws or {}))
    g.map_lower(sns.scatterplot, **(scatter_kws or {}))
    g.map_lower(_add_diagonal)
    g.map_upper(_annotate_corr_mse, weight=data[weight].to_numpy() if weight else None)
    g.add_legend()
    return g


def interactive_heatmap(
    data: pd.DataFrame,
    row_clust: pd.Series | str | None = "auto",
    col_clust: pd.Series | str | None = "auto",
    row_margin: float = 0.2,
    col_margin: float = 0.2,
    height: int = 800,
    width: int = 800,
    highlights: dict[str, str] | None = None,
    **kwargs,
):
    r"""
    Plot an interactive heatmap with clustered rows and columns

    Parameters
    ----------
    data
        Data frame to plot
    row_clust
        Row cluster assignments, "auto" to compute clusters, or None
    col_clust
        Column cluster assignments, "auto" to compute clusters, or None
    row_margin
        Margin size for row annotations
    col_margin
        Margin size for column annotations
    height
        Plot height in pixels
    width
        Plot width in pixels
    highlights
        Dictionary mapping row/column names to highlight colors
    **kwargs
        Additional arguments are passed to
        :class:`~plotly.graph_objects.Heatmap`

    Returns
    -------
    Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        raise ImportError("plotly is required for this function")
    if isinstance(row_clust, str) and row_clust == "auto":
        row_linkage, row_clust = hclust(data)
        row_leaves = leaves_list(row_linkage)
        data = data.iloc[row_leaves, :]
    if isinstance(col_clust, str) and col_clust == "auto":
        col_linkage, col_clust = hclust(data.T)
        col_leaves = leaves_list(col_linkage)
        data = data.iloc[:, col_leaves]
    heatmap = go.Heatmap(
        z=data,
        x=data.columns,
        y=data.index,
        hovertemplate="Row: %{y}<br>Column: %{x}<br>Value: %{z}<extra></extra>",
        **kwargs,
    )
    if row_clust is not None:
        row_clust = row_clust.loc[data.index].astype("category")
        row_pal = sns.color_palette(
            "tab20" if row_clust.cat.categories.size <= 20 else "husl",
            n_colors=max(row_clust.cat.categories.size, 2),
        )
        row_pal = [
            [i / (len(row_pal) - 1), f"rgb{c[0] * 255, c[1] * 255, c[2] * 255}"]
            for i, c in enumerate(row_pal)
        ]
        row_annot = [
            go.Heatmap(
                z=row_clust.cat.codes.to_frame(),
                y=row_clust.index,
                x=[""],
                text=row_clust.to_frame(),
                hovertemplate="Row: %{y}<br>Cluster: %{text}<extra></extra>",
                colorscale=row_pal,
                showscale=False,
                xaxis="x2",
                yaxis="y",
            )
        ]
        row_layout = {
            "xaxis": dict(
                domain=[row_margin, 1], minallowed=-0.5, maxallowed=data.shape[0] + 1.5
            ),
            "xaxis2": dict(domain=[0, 0.05], showticklabels=False, fixedrange=True),
        }
    else:
        row_annot = []
        row_layout = {
            "xaxis": dict(minallowed=-0.5, maxallowed=data.shape[0] + 1.5),
        }
    if col_clust is not None:
        col_clust = col_clust.loc[data.columns].astype("category")
        col_pal = sns.color_palette(
            "tab20" if col_clust.cat.categories.size <= 20 else "husl",
            n_colors=max(col_clust.cat.categories.size, 2),
        )
        col_pal = [
            [i / (len(col_pal) - 1), f"rgb{c[0] * 255, c[1] * 255, c[2] * 255}"]
            for i, c in enumerate(col_pal)
        ]
        col_annot = [
            go.Heatmap(
                z=col_clust.cat.codes.to_frame().T,
                x=col_clust.index,
                y=[""],
                text=col_clust.to_frame().T,
                hovertemplate="Column: %{x}<br>Cluster: %{text}<extra></extra>",
                colorscale=col_pal,
                showscale=False,
                xaxis="x",
                yaxis="y2",
            )
        ]
        col_layout = {
            "yaxis": dict(
                domain=[col_margin, 1], minallowed=-0.5, maxallowed=data.shape[1] + 1.5
            ),
            "yaxis2": dict(domain=[0, 0.05], showticklabels=False, fixedrange=True),
        }
    else:
        col_annot = []
        col_layout = {
            "yaxis": dict(minallowed=-0.5, maxallowed=data.shape[1] + 1.5),
        }
    layout = go.Layout(**row_layout, **col_layout, height=height, width=width)
    fig = go.Figure(data=[heatmap, *row_annot, *col_annot], layout=layout)
    for k, v in (highlights or {}).items():
        try:
            row_idx = data.index.get_loc(k)
            fig.add_shape(
                type="rect",
                x0=data.shape[1] - 0.5,
                x1=data.shape[1] + 1.5,
                y0=row_idx - 0.5,
                y1=row_idx + 0.5,
                line=dict(width=0),
                fillcolor=v,
            )
        except KeyError:
            logger.warning(f"Row {k} not found.")
        try:
            col_idx = data.columns.get_loc(k)
            fig.add_shape(
                type="rect",
                x0=col_idx - 0.5,
                x1=col_idx + 0.5,
                y0=data.shape[0] - 0.5,
                y1=data.shape[0] + 1.5,
                line=dict(width=0),
                fillcolor=v,
            )
        except KeyError:
            logger.warning(f"Column {k} not found.")
    return fig
