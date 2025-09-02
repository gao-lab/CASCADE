r"""
Metrics for evaluating the accuracy of inferred causal structures
"""

from collections.abc import Callable
from functools import lru_cache, reduce, wraps
from operator import or_
from statistics import mean

import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
from anndata import AnnData
from loguru import logger
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm.auto import tqdm

from .data import Targets, aggregate_obs
from .graph import acyclify, demultiplex, filter_edges, multiplex_num
from .typing import SimpleGraph
from .utils import densify

# ------------------------- Causal discovery metrics ---------------------------


TrueDiscMetricFn = Callable[[nx.DiGraph, nx.DiGraph, ...], np.floating]  # type: ignore
RespDiscMetricFn = Callable[[nx.DiGraph, ...], np.floating]  # type: ignore


def _multiplex_compat_true_metric(f: TrueDiscMetricFn) -> TrueDiscMetricFn:
    r"""
    Wraps a metric function to be compatible with multiplexed graphs, in which
    case the returned value is the average over all multiplexed graphs.
    """

    @wraps(f)
    def wrapped(true: nx.DiGraph, pred: nx.DiGraph, **kwargs) -> np.floating:
        if multiplex_num(pred):
            return mean(f(true, p, **kwargs) for p in demultiplex(pred))
        return f(true, pred, **kwargs)

    return wrapped


def _multiplex_compat_resp_metric(f: RespDiscMetricFn) -> RespDiscMetricFn:
    r"""
    Wraps a metric function to be compatible with multiplexed graphs, in which
    case the returned value is the average over all multiplexed graphs.
    """

    @wraps(f)
    def wrapped(pred: nx.DiGraph, **kwargs) -> np.floating:
        if multiplex_num(pred):
            return mean(f(p, **kwargs) for p in demultiplex(pred))
        return f(pred, **kwargs)

    return wrapped


@lru_cache
def cmp_true_pred(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str | None = None,
    scaffold: SimpleGraph | None = None,
) -> pd.DataFrame:
    r"""
    Compare the true and predicted causal graphs in a long-form data frame

    Parameters
    ----------
    true
        True causal graph
    pred
        Predicted causal graph
    edge_attr
        Prediction edge attribute (edges are taken as binary if None)
    scaffold
        Scaffold graph

    Returns
    -------
    Long-form comparison data frame


    .. caution::

        The cache is **UNSAFE** from inplace graph manipulations.
    """
    if scaffold is None:
        scaffold = nx.complete_graph(true.nodes | pred.nodes)
    if not nx.is_directed(scaffold):
        scaffold = nx.DiGraph(scaffold)
    df = nx.to_pandas_edgelist(scaffold)
    df["true"] = [true.has_edge(u, v) for u, v in zip(df["source"], df["target"])]

    if edge_attr is None:
        df["pred"] = [pred.has_edge(u, v) for u, v in zip(df["source"], df["target"])]
    else:
        pred = nx.get_edge_attributes(pred, edge_attr)
        df["pred"] = [pred.get((u, v), 0.0) for u, v in zip(df["source"], df["target"])]
    return df


@lru_cache
def optimal_cutoff(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
) -> np.floating:
    r"""
    Obtain the optimal binary classification cutoff

    Parameters
    ----------
    true
        True causal graph
    pred
        Predicted causal graph
    edge_attr
        Prediction edge attribute
    scaffold
        Scaffold graph

    Returns
    -------
    Optimal binary classification cutoff
    """
    cmp = cmp_true_pred(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    fpr, tpr, thresholds = roc_curve(cmp["true"], cmp["pred"])
    dist = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)  # Distance to top-left corner
    return thresholds[min(np.argmin(dist) + 1, dist.size - 1)]


@_multiplex_compat_true_metric
def disc_acc(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    Accuracy of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    cmp = cmp_true_pred(true, pred, scaffold=scaffold)
    return np.float64(accuracy_score(cmp["true"], cmp["pred"]))  # sklearn #27339


@_multiplex_compat_true_metric
def disc_prec(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    Precision of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    cmp = cmp_true_pred(true, pred, scaffold=scaffold)
    return precision_score(cmp["true"], cmp["pred"], zero_division=0.0)


@_multiplex_compat_true_metric
def disc_recall(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    Recall of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    cmp = cmp_true_pred(true, pred, scaffold=scaffold)
    return recall_score(cmp["true"], cmp["pred"])


@_multiplex_compat_true_metric
def disc_f1(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    F1 score of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    cmp = cmp_true_pred(true, pred, scaffold=scaffold)
    return f1_score(cmp["true"], cmp["pred"])


@_multiplex_compat_true_metric
def disc_auroc(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
) -> np.floating:
    r"""
    Area under ROC curve of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    cmp = cmp_true_pred(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    return roc_auc_score(cmp["true"], cmp["pred"])


@_multiplex_compat_true_metric
def disc_ap(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
) -> np.floating:
    r"""
    Average precision of the predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    cmp = cmp_true_pred(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    return average_precision_score(cmp["true"], cmp["pred"])


@_multiplex_compat_true_metric
def disc_shd(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    Structural hamming distance between the true and predicted causal graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    cmp = cmp_true_pred(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    return (cmp["true"] - cmp["pred"]).abs().sum()


@_multiplex_compat_true_metric
def disc_sid(
    true: nx.DiGraph,
    pred: nx.DiGraph,
    edge_attr: str = "weight",
    scaffold: SimpleGraph | None = None,
    cutoff: float | None = None,
) -> np.floating:
    r"""
    Structural interventional distance between the true and predicted causal
    graph


    .. note::

        See :func:`cmp_true_pred` for argument descriptions.
    """
    from .ri import structIntervDist

    if cutoff is None:
        cutoff = optimal_cutoff(true, pred, edge_attr=edge_attr, scaffold=scaffold)
    pred = filter_edges(pred, cutoff=cutoff)
    if not nx.is_directed_acyclic_graph(true):
        raise ValueError("The true causal graph must be a DAG")
    if not nx.is_directed_acyclic_graph(pred):
        logger.warning("Acyclifying the predicted causal graph.")
        pred = acyclify(pred)
    nodes = sorted(true.nodes | pred.nodes)
    true = nx.to_scipy_sparse_array(true, nodelist=nodes, weight=None, format="coo")
    pred = nx.to_scipy_sparse_array(pred, nodelist=nodes, weight=None, format="coo")
    return np.float64(structIntervDist(true, pred))


def annot_resp(pred: nx.DiGraph, adata: AnnData, interv_key: str) -> None:
    r"""
    Annotate interventional responsiveness for a predicted causal graph

    Parameters
    ----------
    pred
        Predicted causal graph
    adata
        Interventional dataset
    interv_key
        Key in :attr:`~anndata.AnnData.obs` for the intervention variable
    """

    def row_fmt(x):
        return x.tocsr() if issparse(x) else x

    def col_fmt(x):
        return x.tocsc() if issparse(x) else x

    def cohens_d(u, v):
        return (np.mean(v) - np.mean(u)) / np.sqrt(
            (np.var(u) * u.size + np.var(v) * v.size) / (u.size + v.size)
        )

    var_names = adata.var_names
    targets = adata.obs[interv_key].map(Targets)
    all_targets = reduce(or_, targets.unique())
    X = row_fmt(adata.X)

    ctrl = col_fmt(X[targets.map(len) == 0])
    interv = {t: col_fmt(X[targets.map(lambda x: t in x)]) for t in tqdm(all_targets)}

    for (x, y), attr in tqdm(dict(pred.edges).items(), total=pred.number_of_edges()):
        if x not in interv:
            attr["fwd_pval"] = attr["fwd_diff"] = attr["fwd_dist"] = np.nan
        else:
            yidx = var_names.get_loc(y)
            u = densify(ctrl[:, yidx]).ravel()
            v = densify(interv[x][:, yidx]).ravel()
            attr["fwd_pval"] = scipy.stats.ks_2samp(u, v).pvalue
            attr["fwd_diff"] = cohens_d(u, v)
            attr["fwd_dist"] = abs(attr["fwd_diff"])
        if y not in interv:
            attr["rev_pval"] = attr["rev_diff"] = attr["rev_dist"] = np.nan
        else:
            xidx = var_names.get_loc(x)
            u = densify(ctrl[:, xidx]).ravel()
            v = densify(interv[y][:, xidx]).ravel()
            attr["rev_pval"] = scipy.stats.ks_2samp(u, v).pvalue
            attr["rev_diff"] = cohens_d(u, v)
            attr["rev_dist"] = abs(attr["rev_diff"])


@_multiplex_compat_resp_metric
def disc_resp_dist(pred: nx.DiGraph, cutoff: float = 0.5) -> np.floating:
    r"""
    Responsiveness distance of the predicted causal graph

    Parameters
    ----------
    pred
        Predicted causal graph
    cutoff
        Binary classification cutoff

    Returns
    -------
    Responsiveness distance
    """
    pred = filter_edges(pred, cutoff=cutoff)
    return np.nanmean([attr["fwd_dist"] for attr in pred.edges.values()])


@_multiplex_compat_resp_metric
def disc_resp_dist_diff(pred: nx.DiGraph, cutoff: float = 0.5) -> np.floating:
    r"""
    Responsiveness distance difference of the predicted causal graph

    Parameters
    ----------
    pred
        Predicted causal graph
    cutoff
        Binary classification cutoff

    Returns
    -------
    Responsiveness distance difference
    """
    pred = filter_edges(pred, cutoff=cutoff)
    return np.nanmean(
        [attr["fwd_dist"] - attr["rev_dist"] for attr in pred.edges.values()]
    )


@_multiplex_compat_resp_metric
def disc_resp_acc(
    pred: nx.DiGraph, cutoff: float = 0.5, sig: float = 0.1
) -> np.floating:
    r"""
    Responsiveness accuracy of the predicted causal graph

    Parameters
    ----------
    pred
        Predicted causal graph
    cutoff
        Binary classification cutoff
    sig
        Significance level

    Returns
    -------
    Responsiveness accuracy
    """
    pred = filter_edges(pred, cutoff=cutoff)
    n_fwd = len(
        [
            e
            for e, attr in pred.edges.items()
            if attr["fwd_pval"] < sig and attr["rev_pval"] > sig
        ]
    )
    n_rev = len(
        [
            e
            for e, attr in pred.edges.items()
            if attr["fwd_pval"] > sig and attr["rev_pval"] < sig
        ]
    )
    with np.errstate(invalid="ignore"):
        return np.divide(n_fwd, n_fwd + n_rev)


# --------------------- Counterfactual prediction metrics ----------------------


def _ctfact_prep(
    ctrl: AnnData, true: AnnData, pred: AnnData, by: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r"""
    Prepare datasets for counterfactual metric computation

    Parameters
    ----------
    ctrl
        Control dataset
    true
        True interventional effect dataset
    pred
        Predicted counterfactual effect dataset
    by
        Intervention variable to group by in the :attr:`~anndata.AnnData.obs`
        slot

    Returns
    -------
    Control dataset aggregated by intervention
    True interventional effect dataset aggregated by intervention
    Predicted counterfactual effect dataset aggregated by intervention
    """
    ctrl_agg = aggregate_obs(ctrl, by, X_agg="mean", obs_agg={by: "majority"})
    true_agg = aggregate_obs(true, by, X_agg="mean", obs_agg={by: "majority"})
    pred_agg = aggregate_obs(pred, by, X_agg="mean", obs_agg={by: "majority"})
    ctrl_agg.obs.set_index(by, inplace=True)
    true_agg.obs.set_index(by, inplace=True)
    pred_agg.obs.set_index(by, inplace=True)
    ctrl_df = ctrl_agg.to_df()
    true_df = true_agg.to_df()
    pred_df = pred_agg.to_df()
    if ctrl_df.shape[0] != 1 or ctrl_df.index[0] != "":
        raise ValueError("Invalid control dataset")
    if set(true_df.index) != set(pred_df.index):
        raise ValueError("Interventions in `pred` does not match `true`")
    return ctrl_df, true_df, pred_df


def ctfact_mse(
    ctrl: AnnData,
    true: AnnData,
    pred: AnnData,
    by: str,
    top_de: int = None,
    exclude_self: bool = False,
    de_key: str = "rank_genes_groups",
) -> pd.DataFrame:
    r"""
    Mean squared errors of counterfactual prediction

    Parameters
    ----------
    ctrl
        Control dataset
    true
        True interventional effect dataset
    pred
        Predicted counterfactual effect dataset
    by
        Intervention variable to group by in the :attr:`~anndata.AnnData.obs`
        slot
    top_de
        Number of top differentially expressed genes to consider
    exclude_self
        Whether to exclude the perturbed genes themselves
    de_key
        Key to the differential expression results

    Returns
    -------
    Counterfactual metric data frame consisting of columns:

        - "true_mse"
        - "pred_mse"
        - "normalized_mse"
    """
    ctrl_df, true_df, pred_df = _ctfact_prep(ctrl, true, pred, by)
    true_se = (true_df - ctrl_df.iloc[0]).pow(2)
    pred_se = (pred_df - true_df).pow(2)

    all_vars = set(true.var_names)
    true_mse, pred_mse = [], []
    de_groups = [i for i in true.uns[de_key]["names"].dtype.names if i in true_se.index]
    for g in de_groups:
        exclude_vars = set(g.split(",")) if exclude_self else set()
        degs = sc.get.rank_genes_groups_df(true, g, key=de_key)["names"]
        degs = degs[degs.isin(all_vars - exclude_vars)]
        degs = degs.head(n=top_de or degs.size)
        true_mse.append(true_se.loc[g, degs].mean())
        pred_mse.append(pred_se.loc[g, degs].mean())
    true_mse = pd.Series(true_mse, index=de_groups)
    pred_mse = pd.Series(pred_mse, index=de_groups)

    normalized_mse = pred_mse / true_mse
    mse_df = pd.DataFrame(
        {
            "true_mse": true_mse,
            "pred_mse": pred_mse,
            "normalized_mse": normalized_mse,
        }
    )
    return mse_df


def ctfact_delta_pcc(
    ctrl: AnnData,
    true: AnnData,
    pred: AnnData,
    by: str,
    top_de: int = None,
    exclude_self: bool = False,
    de_key: str = "rank_genes_groups",
) -> pd.DataFrame:
    r"""
    Pearson correlation coefficient of counterfactual delta

    Parameters
    ----------
    ctrl
        Control dataset
    true
        True interventional effect dataset
    pred
        Predicted counterfactual effect dataset
    by
        Intervention variable to group by in the :attr:`~anndata.AnnData.obs`
        slot
    top_de
        Number of top differentially expressed genes to consider
    exclude_self
        Whether to exclude the perturbed genes themselves
    de_key
        Key to the differential expression results

    Returns
    -------
    Counterfactual metric data frame containing column "delta_pcc"
    """
    ctrl_df, true_df, pred_df = _ctfact_prep(ctrl, true, pred, by)
    true_delta = true_df - ctrl_df.iloc[0]
    pred_delta = pred_df - ctrl_df.iloc[0]

    all_vars = set(true.var_names)
    de_groups = [
        i for i in true.uns[de_key]["names"].dtype.names if i in true_delta.index
    ]
    pcc = []
    for g in de_groups:
        exclude_vars = set(g.split(",")) if exclude_self else set()
        degs = sc.get.rank_genes_groups_df(true, g, key=de_key)["names"]
        degs = degs[degs.isin(all_vars - exclude_vars)]
        degs = degs.head(n=top_de or degs.size)
        pcc.append(true_delta.loc[g, degs].corr(pred_delta.loc[g, degs]))
    pcc = pd.Series(pcc, index=de_groups)
    pcc_df = pd.DataFrame({"delta_pcc": pcc})
    return pcc_df


def ctfact_dir_acc(
    ctrl: AnnData,
    true: AnnData,
    pred: AnnData,
    by: str,
    top_de: int = None,
    exclude_self: bool = False,
    de_key: str = "rank_genes_groups",
) -> pd.DataFrame:
    r"""
    Directional accuracy of counterfactual predictions

    Parameters
    ----------
    ctrl
        Control dataset
    true
        True interventional effect dataset
    pred
        Predicted counterfactual effect dataset
    by
        Intervention variable to group by in the :attr:`~anndata.AnnData.obs`
        slot
    top_de
        Number of top differentially expressed genes to consider
    exclude_self
        Whether to exclude the perturbed genes themselves
    de_key
        Key to the differential expression results

    Returns
    -------
    Counterfactual metric data frame containing column "dir_acc"
    """
    ctrl_df, true_df, pred_df = _ctfact_prep(ctrl, true, pred, by)
    true_sign = np.sign(true_df - ctrl_df.iloc[0])
    pred_sign = np.sign(pred_df - ctrl_df.iloc[0])
    sign_match = true_sign.eq(pred_sign)

    all_vars = set(true.var_names)
    de_groups = [
        i for i in true.uns[de_key]["names"].dtype.names if i in sign_match.index
    ]
    acc = []
    for g in de_groups:
        exclude_vars = set(g.split(",")) if exclude_self else set()
        degs = sc.get.rank_genes_groups_df(true, g, key=de_key)["names"]
        degs = degs[degs.isin(all_vars - exclude_vars)]
        degs = degs.head(n=top_de or degs.size)
        acc.append(sign_match.loc[g, degs].mean())
    acc = pd.Series(acc, index=de_groups)
    acc_df = pd.DataFrame({"dir_acc": acc})
    return acc_df


# ------------------------ Intervention design metrics -------------------------


def dsgn_hrc_exact(designs: dict[str, pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Exact hit-rate curve for intervention design

    Parameters
    ----------
    designs
        Mapping from true interventions to designed intervention scores

    Returns
    -------
    Quantiles of the designed intervention scores
    Exact hit rates at the quantiles
    """
    l = []
    for true, design in designs.items():
        rank = design.rank(ascending=False)
        try:
            l.append(rank.loc[true] / rank.size)
        except KeyError:
            logger.warning(f"Exact intervention {true} not found in its design!")
            l.append(1.0)
    qtl = np.asarray([0.0, *sorted(l), 1.0])
    hr = np.asarray([*[i / len(l) for i in range(len(l) + 1)], 1.0])
    return qtl, hr


def dsgn_hrc_partial(designs: dict[str, pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Partial hit-rate curve for intervention design

    Parameters
    ----------
    designs
        Mapping from true interventions to designed intervention scores

    Returns
    -------
    Quantiles of the designed intervention scores
    Partial hit rates at the quantiles
    """
    l = []
    for true, design in designs.items():
        rank = design.rank(ascending=False)
        true = Targets(true)
        partial_match = rank.index.map(lambda x: bool(Targets(x) & true))
        if partial_match.any():
            l.append(rank.loc[partial_match].min() / rank.size)
        else:
            logger.warning(f"No partial match for {true} found in its design!")
            l.append(1.0)
    qtl = np.asarray([0.0, *sorted(l), 1.0])
    hr = np.asarray([*[i / len(l) for i in range(len(l) + 1)], 1.0])
    return qtl, hr


def dsgn_auhrc_exact(designs: dict[str, pd.Series]) -> np.floating:
    r"""
    Area under the exact hit-rate curve for intervention design,
    see :func:`dsgn_hrc_exact`

    Parameters
    ----------
    designs
        Mapping from true interventions to designed intervention scores

    Returns
    -------
    Area under the exact hit-rate curve
    """
    qtl, hr = dsgn_hrc_exact(designs)
    return auc(qtl, hr)


def dsgn_auhrc_partial(designs: dict[str, pd.Series]) -> np.floating:
    r"""
    Area under the partial hit-rate curve for intervention design,
    see :func:`dsgn_hrc_partial`

    Parameters
    ----------
    designs
        Mapping from true interventions to designed intervention scores

    Returns
    -------
    Area under the partial hit-rate curve
    """
    qtl, hr = dsgn_hrc_partial(designs)
    return auc(qtl, hr)
