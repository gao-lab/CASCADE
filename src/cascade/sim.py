r"""
Simulation functions
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from numpy.typing import ArrayLike
from scipy.sparse import csr_matrix, eye
from tqdm.auto import tqdm

from .data import Targets
from .typing import RandomState
from .utils import get_random_state

Scale = float | Callable[[int, RandomState], Sequence[float]]


ACTIVATION = {
    "ident": lambda x: x,
    "tanh": np.tanh,
}


class DAGType(IntEnum):
    r"""
    Types of directed acyclic graphs

    Attributes
    ----------
    unif
        Uniform
    sf
        Scale-free
    """

    unif = 0
    sf = 1


def _simulator_core(adata: AnnData) -> AnnData:
    biadj = adata.varp["biadj"]
    if reuse_norm_in := "norm_in" in adata.varm:
        norm_in = adata.varm["norm_in"].T
    else:
        norm_in = np.empty((2, adata.n_vars))
    if reuse_norm_out := "norm_out" in adata.varm:
        norm_out = adata.varm["norm_out"].T
    else:
        norm_out = np.empty((2, adata.n_vars))
    snr = adata.var["snr"].to_numpy()
    act = adata.var["act"].to_numpy()
    exo = adata.layers["exo"]
    scale = adata.layers["scale"]
    topo = list(
        nx.topological_generations(
            nx.from_scipy_sparse_array(biadj, create_using=nx.DiGraph)
        )
    )

    prev = []
    for curr in tqdm(topo, leave=False):
        if prev:
            determined = adata.X[:, prev] @ biadj[prev, :][:, curr]
        else:  # root
            determined = exo[:, curr]
        if not reuse_norm_in:
            norm_in[:, curr] = determined.mean(axis=0), determined.std(axis=0)
        mean_in, std_in = norm_in[:, curr]
        std_in[std_in == 0] = 1
        determined -= mean_in
        determined /= std_in
        for i, j in enumerate(curr):
            determined[:, i] = ACTIVATION[act[j]](determined[:, i])
        if not reuse_norm_out:
            norm_out[:, curr] = determined.mean(axis=0), determined.std(axis=0)
        mean_out, std_out = norm_out[:, curr]
        std_out[std_out == 0] = 1
        determined -= mean_out
        determined /= std_out
        if prev:
            simulated = determined * snr[curr] + exo[:, curr]
        else:  # root
            simulated = determined * np.sqrt(snr[curr] ** 2 + 1)  # Make same variance
        adata.X[:, curr] = scale[:, curr] * simulated
        prev += curr

    adata.varm["norm_in"] = norm_in.T
    adata.varm["norm_out"] = norm_out.T
    return adata


# ----------------------------- Public functions -------------------------------


def generate_dag(
    n: int,
    m: int | float,
    type: DAGType = DAGType.unif,
    random_state: RandomState = None,
) -> nx.DiGraph:
    r"""
    Randomly generate a directed acyclic graph

    Parameters
    ----------
    n
        Number of nodes
    m
        Target in-degree
    type
        Type of DAG to generate, see :class:`DagType`
    random_state
        Random state

    Returns
    -------
    A directed acyclic graph


    .. note::

        - Integer ``m`` is interpreted as a fixed in-degree
        - Floating ``m`` is interpreted as a fraction of upstream nodes
        - Nodes are named ``v0``, ``v1``, ..., ``v{n-1}``
    """
    rnd = get_random_state(random_state)
    adj = eye(n, dtype=bool, format="lil")
    for j in range(n):
        if j and type is DAGType.sf:
            prev_degs = adj[:j].sum(axis=1).A1
            p = prev_degs / prev_degs.sum()
        else:
            p = None
        m_ = m if isinstance(m, int) else round(m * j)
        i = rnd.choice(j, size=min(m_, j), p=p)
        adj[i, j] = True
    adj.setdiag(False)
    adj = adj.tocsr()
    adj.data = adj.data * np.sign(rnd.randn(adj.nnz))
    dag = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
    return nx.relabel_nodes(dag, {i: f"v{i}" for i in dag.nodes}, copy=False)


def simulate_regimes(
    dag: nx.DiGraph,
    design: Mapping[Targets, int],
    interv: Mapping[str, Scale],
    random_state: RandomState = None,
) -> AnnData:
    r"""
    Simulate interventional data based on a causal structure with multiple sets
    of intervention effect in parallel

    Parameters
    ----------
    dag
        A directed acyclic graph representing the causal structure
    design
        A mapping from intervention targets to sample numbers
    interv
        Intervention scaling factor :math:`\lambda` of each intervention target
        or sampler function of such (:math:`\lambda = 0` for knockout, :math:`0
        \lt \lambda \lt 1` for knockdown, :math:`\lambda \gt 1` for knockup)
    random_state
        Random state

    Returns
    -------
    Simulated dataset


    .. note::

        - The signal-to-noise ratio for each simulated variable should be
          provided as a node attribute in ``dag`` called ``"snr"``.
        - The activation function for each simulated variable should be provided
          as a node attribute in ``dag`` called ``"act"``.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("Causal structure is not a directed acyclic graph.")
    rnd = get_random_state(random_state)
    nodes = pd.Index(dag.nodes)
    snr = nx.get_node_attributes(dag, "snr")
    act = nx.get_node_attributes(dag, "act")
    biadj = csr_matrix(
        nx.bipartite.biadjacency_matrix(dag, nodes, nodes)
    )  # anndata does not fully support csr_array yet
    scale = []
    for targets, num in design.items():
        s = np.ones((num, len(nodes)))
        for target in targets:
            v = interv[target]
            s[:, nodes.get_loc(target)] = v(num, rnd) if callable(v) else v
        scale.append(s)
    scale = np.concatenate(scale)
    exo = rnd.normal(size=scale.shape)
    obs = pd.DataFrame(
        {
            "knockout": [",".join(nodes[row]) for row in scale == 0],
            "knockdown": [",".join(nodes[row]) for row in (scale > 0) & (scale < 1)],
            "knockup": [",".join(nodes[row]) for row in scale > 1],
        },
        index=pd.RangeIndex(scale.shape[0]).astype(str),
    )
    var = pd.DataFrame(
        {
            "snr": [snr[node] for node in nodes],
            "act": [act[node] for node in nodes],
        },
        index=nodes,
    )
    observ_mask = np.all(scale == 1, axis=1)
    interv_mask = ~observ_mask
    adata = []
    if observ_mask.any():
        observ = AnnData(
            X=np.empty((observ_mask.sum(), scale.shape[1])),
            obs=obs.loc[observ_mask],
            var=var,
            varp={"biadj": biadj},
            layers={"scale": scale[observ_mask], "exo": exo[observ_mask]},
        )
        _simulator_core(observ)
        adata.append(observ)
        logger.info("Variables will be normalized by observational samples.")
        varm = {"norm_in": observ.varm["norm_in"], "norm_out": observ.varm["norm_out"]}
    else:
        logger.warning("Variables will be normalized by interventional samples.")
        varm = {}
    if interv_mask.any():
        interv = AnnData(
            X=np.empty((interv_mask.sum(), scale.shape[1])),
            obs=obs.loc[interv_mask],
            var=var,
            varm=varm,
            varp={"biadj": biadj},
            layers={"scale": scale[interv_mask], "exo": exo[interv_mask]},
        )
        _simulator_core(interv)
        adata.append(interv)
    return ad.concat(adata, merge="same")


def simulate_random_regimes(
    dag: nx.DiGraph,
    n_obs: int,
    rate: float,
    interv: Mapping[str, Scale],
    random_state: RandomState = None,
) -> AnnData:
    r"""
    Simulate interventional data based on a causal structure with random
    interventions

    Parameters
    ----------
    dag
        A directed acyclic graph representing the causal structure
    n_obs
        Number of samples
    rate
        Interventional rate per node
    interv
        Intervention scaling factor :math:`\lambda` of each intervention target
        or sampler function of such (:math:`\lambda = 0` for knockout, :math:`0
        \lt \lambda \lt 1` for knockdown, :math:`\lambda \gt 1` for knockup)
    act
        Activation function
    snr
        Signal to noise ratio
    random_state
        Random state

    Returns
    -------
    Simulated dataset
    """
    rnd = get_random_state(random_state)
    n_per_node = round(n_obs * rate)
    design = [[] for _ in range(n_obs)]
    for node in dag.nodes:
        for i in rnd.choice(n_obs, n_per_node):
            design[i].append(node)
    design = Counter(map(Targets, design))
    return simulate_regimes(dag, design, interv, random_state=rnd)


def simulate_counterfactual(adata: AnnData, scale: ArrayLike) -> AnnData:
    r"""
    Simulate counterfactual outcome of alternative interventions based on an
    existing simulated dataset

    Parameters
    ----------
    adata
        An existing simulated dataset
    scale
        Counterfactual interventional scale matrix (same shape as ``adata``)

    Returns
    -------
    Counterfactual dataset
    """
    adata = adata.copy()
    adata.layers["scale"] = (scale := np.asarray(scale))
    nodes = adata.var_names
    adata.obs["knockout"] = [",".join(nodes[row]) for row in scale == 0]
    adata.obs["knockdown"] = [",".join(nodes[row]) for row in (scale > 0) & (scale < 1)]
    adata.obs["knockup"] = [",".join(nodes[row]) for row in scale > 0]
    return _simulator_core(adata)
