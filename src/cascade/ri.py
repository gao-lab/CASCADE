r"""
R interface
"""

from collections.abc import Iterable

import networkx as nx
import pandas as pd
from anndata import AnnData
from scipy.sparse import sparray

from .data import Targets
from .utils import MissingDependencyError

try:
    from rpy2 import robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri
    from rpy2.robjects.methods import RS4
    from rpy2.robjects.packages import importr
except ImportError:
    raise MissingDependencyError("rpy2")


# -------------------------------- Converters ----------------------------------


def _digraph2graphnel(digraph: nx.DiGraph) -> RS4:
    graph = importr("graph")
    nodes = ro.StrVector(list(digraph.nodes))
    edgeL = ro.ListVector(
        {
            n: ro.ListVector({"edges": ro.StrVector(list(digraph.successors(n)))})
            for n in nodes
        }
    )
    return graph.graphNEL(nodes, edgeL, edgemode="directed")


def _graphnel2digraph(graphnel: RS4) -> nx.DiGraph:
    graph = importr("graph")
    edges = graph.edges(graphnel)
    return nx.DiGraph(dict(zip(edges.names, map(list, edges))))


def _sparray2dgCMatrix(mat: sparray) -> RS4:
    Matrix = importr("Matrix")
    mat = mat.tocoo()
    return Matrix.sparseMatrix(
        i=ro.IntVector(mat.row + 1),
        j=ro.IntVector(mat.col + 1),
        x=ro.FloatVector(mat.data),
        dims=ro.IntVector(mat.shape),
    )


# --------------------------------- Wrappers -----------------------------------


def dag2essgraph(
    dag: nx.DiGraph, targets_list: Iterable[Targets] | None = None
) -> nx.DiGraph:
    r"""
    Convert DAG to essential graph representing its Markov equivalence class

    Parameters
    ----------
    dag
        DAG to convert
    targets_list
        A list of interventional targets

    Returns
    -------
    Essential graph representing the Markov equivalence class of the DAG


    .. note::

        This is a wrapper of the R function ``pcalg::dag2essgraph``.
    """
    pcalg = importr("pcalg")
    targets_list = targets_list or []
    nodes = pd.Index(dag.nodes)
    targets = [ro.IntVector(nodes.get_indexer(targets) + 1) for targets in targets_list]
    dag = _digraph2graphnel(dag)
    ess = pcalg.dag2essgraph(dag, targets)
    return _graphnel2digraph(ess)


def structIntervDist(true: sparray, pred: sparray) -> float:
    r"""
    Structural Interventional Distance (SID)

    Parameters
    ----------
    true
        True adjacency matrix
    pred
        Predicted adjacency matrix

    Returns
    -------
    Structural Interventional Distance


    .. note::

        This is a wrapper of the R function ``SID::structIntervDist``.
    """
    sid = importr("SID")
    true = _sparray2dgCMatrix(true)
    pred = _sparray2dgCMatrix(pred)
    return sid.structIntervDist(true, pred).rx2("sid")[0]


def pc(
    adata: AnnData,
    scaffold: nx.Graph | None = None,
    alpha: float = 0.01,
    numCores: int = 1,
    verbose: bool = False,
) -> nx.DiGraph:
    r"""
    PC algorithm

    Parameters
    ----------
    adata
        Input dataset
    scaffold
        Scaffold graph
    alpha
        Significance level
    numCores
        Number of cores to use
    verbose
        Verbosity

    Returns
    -------
    Inferred causal graph


    .. note::

        This is a wrapper of the R function ``pcalg::pc``.
    """
    pcalg = importr("pcalg")
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().py2rpy(adata.to_df())
    with (ro.default_converter + numpy2ri.converter).context():
        if scaffold is None:
            fixedGaps = ro.r("NULL")
        else:
            fixedGaps = ro.conversion.get_conversion().py2rpy(
                ~nx.to_numpy_array(
                    nx.Graph(scaffold),
                    nodelist=adata.var_names,
                    dtype=bool,
                    weight=None,
                )
            )
    pcAlgo = pcalg.pc(
        suffStat=ro.ListVector({"C": ro.r("cor")(df), "n": ro.r("nrow")(df)}),
        indepTest=pcalg.gaussCItest,
        alpha=alpha,
        labels=ro.r("colnames")(df),
        fixedGaps=fixedGaps,
        skel_method="stable.fast" if numCores > 1 else "stable",
        numCores=numCores,
        verbose=verbose,
    )
    digraph = _graphnel2digraph(pcAlgo.slots["graph"])
    nx.set_edge_attributes(digraph, 1.0, "weight")
    return digraph


def ges(
    adata: AnnData,
    scaffold: nx.Graph | None = None,
    score: str = "GaussL0penObsScore",
    verbose: bool = False,
) -> nx.DiGraph:
    r"""
    GES algorithm

    Parameters
    ----------
    adata
        Input dataset
    scaffold
        Scaffold graph
    score
        Score function
    verbose
        Verbosity

    Returns
    -------
    Inferred causal graph

    .. note::

        This is a wrapper of the R function ``pcalg::ges``.
    """
    pcalg = importr("pcalg")
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().py2rpy(adata.to_df())
    with (ro.default_converter + numpy2ri.converter).context():
        if scaffold is None:
            fixedGaps = ro.r("NULL")
        else:
            fixedGaps = ro.conversion.get_conversion().py2rpy(
                ~nx.to_numpy_array(
                    nx.Graph(scaffold),
                    nodelist=adata.var_names,
                    dtype=bool,
                    weight=None,
                )
            )
    score = ro.r("new")(score, df)
    essgraph = pcalg.ges(score, fixedGaps=fixedGaps, verbose=verbose).rx2("essgraph")
    digraph = _graphnel2digraph(ro.r("as")(essgraph, "graphNEL"))
    nx.set_edge_attributes(digraph, 1.0, "weight")
    return digraph


def gies(
    adata: AnnData,
    interv_key: str,
    scaffold: nx.Graph | None = None,
    score: str = "GaussL0penIntScore",
    verbose: bool = False,
) -> nx.DiGraph:
    r"""
    GIES algorithm

    Parameters
    ----------
    adata
        Input dataset
    interv_key
        Key in :attr:`~anndata.AnnData.obs` containing interventional targets
    scaffold
        Scaffold graph
    score
        Score function
    verbose
        Verbosity

    Returns
    -------
    Inferred causal graph


    .. note::

        This is a wrapper of the R function ``pcalg::gies``.
    """
    pcalg = importr("pcalg")
    with (ro.default_converter + pandas2ri.converter).context():
        df = ro.conversion.get_conversion().py2rpy(adata.to_df())
    target_pd_series = adata.obs[interv_key].map(Targets)
    target_pd_index = pd.Index(sorted(set(target_pd_series)))
    target_index = ro.IntVector(target_pd_index.get_indexer(target_pd_series) + 1)
    targets = ro.ListVector.from_length(len(target_pd_index))
    for i, t in enumerate(target_pd_index):
        targets[i] = ro.IntVector(sorted(adata.var_names.get_indexer(t) + 1))
    with (ro.default_converter + numpy2ri.converter).context():
        if scaffold is None:
            fixedGaps = ro.r("NULL")
        else:
            fixedGaps = ro.conversion.get_conversion().py2rpy(
                ~nx.to_numpy_array(
                    nx.Graph(scaffold),
                    nodelist=adata.var_names,
                    dtype=bool,
                    weight=None,
                )
            )
    score = ro.r("new")(score, df, targets, target_index)
    essgraph = pcalg.gies(score, fixedGaps=fixedGaps, verbose=verbose).rx2("essgraph")
    digraph = _graphnel2digraph(ro.r("as")(essgraph, "graphNEL"))
    nx.set_edge_attributes(digraph, 1.0, "weight")
    return digraph
