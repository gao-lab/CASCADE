r"""
Utilities for manipulating causal graphs
"""

from collections.abc import Callable, Collection
from functools import lru_cache, reduce
from itertools import chain
from operator import or_
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from scipy.sparse import eye
from tqdm.auto import tqdm

from .typing import SimpleGraph


def multiplex(
    *graphs: SimpleGraph, edge_attr: str = "weight", na_fill: Any = 0.0
) -> SimpleGraph:
    r"""
    Combine multiple graphs into a single graph by multiplexing an edge
    attribute

    Parameters
    ----------
    *graphs
        Graphs to be multiplexed
    edge_attr
        Edge attribute to be multiplexed
    na_fill
        Value to fill in when an edge is not found in certain graphs

    Returns
    -------
    Multiplexed graph


    .. note::

        All attributes except ``edge_attr`` will be dropped, because it is
        generally not obvious how attributes of nodes and edges missing in
        certain multiplexed graphs should be filled.
    """
    graph_type = {type(graph) for graph in graphs}
    if len(graph_type) > 1:
        raise TypeError("Graphs must be of the same type")
    graph_type = graph_type.pop()
    multiplexed = graph_type()
    multiplexed.add_nodes_from(chain(*(graph.nodes for graph in graphs)))
    multiplexed.add_edges_from(chain(*(graph.edges for graph in graphs)))
    attrs_list = [nx.get_edge_attributes(graph, edge_attr) for graph in graphs]
    for e, attr in tqdm(multiplexed.edges.items(), leave=False):
        attr[edge_attr] = [attrs.get(e, na_fill) for attrs in attrs_list]
    return multiplexed


@lru_cache
def multiplex_num(graph: SimpleGraph, edge_attr: str = "weight") -> int:
    r"""
    Get the number of multiplexed graphs according to an edge attribute

    Parameters
    ----------
    graph
        A multiplex graph
    edge_attr
        Multiplexed edge attribute

    Returns
    -------
    Multiplex number


    .. note::

        Singularly multiplexed graph has num = 1, while non-multiplexed graph
        has num = 0.


    .. caution::

        The cache is **UNSAFE** from inplace graph manipulations.
    """
    attrs = nx.get_edge_attributes(graph, edge_attr)
    num = {len(val) if isinstance(val, Collection) else 0 for val in attrs.values()}
    if len(num) > 1:
        raise ValueError("Edge attribute must have consistent size")
    if len(num) == 0:
        return 1  # Empty graph is treated as not multiplexed
    return num.pop()


@lru_cache
def demultiplex(graph: SimpleGraph, edge_attr: str = "weight") -> list[SimpleGraph]:
    r"""
    Split one single graph into multiple graphs by demultiplexing an edge attribute

    Parameters
    ----------
    graph
        Graph to be demultiplexed
    edge_attr
        Edge attribute to be demultiplexed

    Returns
    -------
    List of demultiplexed graphs


    .. caution::

        The cache is **UNSAFE** from inplace graph manipulations.
    """
    num = multiplex_num(graph, edge_attr=edge_attr)
    if num == 0:  # Non-multiplexed
        return graph
    attrs = nx.get_edge_attributes(graph, edge_attr)
    demultiplexed = [graph.copy() for _ in range(num)]
    for e, val in tqdm(attrs.items(), leave=False):
        for i, g in enumerate(demultiplexed):
            g.edges[e][edge_attr] = val[i]
    return demultiplexed


def map_edges(
    graph: SimpleGraph,
    edge_attr: str = "weight",
    fn: Callable = lambda x: x,
) -> SimpleGraph:
    r"""
    Map edge attribute by a function

    Parameters
    ----------
    graph
        Graph to be mapped
    edge_attr
        Edge attribute to be mapped
    fn
        Mapping function

    Returns
    -------
    Mapped graph
    """
    mapped = graph.copy()
    for attr in mapped.edges.values():
        attr[edge_attr] = fn(attr[edge_attr])
    return mapped


def filter_edges(
    graph: SimpleGraph,
    edge_attr: str = "weight",
    cutoff: float | None = None,
    n_top: int | None = None,
) -> SimpleGraph:
    r"""
    Filter graph by an edge attribute

    Parameters
    ----------
    graph
        Graph to be filtered
    edge_attr
        Edge attribute used to filter the graph
    cutoff
        Cutoff value for the edge attribute
    n_top
        Number of top edges to be kept

    Returns
    -------
    Filtered graph


    .. note::

        Exactly one of ``cutoff`` and ``n_top`` should be specified.
    """
    if (cutoff is None) == (n_top is None):
        raise ValueError("Exactly one of cutoff and n_top should be specified")
    edge_attr = nx.get_edge_attributes(graph, edge_attr)
    if cutoff is None:
        cutoff = sorted(edge_attr.values(), reverse=True)[n_top]
    filtered = type(graph)(**graph.graph)
    filtered.add_nodes_from(graph.nodes.items())
    filtered.add_edges_from(
        (*e, graph.edges[e]) for e, attr in edge_attr.items() if attr > cutoff
    )
    return filtered


def acyclify(digraph: nx.DiGraph, edge_attr: str = "weight") -> nx.DiGraph:
    r"""
    Acyclify a directed graph by iteratively removing cycle-inducing edges with
    the lowest weights

    Parameters
    ----------
    digraph
        Directed graph
    edge_attr
        Attribute key for edge weights

    Returns
    -------
    Acyclic directed graph


    .. caution::

        This might not be reproducible due to the unstable order of identified
        cycles.
    """
    if nx.is_directed_acyclic_graph(digraph):
        return digraph
    dag = digraph.copy()
    attrs = nx.get_edge_attributes(dag, edge_attr)
    for _ in tqdm(range(dag.number_of_edges()), mininterval=1, leave=False):
        try:
            cycle = nx.find_cycle(dag)
        except nx.NetworkXNoCycle:
            break
        min_edge = min(cycle, key=lambda e: attrs[e])
        dag.remove_edge(*min_edge)
    return dag


def marginalize(digraph: nx.DiGraph, margin: Collection, max_steps: int) -> nx.DiGraph:
    r"""
    Marginalize a directed graph by keeping only a subset of observed nodes,
    optionally inferring indirect connections up to a maximal number of steps
    mediated by latent nodes.

    Parameters
    ----------
    digraph
        Directed graph
    margin
        A list of marginal nodes
    max_steps
        The maximal number of steps to infer indirect edges

    Returns
    -------
    Marginalized graph


    .. note::

        A new edge attribute "marginalize" is added that indicates whether
        the edge is direct or indirect.
    """
    if not nx.is_directed(digraph):
        raise TypeError("Input graph must be directed")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    margin = set(margin)
    if missing := margin - digraph.nodes:
        logger.warning(
            f"{len(missing)} nodes are missing from the input graph "
            "and will be ignored."
        )
    marginalized = nx.DiGraph(**digraph.graph)
    marginalized.add_nodes_from((u, d) for u, d in digraph.nodes.items() if u in margin)
    marginalized.add_edges_from(
        [
            (u, v, d)
            for (u, v), d in digraph.edges.items()
            if u in margin and v in margin
        ],
        marginalize="direct",
    )
    if max_steps == 0:
        return marginalized

    nodelist = pd.Index(digraph.nodes)
    margin, latent = list(marginalized.nodes), list(digraph.nodes - margin)
    adj = nx.to_scipy_sparse_array(digraph, nodelist=nodelist, weight=None)
    margin_idx = nodelist.get_indexer(margin)
    latent_idx = nodelist.get_indexer(latent)
    adj_margin_margin = adj[margin_idx, :][:, margin_idx]
    adj_margin_latent = adj[margin_idx, :][:, latent_idx]
    adj_latent_margin = adj[latent_idx, :][:, margin_idx]
    adj_latent_latent = adj[latent_idx, :][:, latent_idx]

    accumulator = eye(latent_idx.size)
    latent_steps = []
    for step in range(max_steps):
        accumulator = accumulator @ adj_latent_latent if step else accumulator
        latent_steps.append(accumulator)
    inf_latent_latent = sum(latent_steps)
    inf_margin_margin = adj_margin_latent @ inf_latent_latent @ adj_latent_margin

    inf_margin_margin = (
        inf_margin_margin.astype(bool).astype(int)
        - adj_margin_margin.astype(bool).astype(int)
    ) > 0
    inf = nx.from_scipy_sparse_array(inf_margin_margin, create_using=nx.DiGraph)
    nx.relabel_nodes(inf, dict(enumerate(margin)), copy=False)
    marginalized.add_edges_from(inf.edges, marginalize="indirect")
    return marginalized


def assemble_scaffolds(*graphs: SimpleGraph, nodes: list[str] = None) -> nx.DiGraph:
    r"""
    Assemble multiple scaffold graphs into a heterogeneous one given a specific
    node list

    Parameters
    ----------
    *graphs
        Scaffold graphs to assemble
    nodes
        Node list

    Returns
    -------
    Assembled scaffold graph
    """
    graphs = [
        marginalize(
            graph.to_directed(),
            nodes,
            max_steps=graph.graph.get("marginalize_steps", 0),
        )
        for graph in graphs
    ]
    for graph in graphs:
        nx.set_edge_attributes(
            graph, graph.graph.get("data_source", "unknown"), name="data_source"
        )
        nx.set_edge_attributes(
            graph, graph.graph.get("evidence_type", "unknown"), name="evidence_type"
        )
    assembled = nx.compose_all(graphs)
    assembled.add_nodes_from(nodes)
    assembled.remove_edges_from([(v, v) for v in assembled.nodes])
    return assembled


def node_stats(graph: nx.DiGraph) -> pd.DataFrame:
    r"""
    Get node statistics of a graph

    Parameters
    ----------
    graph
        Graph

    Returns
    -------
    Node statistics
    """
    topo_gens = (
        {g: i for i, gen in enumerate(nx.topological_generations(graph)) for g in gen}
        if nx.is_directed_acyclic_graph(graph)
        else {}
    )
    rows = [
        {
            "node": n,
            "in_degree": graph.in_degree(n),
            "out_degree": graph.out_degree(n),
            "n_ancestors": len(nx.ancestors(graph, n)),
            "n_descendants": len(nx.descendants(graph, n)),
            "topo_gen": topo_gens.get(n, np.nan),
        }
        for n in graph.nodes
    ]
    return pd.DataFrame(rows).set_index("node")


def annotate_explanation(
    digraph: nx.DiGraph, ctfact: AnnData, causal_map: pd.DataFrame, cutoff: float = 0.1
) -> nx.DiGraph:
    r"""
    Annotate counterfactual explanation to the causal graph

    Parameters
    ----------
    digraph
        Causal graph (from :meth:`~cascade.model.CASCADE.export_causal_graph`)
    ctfact
        Dataset with counterfactual explanation (from
        :meth:`~cascade.model.CASCADE.explain`)
    causal_map
        Causal map (from :meth:`~cascade.model.CASCADE.export_causal_map`)
    cutoff
        Minimal cutoff of absolute total change for a gene to be annotated with
        contributions (small changes cannot be reliably annotated)

    Returns
    -------
    Annotated causal graph


    .. tip::

        It is strongly recommended to limit cells to a single perturbation for
        use as ``ctfact``.
    """
    nil = ctfact.layers["X_nil"].mean(axis=(0, -1)).astype(float)
    ctrb_i = ctfact.layers["X_ctrb_i"].mean(axis=(0, -1)).astype(float)
    ctrb_s = ctfact.layers["X_ctrb_s"].mean(axis=(0, -1)).astype(float)
    ctrb_z = ctfact.layers["X_ctrb_z"].mean(axis=(0, -1)).astype(float)
    ctrb_ptr = ctfact.layers["X_ctrb_ptr"].mean(axis=(0, -1)).astype(float)
    tot = ctfact.layers["X_tot"].mean(axis=(0, -1)).astype(float)

    nil = pd.Series(nil, index=ctfact.var_names)
    ctrb_i = pd.Series(ctrb_i, index=ctfact.var_names)
    ctrb_s = pd.Series(ctrb_s, index=ctfact.var_names)
    ctrb_z = pd.Series(ctrb_z, index=ctfact.var_names)
    ctrb_ptr = pd.DataFrame(ctrb_ptr, index=ctfact.var_names)
    tot = pd.Series(tot, index=ctfact.var_names)

    diff_tot = tot - nil
    diff_i = ctrb_i - nil
    diff_s = ctrb_s - nil
    diff_z = ctrb_z - nil
    diff_ptr = ctrb_ptr.sub(nil, axis="index")
    ann_tot = pd.DataFrame(
        {
            "sign_tot": np.sign(diff_tot),
            "diff_tot": diff_tot,
        }
    )

    mask = diff_tot.abs() > cutoff  # Only annotate contribution for large effects
    diff_tot = diff_tot.loc[mask]
    diff_i = diff_i.loc[mask]
    diff_s = diff_s.loc[mask]
    diff_z = diff_z.loc[mask]
    diff_ptr = diff_ptr.loc[mask]
    causal_map = causal_map.loc[mask]
    ann_sz = pd.DataFrame(
        {
            "sign_i": np.sign(diff_i),
            "sign_s": np.sign(diff_s),
            "sign_z": np.sign(diff_z),
            "frac_i": np.clip(diff_i / diff_tot, 0, 1),
            "frac_s": np.clip(diff_s / diff_tot, 0, 1),
            "frac_z": np.clip(diff_z / diff_tot, 0, 1),
        }
    )

    sign_ptr: pd.DataFrame = np.sign(diff_ptr)
    sign_ptr = sign_ptr.reset_index().melt(id_vars="index", value_name="sign")
    frac_ptr = np.clip(diff_ptr.div(diff_tot, axis="index"), 0, 1)
    frac_ptr = frac_ptr.reset_index().melt(id_vars="index", value_name="frac")
    causal_map = causal_map.reset_index().melt(id_vars="index", value_name="parent")
    ann_ptr = reduce(
        pd.merge, [causal_map.query("parent != '<pad>'"), frac_ptr, sign_ptr]
    )
    ann_ptr_sum = ann_ptr.groupby("index")["frac"].sum().to_frame()

    digraph = digraph.copy()
    digraph.graph["cutoff"] = cutoff
    for idx, row in ann_tot.iterrows():
        attr = digraph.nodes[idx]
        attr["sign_tot"] = "down" if row["sign_tot"] == -1 else "up"
        attr["diff_tot"] = row["diff_tot"]
    for idx, row in ann_sz.iterrows():
        attr = digraph.nodes[idx]
        attr["sign_i"] = "down" if row["sign_i"] == -1 else "up"
        attr["sign_s"] = "down" if row["sign_s"] == -1 else "up"
        attr["sign_z"] = "down" if row["sign_z"] == -1 else "up"
        attr["frac_i"] = row["frac_i"]
        attr["frac_s"] = row["frac_s"]
        attr["frac_z"] = row["frac_z"]
    for idx, row in ann_ptr_sum.iterrows():
        attr = digraph.nodes[idx]
        attr["frac_ptr"] = row["frac"]
    for _, row in ann_ptr.iterrows():
        u, v = row["parent"], row["index"]
        attr = digraph.edges[u, v]
        attr["sign"] = (
            "represses"
            if row["sign"] * ann_tot.loc[u, "sign_tot"] == -1
            else "activates"
        )
        attr["frac"] = row["frac"]
    return digraph


def core_explanation_graph(
    annotated: nx.DiGraph,
    leaves: list[str],
    min_frac_ptr: float = 0.05,
    depth_limit: int | None = None,
) -> nx.DiGraph:
    r"""
    Extract the core explanation graph from an annotated causal graph that
    explains the predicted change in a list of leaf nodes

    Parameters
    ----------
    annotated
        Annotated causal graph (from :func:`annotate_explanation`)
    leaves
        List of leaf nodes
    min_frac_ptr
        Minimal explained fraction for an edge to be considered
    depth_limit
        Depth limit for the breadth-first search

    Returns
    -------
    Core explanation graph
    """
    cutoff = annotated.graph["cutoff"]
    annotated = annotated.subgraph(
        u for u, d in annotated.nodes.items() if abs(d.get("diff_tot", 0)) > cutoff
    )
    annotated = filter_edges(annotated, edge_attr="frac", cutoff=min_frac_ptr)
    bfs_trees = [
        nx.bfs_tree(annotated, leaf, reverse=True, depth_limit=depth_limit)
        for leaf in leaves
        if leaf in annotated.nodes
    ]
    bfs_nodes = reduce(or_, [tree.nodes for tree in bfs_trees])
    return annotated.subgraph(bfs_nodes)


def prep_cytoscape(
    annotated: nx.DiGraph, scaffold: nx.DiGraph, perts: list[str], leaves: list[str]
) -> nx.DiGraph:
    r"""
    Prepare a graph for visualization in `Cytoscape <https://cytoscape.org>`_

    Parameters
    ----------
    annotated
        Annotated causal graph (from :func:`annotate_explanation`)
    scaffold
        Scaffold graph (from :meth:`~cascade.model.CASCADE.export_causal_graph`)
    perts
        List of perturbed nodes
    leaves
        List of leaf nodes to explain

    Returns
    -------
    Cytoscape-ready graph


    .. tip::

        Please visit TODO to obtain a template Cytoscape file containing
        corresponding styles.
    """
    annotated = annotated.copy()
    for u, d in annotated.nodes.items():
        d["frac_i"] = d.get("frac_i", 0)
        d["frac_s"] = d.get("frac_s", 0)
        d["frac_z"] = d.get("frac_z", 0)
        d["frac_ptr"] = d.get("frac_ptr", 0)
        if u in perts:
            d["role"] = "interv"
        elif u in leaves:
            d["role"] = "leaf"
        else:
            d["role"] = "med"
    for e, d in annotated.edges.items():
        scf = scaffold.edges[e]
        d["frac"] = np.clip(d.get("frac", 0), 0, 1)
        d["evidence_type"] = scf.get("evidence_type", "unknown")
    return annotated
