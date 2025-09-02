from itertools import product
from random import sample

import networkx as nx
from pytest import raises

from cascade.graph import (
    acyclify,
    demultiplex,
    filter_edges,
    map_edges,
    marginalize,
    multiplex,
    multiplex_num,
    node_stats,
)
from cascade.sim import generate_dag


def test_multiplex_demultiplex_filter():
    g0 = nx.DiGraph()
    assert multiplex_num(g0) == 1
    g1 = generate_dag(15, 0.3)
    g2 = generate_dag(15, 0.3)
    nx.set_edge_attributes(g1, 1.0, name="weight")
    nx.set_edge_attributes(g2, 1.0, name="weight")
    mg = multiplex(g1, g2)
    assert mg.edges == g1.edges | g2.edges
    g1_, g2_ = demultiplex(mg)
    assert g1_.edges == g2_.edges == mg.edges
    g1__ = filter_edges(g1_, cutoff=0.5)
    g2__ = filter_edges(g2_, cutoff=0.5)
    assert g1__.edges == g1.edges
    assert g2__.edges == g2.edges
    g1__ = filter_edges(g1_, n_top=g1.number_of_edges())
    g2__ = filter_edges(g2_, n_top=g2.number_of_edges())
    assert g1__.edges == g1.edges
    assert g2__.edges == g2.edges
    assert g1 is demultiplex(g1) and g2 is demultiplex(g2)
    with raises(TypeError):
        multiplex(g1, nx.Graph())
    mg = mg.copy()  # Change hash
    mg.add_edge(0, 1, weight=[1.0, 2.0, 3.0])
    with raises(ValueError):
        demultiplex(mg)
    with raises(ValueError):
        filter_edges(g1_)
    with raises(ValueError):
        filter_edges(g1_, cutoff=0.5, n_top=5)


def test_map_edges():
    g = generate_dag(15, 0.3)
    nx.set_edge_attributes(g, [1, 2, 3], name="weight")
    g = map_edges(g, edge_attr="weight", fn=sum)
    assert set(nx.get_edge_attributes(g, "weight").values()) == {6}


def test_acyclify():
    dag = generate_dag(15, 0.3)
    nx.set_edge_attributes(dag, 1.0, name="weight")
    for u, v in product(dag.nodes, repeat=2):
        if u == v or (u, v) in dag.edges:
            continue
        dag.add_edge(u, v, weight=0.1)
        if not nx.is_directed_acyclic_graph(dag):
            break
    assert nx.is_directed_acyclic_graph(acyclify(dag))


def test_marginalize():
    dag = generate_dag(15, 0.5)
    vars = sample(sorted(dag.nodes), 7) + ["X"]
    marginalized = marginalize(dag, vars, max_steps=0)
    assert set(marginalized.edges) == set(dag.subgraph(vars).edges)
    marginalized = marginalize(dag, vars, max_steps=7)
    assert set(marginalized.edges) - set(dag.subgraph(vars).edges)
    with raises(TypeError):
        marginalized = marginalize(nx.Graph(dag), vars, max_steps=0)
    with raises(ValueError):
        marginalized = marginalize(dag, vars, max_steps=-1)


def test_node_stats():
    dag = generate_dag(15, 0.5)
    node_stats(dag)
