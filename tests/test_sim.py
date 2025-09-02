import networkx as nx
from pytest import mark, raises

from cascade.data import Targets
from cascade.sim import (
    DAGType,
    generate_dag,
    simulate_counterfactual,
    simulate_random_regimes,
    simulate_regimes,
)

from .utils import array_close


@mark.parametrize("m", [0.1, 10])
@mark.parametrize("type", ["unif", "sf"])
def test_generate_dag(m, type):
    dag1 = generate_dag(100, m, type=DAGType[type], random_state=0)
    dag2 = generate_dag(100, m, type=DAGType[type], random_state=0)
    dag3 = generate_dag(100, m, type=DAGType[type], random_state=42)
    assert nx.is_directed_acyclic_graph(dag1)
    assert nx.is_directed_acyclic_graph(dag2)
    assert nx.is_directed_acyclic_graph(dag3)
    assert set(dag1.edges) == set(dag2.edges)
    assert set(dag1.edges) != set(dag3.edges)
    with raises(KeyError):
        generate_dag(100, m, type=DAGType[type.upper()])


def test_simulate_regimes(dag, dg):
    design = {
        Targets(""): 10,
        Targets("A"): 10,
        Targets("B"): 10,
        Targets("C"): 10,
        Targets("D"): 10,
        Targets("E"): 10,
        Targets("F"): 10,
        Targets("G"): 10,
        Targets("H"): 10,
        Targets("I"): 10,
        Targets("A,E"): 10,
        Targets("B,F,H"): 10,
        Targets("A,C,D,G,I"): 10,
    }
    interv = {
        "A": 0.0,
        "B": 2.0,
        "C": 0.2,
        "D": 0.5,
        "E": 5.0,
        "F": 3.0,
        "G": 0.1,
        "H": 0.7,
        "I": 0.0,
    }
    adata1 = simulate_regimes(dag, design, interv, random_state=0)
    adata2 = simulate_regimes(dag, design, interv, random_state=0)
    adata3 = simulate_regimes(dag, design, interv, random_state=42)
    assert adata1.shape == adata2.shape == adata3.shape == (130, 9)
    assert array_close(adata1.X, adata2.X)
    assert not array_close(adata1.X, adata3.X)
    with raises(ValueError):
        simulate_regimes(dg, design, interv)


def test_simulate_random_regimes(dag, dg):
    interv = {
        "A": 0.0,
        "B": 2.0,
        "C": 0.0,
        "D": 0.5,
        "E": 0.0,
        "F": 0.0,
        "G": 1.2,
        "H": 0.0,
        "I": 0.0,
    }
    adata1 = simulate_random_regimes(dag, 300, 0.1, interv, random_state=0)
    adata2 = simulate_random_regimes(dag, 300, 0.1, interv, random_state=0)
    adata3 = simulate_random_regimes(dag, 300, 0.1, interv, random_state=42)
    nx.set_node_attributes(dag, "ident", "act")
    adata4 = simulate_random_regimes(dag, 300, 0.1, interv, random_state=42)
    assert adata1.shape == adata2.shape == adata3.shape == adata4.shape == (300, 9)
    assert array_close(adata1.X, adata2.X)
    assert not array_close(adata1.X, adata3.X)
    assert not array_close(adata3.X, adata4.X)
    with raises(ValueError):
        simulate_random_regimes(dg, 300, 0.1, interv)


def test_simulate_counterfactual(dag):
    design = {
        Targets("A"): 10,
        Targets("B"): 10,
        Targets("C"): 10,
        Targets("D"): 10,
        Targets("E"): 10,
        Targets("F"): 10,
        Targets("G"): 10,
        Targets("H"): 10,
        Targets("I"): 10,
        Targets("A,E"): 10,
        Targets("B,F,H"): 10,
        Targets("A,C,D,G,I"): 10,
    }
    interv = {
        "A": 0.0,
        "B": 2.0,
        "C": 0.2,
        "D": 0.5,
        "E": 5.0,
        "F": 3.0,
        "G": 0.1,
        "H": 0.7,
        "I": 0.0,
    }
    adata1 = simulate_regimes(dag, design, interv)
    scale = adata1.layers["scale"].copy()
    adata2 = simulate_counterfactual(adata1, scale)
    assert array_close(adata1.X, adata2.X)
    scale[:, 0] = 0.5
    adata3 = simulate_counterfactual(adata1, scale)
    assert not array_close(adata1.X, adata3.X)
