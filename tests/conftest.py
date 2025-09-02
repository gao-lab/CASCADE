import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData
from pytest import fixture, mark

import cascade


def pytest_addoption(parser):
    parser.addoption("--use-ri", dest="use_ri", action="store_true", default=False)
    parser.addoption("--precision", dest="precision", type=str, default="32-true")


def pytest_configure(config):
    config.addinivalue_line("markers", "use_ri: mark test to run only with --use-ri")
    cascade.config.PRECISION = config.getoption("precision")
    cascade.config.NUM_WORKERS = 0  # Avoid random hangs
    cascade.config.MIN_DELTA = 0.1
    cascade.config.PATIENCE = 1
    if cascade.config.LOG_LEVEL != "DEBUG":
        cascade.config.LOG_LEVEL = "DEBUG"


def pytest_collection_modifyitems(config, items):
    if not config.option.use_ri:
        skip_use_ri = mark.skip(reason="only runs with --use-ri")
        for item in items:
            if "use_ri" in item.keywords:
                item.add_marker(skip_use_ri)


@fixture
def dag() -> nx.DiGraph:
    dag = nx.DiGraph(
        [
            ("A", "C"),
            ("A", "D"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("C", "G"),
            ("F", "G"),
            ("G", "D"),
            ("E", "H"),
            ("E", "I"),
        ]
    )
    nx.set_edge_attributes(dag, 1.0, "weight")
    nx.set_node_attributes(dag, 1.0, "snr")
    nx.set_node_attributes(dag, "tanh", "act")
    return dag


@fixture
def dg() -> nx.DiGraph:
    dg = nx.DiGraph(
        [
            ("A", "C"),
            ("A", "D"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("C", "G"),
            ("F", "A"),
            ("G", "D"),
            ("E", "H"),
            ("E", "I"),
        ]
    )
    nx.set_edge_attributes(dg, 1.0, "weight")
    nx.set_node_attributes(dg, 10.0, "snr")
    nx.set_node_attributes(dg, "tanh", "act")
    return dg


@fixture
def adata() -> AnnData:
    X = np.stack(
        [np.random.negative_binomial(5 * (i + 1), 0.5, size=1000) for i in range(3)]
        * 3,
        axis=1,
    )
    return AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "interv": [""] * 200
                + ["B"] * 200
                + ["D"] * 200
                + ["C,D"] * 100
                + ["D,C"] * 100
                + ["E"] * 200,
                "ctfact": ["D"] * 200
                + ["F"] * 200
                + ["G"] * 200
                + ["F,G"] * 100
                + ["G,F"] * 100
                + ["H"] * 200,
                "weight": np.random.rand(1000),
                "size": X.sum(axis=1) + (np.random.rand(1000) * 1000).round(),
                "complete": ",".join("ABCDEFGHI"),
            },
            index=map(str, range(1000)),
        ),
        obsm={"covariate": np.random.randn(1000, 1)},
        varm={"repr": np.random.randn(9, 2)},
        var=pd.DataFrame(
            {"group": ["G1", "G2", "G3"] * 3},
            index=list("ABCDEFGHI"),
        ),
    )
