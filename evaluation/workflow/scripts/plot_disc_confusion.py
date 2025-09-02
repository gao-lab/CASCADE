from argparse import ArgumentParser, Namespace
from pathlib import Path
from statistics import mean
from typing import Iterable

import networkx as nx
from matplotlib import pyplot as plt

from cascade.graph import map_edges
from cascade.metrics import optimal_cutoff
from cascade.plot import plot_adj_confusion, set_figure_params


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=None)
    return parser.parse_args()


def main(args: Namespace) -> None:
    true = nx.read_gml(args.true)
    pred = nx.read_gml(args.pred)
    scaffold = nx.read_gml(args.scaffold)

    pred = map_edges(pred, fn=lambda x: mean(x) if isinstance(x, Iterable) else x)
    cutoff = (
        optimal_cutoff(true, pred, scaffold=scaffold)
        if args.cutoff is None
        else args.cutoff
    )

    true = nx.to_pandas_adjacency(true, weight=None, dtype=bool)
    pred = nx.to_pandas_adjacency(pred) > cutoff
    mask = nx.to_pandas_adjacency(scaffold, weight=None) == 0

    set_figure_params()
    fig = plt.figure(figsize=(10, 10))
    plot_adj_confusion(true, pred, mask=mask, cbar=False, square=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main(parse_args())
