from argparse import ArgumentParser, Namespace
from pathlib import Path
from statistics import mean
from typing import Iterable

import networkx as nx
from matplotlib import pyplot as plt

from cascade.graph import map_edges
from cascade.plot import plot_adj, set_figure_params


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cluster", default=False, action="store_true")
    return parser.parse_args()


def main(args: Namespace) -> None:
    input = nx.read_gml(args.input)
    try:
        input = map_edges(input, fn=lambda x: mean(x) if isinstance(x, Iterable) else x)
        adj = nx.to_pandas_adjacency(input)
    except KeyError:
        adj = nx.to_pandas_adjacency(input, weight=None)

    if args.scaffold is None:
        mask = None
    else:
        scaffold = nx.read_gml(args.scaffold)
        mask = nx.to_pandas_adjacency(scaffold, weight=None) == 0

    set_figure_params()
    fig = plt.figure(figsize=(10, 10))
    plot_adj(
        adj,
        mask=mask,
        center=0,
        vmin=0,
        vmax=1,
        cluster=args.cluster,
        square=not args.cluster,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)


if __name__ == "__main__":
    main(parse_args())
