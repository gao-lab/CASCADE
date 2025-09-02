from argparse import ArgumentParser, Namespace
from itertools import product
from pathlib import Path
from random import Random

import networkx as nx


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tpr", type=float, required=True)
    parser.add_argument("--fpr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    dag = nx.read_gml(args.input)
    true_edges = set(dag.edges)
    self_loops = {(node, node) for node in dag.nodes}
    false_edges = set(product(dag.nodes, repeat=2)) - self_loops - true_edges
    random = Random(args.seed)
    n_true = round(len(true_edges) * args.tpr)
    n_false = round(len(false_edges) * args.fpr)
    true_edges = random.sample(sorted(true_edges), k=n_true)
    false_edges = random.sample(sorted(false_edges), k=n_false)
    corrupted = nx.DiGraph()
    corrupted.add_nodes_from(dag.nodes)
    corrupted.add_edges_from(true_edges + false_edges)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(corrupted, args.output)


if __name__ == "__main__":
    main(parse_args())
