from argparse import ArgumentParser, Namespace
from pathlib import Path

import networkx as nx

from cascade.sim import DAGType, generate_dag
from cascade.utils import get_random_state


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--n-vars", type=int, required=True)
    parser.add_argument("--in-degree", type=float, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    if args.in_degree >= 1:
        args.in_degree = int(args.in_degree)
    elif args.in_degree <= 0:
        raise ValueError(f"Invalid in_degree: {args.in_degree}")
    rnd = get_random_state(args.seed)

    dag = generate_dag(
        args.n_vars,
        args.in_degree,
        type=DAGType[args.type],
        random_state=rnd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(dag, args.output)


if __name__ == "__main__":
    main(parse_args())
