from argparse import ArgumentParser, Namespace
from pathlib import Path

import networkx as nx

from cascade.data import Targets
from cascade.sim import simulate_regimes
from cascade.utils import get_random_state


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-obs", type=int, required=True)
    parser.add_argument("--int-frac", type=float, required=True)
    parser.add_argument("--act", type=str, required=True)
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    causal = nx.read_gml(args.input)
    nx.set_node_attributes(causal, args.snr, "snr")
    nx.set_node_attributes(causal, args.act, "act")
    rnd = get_random_state(args.seed)
    targets_list = rnd.choice(
        causal.nodes, round(args.int_frac * causal.number_of_nodes()), replace=False
    ).tolist()
    targets_list.append("")
    adata = simulate_regimes(
        causal,
        {
            Targets(targets): round(args.n_obs / len(targets_list))
            for targets in targets_list
        },
        {node: 0.0 for node in causal.nodes},
        random_state=rnd,
    )
    del adata.obs["knockdown"]
    del adata.obs["knockup"]
    adata = adata[rnd.permutation(adata.n_obs)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
