from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import networkx as nx
import pandas as pd

from cascade.data import Targets
from cascade.utils import get_random_state


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-graph", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: Namespace) -> None:
    graph = nx.read_gml(args.input_graph)
    adata = ad.read_h5ad(args.input_data)
    assert (adata.var_names == pd.Index(graph.nodes)).all()

    interv_col = adata.obs["knockout"].map(Targets)
    rnd = get_random_state(args.seed)
    keep_vars = rnd.choice(
        adata.var_names,
        size=round(args.frac * adata.n_vars),
        replace=False,
    )

    var_left = set(keep_vars) | {""}
    var_mask = adata.var_names.isin(var_left)
    obs_mask = interv_col.map(lambda x: not (x - var_left)).astype(bool)
    sub_adata = adata[obs_mask, var_mask].copy()
    sub_graph = graph.subgraph(sub_adata.var_names)

    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(sub_graph, args.output_graph)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    sub_adata.write(args.output_data, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
