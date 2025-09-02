from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import networkx as nx


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input, backed="r")
    complete = nx.complete_graph(adata.var_names, create_using=nx.DiGraph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(complete, args.output)


if __name__ == "__main__":
    main(parse_args())
