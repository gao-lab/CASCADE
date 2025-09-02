#!/usr/bin/env python

from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import networkx as nx
import yaml

from cascade.ri import ges


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", type=str, default="GaussL0penObsScore")
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def main(args):
    adata = ad.read_h5ad(args.input)
    scaffold = nx.read_gml(args.scaffold)
    start_time = time()
    digraph = ges(
        adata,
        nx.Graph(scaffold),
        score=args.score,
        verbose=args.verbose,
    )
    elapsed_time = time() - start_time
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(digraph, args.output)
    args.info.parent.mkdir(parents=True, exist_ok=True)
    with args.info.open("w") as f:
        yaml.dump(
            {
                "cmd": " ".join(argv),
                "args": vars(args),
                "time": elapsed_time,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
            },
            f,
        )


if __name__ == "__main__":
    main(parse_args())
