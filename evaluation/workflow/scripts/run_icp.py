#!/usr/bin/env python

from argparse import ArgumentParser
from itertools import chain, combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import causalicp as icp  # type: ignore
import networkx as nx
import numpy as np
import yaml
from joblib import Parallel, delayed
from scipy.sparse import issparse
from tqdm.auto import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interv-key", type=str, required=True)
    parser.add_argument("--deg-limit", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--color", default=False, action="store_true")
    return parser.parse_args()


def powerset(s, limit):
    s = set(s)
    return [
        set(item)
        for item in chain.from_iterable(
            combinations(s, r) for r in range(min(limit, len(s) + 1))
        )
    ]


def main(args):
    adata = ad.read_h5ad(args.input)
    scaffold = nx.DiGraph(nx.read_gml(args.scaffold))
    X = adata.X.toarray() if issparse(adata.X) else adata.X
    regimes = set(adata.obs[args.interv_key])
    X = [X[np.where(adata.obs[args.interv_key] == r)[0]] for r in regimes]
    nx.relabel_nodes(
        scaffold, {v: i for i, v in enumerate(adata.var_names)}, copy=False
    )

    @delayed
    def job(k):
        return (
            icp.fit(
                X,
                k,
                alpha=args.alpha,
                sets=powerset(scaffold.predecessors(k), args.deg_limit),
                verbose=args.verbose,
                color=args.color,
            ).estimate
            or set()
        )

    start_time = time()
    result = Parallel(n_jobs=args.n_jobs)(job(k) for k in tqdm(range(adata.n_vars)))
    digraph = nx.DiGraph()
    digraph.add_nodes_from(range(adata.n_vars))
    digraph.add_edges_from((p, k) for k, parents in enumerate(result) for p in parents)
    nx.relabel_nodes(digraph, {i: v for i, v in enumerate(adata.var_names)}, copy=False)
    nx.set_edge_attributes(digraph, 1, "weight")
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
