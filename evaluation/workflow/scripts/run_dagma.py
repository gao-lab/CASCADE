#!/usr/bin/env python

import argparse
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import networkx as nx
import numpy as np
import torch
import yaml
from dagma.nonlinear import DagmaMLP, DagmaNonlinear  # type: ignore
from scipy.sparse import issparse


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hid-dim", type=int, default=10)
    parser.add_argument("--lambda1", type=float, default=0.02)
    parser.add_argument("--lambda2", type=float, default=0.005)
    parser.add_argument("-T", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--info", type=Path, required=True)
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    return parser.parse_args()


def main(args):
    adata = ad.read_h5ad(args.input)
    X = torch.as_tensor(adata.X.toarray() if issparse(adata.X) else adata.X)
    start_time = time()
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    eq_model = DagmaMLP(dims=[adata.n_vars, 10, 1], bias=True, dtype=torch.double)
    if args.gpu:
        X = X.cuda()
        eq_model = eq_model.cuda()
        eq_model.I = eq_model.I.cuda()  # noqa: E741
    model = DagmaNonlinear(eq_model, verbose=args.verbose, dtype=torch.double)
    adj = model.fit(X, lambda1=args.lambda1, lambda2=args.lambda2, T=args.T)
    digraph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    nx.relabel_nodes(digraph, {i: v for i, v in enumerate(adata.var_names)}, copy=False)
    elapsed_time = time() - start_time
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(digraph, args.output)
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
