#!/usr/bin/env python

import argparse
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import networkx as nx
import numpy as np
import tensorflow as tf  # type: ignore
import yaml
from BNGPU import NOBEARS  # type: ignore
from scipy.sparse import issparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--rho-init", type=float, default=10.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--info", type=Path, required=True)
    return parser.parse_args()


def main(args):
    adata = ad.read_h5ad(args.input)
    X = adata.X.toarray() if issparse(adata.X) else adata.X

    start_time = time()
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)
    W_init = NOBEARS.W_reg_init(X).astype("float32")
    X = np.vstack([X] * 2)

    tf.reset_default_graph()
    clf = NOBEARS.NoBearsTF(poly_degree=args.poly_degree, rho_init=args.rho_init)
    clf.construct_graph(X, W_init)

    sess = tf.Session()
    sess.run(clf.graph_nodes["init_vars"])
    clf.model_init_train(sess)
    clf.model_train(sess)
    W_est = sess.run(clf.graph_nodes["weight_ema"])

    digraph = nx.from_numpy_array(W_est, create_using=nx.DiGraph)
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
