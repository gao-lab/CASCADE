from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from cascade.data import aggregate_obs, get_all_targets


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--interv-key", type=str, required=True)
    train.add_argument("--dim", type=int, default=10)
    train.add_argument("--lam", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--info", type=Path, default=None)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-data", type=Path, required=True)
    predict.add_argument("--input-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--info", type=Path, default=None)
    design = subparsers.add_parser("design")
    design.add_argument("--input-model", type=Path, required=True)
    design.add_argument("--input-target", type=Path, required=True)
    design.add_argument("--pool", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--design-size", type=int, required=True)
    design.add_argument("-k", type=int, default=30)
    design.add_argument("--n-neighbors", type=int, default=1)
    design.add_argument("--info", type=Path, default=None)
    return parser.parse_args()


def solve_wandb(Y, G, P, lam):
    Y = np.asarray(Y)
    G = np.asarray(G)
    P = np.asarray(P)
    b = Y.mean(axis=1, keepdims=True)
    W = (
        np.linalg.inv(G.T @ G + lam * np.eye(G.shape[1]))
        @ G.T
        @ (Y - b)
        @ P
        @ np.linalg.inv(P.T @ P + lam * np.eye(P.shape[1]))
    )
    return W, b


def train(args):
    train = ad.read_h5ad(args.input)

    start_time = time()
    train_agg = aggregate_obs(train, args.interv_key, X_agg="mean").to_df()
    ctrl = train_agg.loc[""]
    train_diff = train_agg.loc[train_agg.index != ""].sub(ctrl).T
    G = pd.DataFrame(
        PCA(n_components=args.dim, random_state=args.seed).fit_transform(train_diff),
        index=train_diff.index,
    )
    singles = train_diff.columns[train_diff.columns.str.count(",") == 0]
    P = G.loc[singles]
    Y = train_diff.loc[:, singles]
    W, b = solve_wandb(Y, G, P, args.lam)
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, index=G.index, G=G, W=W, b=b, ctrl=ctrl)
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


def predict(args):
    data = ad.read_h5ad(args.input_data)
    model = np.load(args.input_model, allow_pickle=True)
    index, G, W, b, ctrl = (
        model["index"],
        model["G"],
        model["W"],
        model["b"],
        model["ctrl"],
    )
    index = pd.Index(index)
    ctrl = pd.Series(ctrl, index=index)

    start_time = time()
    all_targets = sorted(get_all_targets(data, args.interv_key))
    P = G[index.get_indexer(all_targets)]
    pred_diff = pd.DataFrame(G @ W @ P.T + b, index=index, columns=all_targets).T
    ctfact = pd.DataFrame.from_dict(
        {
            k: ctrl + sum(pred_diff.loc[i] for i in k.split(","))
            for k in data.obs[args.interv_key].unique()
        },
        orient="index",
    )
    ctfact = ad.AnnData(ctfact)
    ctfact.obs[args.interv_key] = ctfact.obs_names
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctfact.write(args.output, compression="gzip")
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


def design(args):
    target = ad.read_h5ad(args.input_target)
    model = np.load(args.input_model, allow_pickle=True)
    index, G, W, b, ctrl = (
        model["index"],
        model["G"],
        model["W"],
        model["b"],
        model["ctrl"],
    )
    index = pd.Index(index)
    ctrl = pd.Series(ctrl, index=index)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()

    start_time = time()
    search_space = [
        ",".join(sorted(c))
        for s in range(args.design_size + 1)
        for c in combinations(pool, s)
    ]
    all_targets = sorted(pool)
    P = G[index.get_indexer(all_targets)]
    pred_diff = pd.DataFrame(G @ W @ P.T + b, index=index, columns=all_targets)
    pred_diff[""] = 0
    pred_diff = pred_diff.T
    ctfact = pd.DataFrame.from_dict(
        {k: ctrl + sum(pred_diff.loc[i] for i in k.split(",")) for k in search_space},
        orient="index",
    )

    neighbor = NearestNeighbors(n_neighbors=args.n_neighbors).fit(ctfact.to_numpy())
    nni = neighbor.kneighbors(target.X, return_distance=False)
    votes = ctfact.index[nni.ravel()].value_counts()
    outcast = [item for item in search_space if item not in votes.index]
    outcast = pd.Series(0, index=outcast, name="count")
    design = pd.concat([votes, outcast])
    design = design.to_frame().rename(columns={"count": "votes"})
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output)
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


if __name__ == "__main__":
    args = parse_args()
    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "predict":
        predict(args)
    else:  # args.subcommand == "design"
        design(args)
