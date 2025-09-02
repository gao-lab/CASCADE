from argparse import ArgumentParser
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

from cascade.data import aggregate_obs


def parse_args():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-train", type=Path, required=True)
    predict.add_argument("--input-data", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--info", type=Path, default=None)
    design = subparsers.add_parser("design")
    design.add_argument("--input-train", type=Path, required=True)
    design.add_argument("--input-target", type=Path, required=True)
    design.add_argument("--pool", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--interv-key", type=str, required=True)
    design.add_argument("--design-size", type=int, required=True)
    design.add_argument("-k", type=int, default=30)
    design.add_argument("--n-neighbors", type=int, default=1)
    design.add_argument("--info", type=Path, default=None)
    return parser.parse_args()


def predict(args):
    train = ad.read_h5ad(args.input_train)
    data = ad.read_h5ad(args.input_data)
    start_time = time()
    train_agg = aggregate_obs(train, args.interv_key, X_agg="mean").to_df()
    train_diff = train_agg.loc[train_agg.index != ""].sub(train_agg.loc[""])
    ctfact = pd.DataFrame.from_dict(
        {
            k: train_agg.loc[""]
            + sum(train_diff.loc[i] for i in k.split(",") if i in train_diff.index)
            for k in data.obs[args.interv_key].unique()
        },
        orient="index",
    )  # Sum the diff for constituent perturbations that exist the training set
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
    train = ad.read_h5ad(args.input_train)
    target = ad.read_h5ad(args.input_target)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()

    start_time = time()
    train_agg = aggregate_obs(train, args.interv_key, X_agg="mean").to_df()
    train_diff = train_agg.loc[train_agg.index != ""].sub(train_agg.loc[""])

    search_space = [
        ",".join(sorted(c))
        for s in range(args.design_size + 1)
        for c in combinations(pool, s)
    ]
    ctfact = pd.DataFrame.from_dict(
        {
            k: train_agg.loc[""]
            + sum(train_diff.loc[i] for i in k.split(",") if i in train_diff.index)
            for k in search_space
        },
        orient="index",
    )  # Sum the diff for constituent perturbations that exist the training set

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
    if args.subcommand == "predict":
        predict(args)
    else:  # args.subcommand == "design"
        design(args)
