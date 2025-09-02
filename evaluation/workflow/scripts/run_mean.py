from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import pandas as pd
import yaml

from cascade.data import aggregate_obs


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-train", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interv-key", type=str, required=True)
    parser.add_argument("--info", type=Path, default=None)
    return parser.parse_args()


def run(args):
    train = ad.read_h5ad(args.input_train)
    data = ad.read_h5ad(args.input_data)
    start_time = time()
    train_agg = aggregate_obs(train, args.interv_key, X_agg="mean").to_df()
    train_mean = train_agg.loc[train_agg.index != ""].mean(axis=0)
    ctfact = pd.DataFrame.from_dict(
        {k: train_mean for k in data.obs[args.interv_key].unique()}, orient="index"
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


if __name__ == "__main__":
    run(parse_args())
