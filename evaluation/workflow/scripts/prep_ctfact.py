from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-ctrl", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: Namespace) -> None:
    ctrl = ad.read_h5ad(args.input_ctrl)
    data = ad.read_h5ad(args.input_data)
    interv_cols = set(data.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()
    data = data[data.obs[interv_col] != ""].copy()

    rnd = np.random.RandomState(args.seed)
    idx = rnd.choice(ctrl.n_obs, data.n_obs, replace=True)
    data.X = ctrl.X[idx].copy()
    if "counts" in data.layers:
        data.layers["counts"] = ctrl.layers["counts"][idx].copy()
    if "ncounts" in data.obs:
        data.obs["ncounts"] = ctrl.obs["ncounts"].iloc[idx].to_list()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
