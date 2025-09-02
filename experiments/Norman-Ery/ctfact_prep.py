from argparse import ArgumentParser, Namespace
from functools import reduce
from operator import or_
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from cascade.data import Targets


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-designs", type=int, default=5)
    parser.add_argument("--max-ctrl", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: Namespace) -> None:
    ctrl = ad.read_h5ad(args.ctrl)
    target = ad.read_h5ad(args.target)
    design = pd.read_csv(args.design, index_col=0, keep_default_na=False)
    prep = target.copy()

    rnd = np.random.RandomState(args.seed)
    comb = design.index[: args.top_designs].to_list()
    ind = list(reduce(or_, map(Targets, comb)))
    intervs = sorted({*comb, *ind, ""})
    n_ctrl = min(ctrl.n_obs, args.max_ctrl)
    prep_n_obs = n_ctrl * len(intervs)
    ctrl_idx = rnd.choice(ctrl.n_obs, prep_n_obs, replace=True)
    target_idx = rnd.choice(target.n_obs, prep_n_obs, replace=True)
    prep = ctrl[ctrl_idx].copy()
    prep.obs_names_make_unique()
    if "covariate" in target.obsm:
        prep.obsm["covariate"] = target.obsm["covariate"][target_idx]
    prep.obs["knockup"] = [item for item in intervs for _ in range(n_ctrl)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prep.write_h5ad(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
