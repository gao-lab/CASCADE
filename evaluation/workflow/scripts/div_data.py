from argparse import ArgumentParser, Namespace
from functools import reduce
from operator import or_
from pathlib import Path

import anndata as ad
import numpy as np

from cascade.data import Targets


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-test", type=Path, required=True)
    parser.add_argument("--kg", type=float, default=0.9)
    parser.add_argument("--kc", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input)
    interv_cols = set(adata.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = adata.obs[interv_cols.pop()].map(Targets)
    interv_uniq = interv_col.unique()
    assert Targets() in interv_uniq
    all_intervs = reduce(or_, interv_uniq)

    rnd = np.random.RandomState(args.seed)
    seen_intervs = Targets(
        rnd.choice(
            sorted(all_intervs),
            size=round(args.kg * len(all_intervs)),
            replace=False,
        )
    )
    unseen_intervs = all_intervs - seen_intervs

    interv_uniq_single = interv_uniq[
        interv_uniq.map(len, na_action="ignore").astype(int) == 1
    ]
    interv_uniq_single_train = {p for p in interv_uniq_single if (p & seen_intervs)}
    interv_uniq_single_test = {p for p in interv_uniq_single if (p & unseen_intervs)}
    interv_uniq_combo = interv_uniq[
        interv_uniq.map(len, na_action="ignore").astype(int) > 1
    ]
    interv_uniq_combo_all_seen = {
        p for p in interv_uniq_combo if not (p & unseen_intervs)
    }
    interv_uniq_combo_all_seen_train = set(
        rnd.choice(
            sorted(interv_uniq_combo_all_seen),
            size=round(args.kc * len(interv_uniq_combo_all_seen)),
            replace=False,
        )
    )
    interv_uniq_combo_all_seen_test = (
        interv_uniq_combo_all_seen - interv_uniq_combo_all_seen_train
    )
    interv_uniq_combo_any_unseen = {
        p for p in interv_uniq_combo if (p & unseen_intervs)
    }

    interv_uniq_train = (
        {Targets()} | interv_uniq_single_train | interv_uniq_combo_all_seen_train
    )
    interv_uniq_test = (
        interv_uniq_single_test
        | interv_uniq_combo_all_seen_test
        | interv_uniq_combo_any_unseen
    )
    adata_train = adata[interv_col.isin(interv_uniq_train)].copy()
    adata_train.obs["category"] = interv_col.loc[adata_train.obs_names].map(
        {p: f"{len(p)} seen" for p in interv_uniq_train}
    )
    adata_test = adata[interv_col.isin(interv_uniq_test)].copy()
    adata_test.obs["category"] = interv_col.loc[adata_test.obs_names].map(
        {p: f"{len(p - seen_intervs)}/{len(p)} unseen" for p in interv_uniq_test}
    )

    args.output_train.parent.mkdir(parents=True, exist_ok=True)
    adata_train.write(args.output_train, compression="gzip")
    args.output_test.parent.mkdir(parents=True, exist_ok=True)
    adata_test.write(args.output_test, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
