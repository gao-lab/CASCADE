from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import issparse

from cascade.metrics import ctfact_delta_pcc, ctfact_dir_acc, ctfact_mse


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output-each", type=Path, required=True)
    parser.add_argument("--output-category", type=Path, required=True)
    parser.add_argument("--edist-cutoff", type=float, default=2.0)
    parser.add_argument("--top-de", type=int, nargs="*", default=[20])
    parser.add_argument("--exclude-self", default=False, action="store_true")
    parser.add_argument("--log-normalize", default=False, action="store_true")
    return parser.parse_args()


def main(args: Namespace) -> None:
    ctrl = ad.read_h5ad(args.ctrl)
    true = ad.read_h5ad(args.true)
    pred = ad.read_h5ad(args.pred)
    interv_cols = set(true.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()
    true = true[true.obs[interv_col] != ""].copy()
    pred = pred[pred.obs[interv_col] != ""].copy()
    extra_annot = (
        true.obs.loc[:, [interv_col, "edist", "category"]]
        .drop_duplicates(subset=interv_col)
        .set_index(interv_col)
    )

    if args.log_normalize:
        if issparse(pred.X):
            pred.X = pred.X.multiply(1e4 / np.asarray(pred.obs[["ncounts"]])).log1p()
        else:
            pred.X = np.log1p(pred.X * (1e4 / np.asarray(pred.obs[["ncounts"]])))

    combined_list = []
    top_de_suffix = {k: "" if k is None else f"_top{k}" for k in [*args.top_de, None]}
    for top_de, suffix in top_de_suffix.items():
        metric_args = (ctrl, true, pred, interv_col)
        metric_kwargs = {"top_de": top_de, "exclude_self": args.exclude_self}
        mse = ctfact_mse(*metric_args, **metric_kwargs)
        pcc = ctfact_delta_pcc(*metric_args, **metric_kwargs)
        acc = ctfact_dir_acc(*metric_args, **metric_kwargs)
        combined = mse.join(pcc).join(acc)
        combined.columns = combined.columns + suffix
        combined_list.append(combined)
    combined = pd.concat(combined_list, axis=1)
    combined = combined.join(extra_annot)

    args.output_each.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_each)

    args.output_category.parent.mkdir(parents=True, exist_ok=True)
    combined_category = (
        combined.loc[combined["edist"] > args.edist_cutoff]
        .groupby("category", observed=True)
        .mean()
    )
    for suffix in top_de_suffix.values():
        combined_category[f"normalized_mse{suffix}"] = (
            combined_category[f"pred_mse{suffix}"]
            / combined_category[f"true_mse{suffix}"]
        )
    combined_category = combined_category.reset_index().melt(
        id_vars="category", var_name="metric"
    )
    combined_category.index = [
        f"{metric} ({category})"
        for metric, category in zip(
            combined_category["metric"], combined_category["category"]
        )
    ]
    with args.output_category.open("w") as f:
        yaml.dump(combined_category["value"].to_dict(), f)


if __name__ == "__main__":
    main(parse_args())
