from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.decomposition import TruncatedSVD

from cascade.data import aggregate_obs


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input)
    interv_cols = set(adata.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()

    adata_agg = aggregate_obs(adata, interv_col, X_agg="mean").to_df()
    adata_diff = adata_agg.loc[adata_agg.index != ""] - adata_agg.loc[""]
    adata_diff /= adata_diff.std()

    n_comps = min(adata_diff.shape) - 1
    print(f"Running SVD up to {n_comps} components...")
    truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
    svd = truncated_svd.fit_transform(adata_diff.T)
    argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
    exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
    svd = svd[:, argsort]
    knee = KneeLocator(
        np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
    )
    exp_var = exp_var_ratio[: knee.knee + 1].sum()
    print(f"Identified knee = {knee.knee}")
    print(f"Total explained variance = {exp_var}")
    emb = pd.DataFrame(svd[:, : knee.knee + 1], index=adata.var_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    emb.to_csv(args.output)


if __name__ == "__main__":
    main(parse_args())
