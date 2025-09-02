from argparse import ArgumentParser, Namespace
from os.path import relpath
from pathlib import Path

import anndata as ad
import numpy as np

from cascade.data import neighbor_impute


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--use-rep", type=str, default="X_pca")
    return parser.parse_args()


def main(args: Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.k == 0:
        print("Symlinking input to output directly...")
        args.output.symlink_to(relpath(args.input, args.output.parent))
        return

    print("Reading input...")
    adata = ad.read_h5ad(args.input)

    interv_cols = set(adata.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()

    print("Running neighbor imputation...")
    adata.X = np.expm1(adata.X)
    adata = neighbor_impute(
        adata,
        k=args.k,
        use_rep=args.use_rep,
        use_batch=interv_col,
        X_agg="mean",
        obs_agg={"ncounts": "sum"},
        obsm_agg={"covariate": "mean"},
        layers_agg={"counts": "sum"},
    )
    adata.X = np.log1p(adata.X)

    print("Updating ncounts covariate...")
    log_ncounts = np.log1p(adata.obs["ncounts"]).to_numpy()
    log_ncounts = (log_ncounts - log_ncounts.mean()) / log_ncounts.std()
    adata.obsm["covariate"][:, 0] = log_ncounts

    print("Writing output...")
    adata.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
