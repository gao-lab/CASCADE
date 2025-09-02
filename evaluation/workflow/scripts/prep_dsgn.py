from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata = adata[adata.obs[interv_col] == args.output.stem.replace("+", ",")].copy()
    adata.write(args.output, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
