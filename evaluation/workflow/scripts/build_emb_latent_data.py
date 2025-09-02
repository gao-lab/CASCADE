from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import pandas as pd


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--input-emb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input_data)
    emb = pd.read_csv(args.input_emb, index_col=0)
    emb = emb.reindex(adata.var_names).dropna()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    emb.to_csv(args.output)


if __name__ == "__main__":
    main(parse_args())
