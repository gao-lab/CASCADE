from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", type=int, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    data = ad.read_h5ad(args.input)
    interv_cols = set(data.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()
    requests = (
        data.obs.loc[:, [interv_col, "edist"]]
        .drop_duplicates(subset=interv_col)
        .sort_values("edist", ascending=False)
        .head(args.k)[interv_col]
        .str.replace(",", "+")
    )  # Must not contain commas due to snakemake executor limitation
    args.output.mkdir(parents=True, exist_ok=True)
    for request in requests:
        (args.output / f"{request}").touch()


if __name__ == "__main__":
    main(parse_args())
