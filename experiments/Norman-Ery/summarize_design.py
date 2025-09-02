from argparse import ArgumentParser, Namespace

import pandas as pd
import parse
from snakemake.script import Snakemake


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+")
    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    args.params = Namespace(pattern=args.pattern)
    args.output = [args.output]
    del args.pattern
    return args


def main(snakemake: Namespace | Snakemake) -> None:
    df = []
    for item in set(snakemake.input):
        entry = parse.parse(snakemake.params.pattern, item)
        if entry:
            conf = entry.named
        else:
            continue
        df.append(
            pd.read_csv(item, index_col=0).reset_index().head(n=20).assign(**conf)
        )
    df = pd.concat(df)
    df["gene"] = df.pop("index")
    df["score"] = df.pop("score")
    df.to_csv(snakemake.output[0], index=False)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = parse_args()
    main(snakemake)
