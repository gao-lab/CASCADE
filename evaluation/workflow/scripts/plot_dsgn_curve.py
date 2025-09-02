from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from cascade.metrics import dsgn_hrc_exact, dsgn_hrc_partial
from cascade.plot import set_figure_params

set_figure_params()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--designs", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    designs = {
        design.parent.name.replace("+", ","): pd.read_csv(
            design, index_col=0, keep_default_na=False
        )
        for design in args.designs
    }
    designs = {k: v.get("score", v.get("votes")) for k, v in designs.items()}
    if not designs:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.touch()
        return

    max_size = max(len(design.split(",")) for design in designs)
    qtl, hr = dsgn_hrc_exact(designs)
    df = pd.DataFrame({"qtl": qtl, "hr": hr, "Match": "Exact"})
    if max_size > 1:
        qtl, hr = dsgn_hrc_partial(designs)
        df_partial = pd.DataFrame({"qtl": qtl, "hr": hr, "Match": "Partial"})
        df = pd.concat([df, df_partial], ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="qtl", y="hr", hue="Match", ax=ax)
    sns.scatterplot(data=df, x="qtl", y="hr", hue="Match", legend=False, ax=ax)
    ax.set_xlabel("Design quantile")
    ax.set_ylabel("Hit rate")
    fig.savefig(args.output)


if __name__ == "__main__":
    main(parse_args())
