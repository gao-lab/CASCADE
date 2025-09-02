from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
import yaml

from cascade.metrics import dsgn_auhrc_exact, dsgn_auhrc_partial


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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(
            {
                "auhrc_exact": dsgn_auhrc_exact(designs).item() if designs else 0.0,
                "auhrc_partial": dsgn_auhrc_partial(designs).item() if designs else 0.0,
            },
            f,
        )


if __name__ == "__main__":
    main(parse_args())
