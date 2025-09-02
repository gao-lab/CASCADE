from argparse import ArgumentParser, Namespace
from pathlib import Path
from statistics import mean
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from cascade.graph import map_edges
from cascade.metrics import cmp_true_pred
from cascade.plot import plot_colored_curves, set_figure_params

set_figure_params()


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--true", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--roc", type=Path, required=True)
    parser.add_argument("--prc", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    true = nx.read_gml(args.true)
    pred = nx.read_gml(args.pred)
    scaffold = nx.read_gml(args.scaffold)

    pred = map_edges(pred, fn=lambda x: mean(x) if isinstance(x, Iterable) else x)
    df = cmp_true_pred(true, pred, edge_attr="weight", scaffold=scaffold)

    fpr, tpr, thresholds = roc_curve(df["true"], df["pred"])
    roc = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds})
    fig = plt.figure(figsize=(5.5, 5))
    ax = plot_colored_curves(
        x="FPR",
        y="TPR",
        hue="Threshold",
        data=roc,
        vmin=0.0,
        vmax=1.0,
    )
    ax.plot([0, 1], [0, 1], color="darkred", linestyle="--")
    args.roc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.roc)

    precision, recall, thresholds = precision_recall_curve(df["true"], df["pred"])
    prc = pd.DataFrame(
        {
            "Precision": precision,
            "Recall": recall,
            "Threshold": np.concatenate([np.array([0]), thresholds]),
        }
    )
    fig = plt.figure(figsize=(5.5, 5))
    ax = plot_colored_curves(
        x="Recall",
        y="Precision",
        hue="Threshold",
        data=prc,
        vmin=0.0,
        vmax=1.0,
    )
    true_frac = df["true"].sum() / df.shape[0]
    ax.plot([0, 1], [true_frac, true_frac], color="darkred", linestyle="--")
    args.prc.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.prc)


if __name__ == "__main__":
    main(parse_args())
