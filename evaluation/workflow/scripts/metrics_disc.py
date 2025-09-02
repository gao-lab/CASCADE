from argparse import ArgumentParser, Namespace
from pathlib import Path
from statistics import mean

import anndata as ad
import networkx as nx
import yaml

from cascade.graph import filter_edges, map_edges
from cascade.metrics import (
    annot_resp,
    disc_acc,
    disc_ap,
    disc_auroc,
    disc_f1,
    disc_prec,
    disc_recall,
    disc_resp_acc,
    disc_resp_dist,
    disc_resp_dist_diff,
    disc_shd,
    disc_sid,
)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    true = subparsers.add_parser("true")
    true.add_argument("--true", type=Path, required=True)
    true.add_argument("--pred", type=Path, required=True)
    true.add_argument("--scaffold", type=Path, required=True)
    true.add_argument("--output", type=Path, required=True)
    true.add_argument("--cutoff", type=float, default=None)
    resp = subparsers.add_parser("resp")
    resp.add_argument("--pred", type=Path, required=True)
    resp.add_argument("--resp", type=Path, nargs="+")
    resp.add_argument("--output", type=Path, required=True)
    resp.add_argument("--cutoff", type=float, required=True)
    resp.add_argument("--sig", type=float, default=0.1)
    resp.add_argument("--confident", default=False, action="store_true")
    return parser.parse_args()


def run_true(args: Namespace) -> None:
    true = nx.read_gml(args.true)
    pred = nx.read_gml(args.pred)
    scf = nx.read_gml(args.scaffold)
    c = args.cutoff

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(
            {
                "acc": disc_acc(true, pred, scaffold=scf, cutoff=c).item(),
                "prec": disc_prec(true, pred, scaffold=scf, cutoff=c).item(),
                "recall": disc_recall(true, pred, scaffold=scf, cutoff=c).item(),
                "f1": disc_f1(true, pred, scaffold=scf, cutoff=c).item(),
                "auroc": disc_auroc(true, pred, scaffold=scf).item(),
                "ap": disc_ap(true, pred, scaffold=scf).item(),
                "shd": disc_shd(true, pred, scaffold=scf, cutoff=c).item(),
                "sid": disc_sid(true, pred, scaffold=scf, cutoff=c).item(),
            },
            f,
        )


def run_resp(args: Namespace) -> None:
    pred = nx.read_gml(args.pred)
    resp = ad.concat([ad.read_h5ad(item) for item in args.resp])

    interv_cols = set(resp.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()

    if args.confident:
        pred = map_edges(pred, fn=mean)
        pred = filter_edges(pred, cutoff=args.cutoff)
    annot_resp(pred, resp, interv_col)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.dump(
            {
                "resp_dist": disc_resp_dist(pred, cutoff=args.cutoff).item(),
                "resp_dist_diff": disc_resp_dist_diff(pred, cutoff=args.cutoff).item(),
                "resp_acc": disc_resp_acc(
                    pred, cutoff=args.cutoff, sig=args.sig
                ).item(),
            },
            f,
        )


def main() -> None:
    args = parse_args()
    if args.mode == "true":
        run_true(args)
    else:  # args.mode == "resp"
        run_resp(args)


if __name__ == "__main__":
    main()
