from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import networkx as nx
import pandas as pd
import scanpy as sc

from cascade.data import Targets


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-graph", type=Path, required=True)
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--output-graph", type=Path, required=True)
    parser.add_argument("--output-data", type=Path, required=True)
    parser.add_argument("--n-vars", type=int, default=500)
    return parser.parse_args()


def main(args: Namespace) -> None:
    graph = nx.read_gml(args.input_graph)
    adata = ad.read_h5ad(args.input_data)
    # assert (adata.var_names == pd.Index(graph.nodes)).all()

    interv_cols = set(adata.obs.columns) & {"knockout", "knockdown", "knockup"}
    assert len(interv_cols) == 1
    interv_col = interv_cols.pop()
    adata.obs[interv_col] = adata.obs[interv_col].map(Targets)
    edist = adata.obs[[interv_col, "edist"]].drop_duplicates(subset=interv_col).dropna()
    max_edist_per_var = {}
    for targets, edist in zip(edist[interv_col], edist["edist"]):
        for target in targets:
            if target in max_edist_per_var:
                max_edist_per_var[target] = max(max_edist_per_var[target], edist)
            else:
                max_edist_per_var[target] = edist
    adata.var["max_edist"] = adata.var_names.map(max_edist_per_var)
    adata.var["has_edist"] = ~adata.var["max_edist"].isna()

    keep_vars = (
        adata.var.query("has_edist")
        .sort_values("max_edist", ascending=False)
        .index[: args.n_vars]
    )
    if keep_vars.size < args.n_vars:
        de_occurrences = pd.concat(
            [
                sc.get.rank_genes_groups_df(adata, var, pval_cutoff=0.01)["names"]
                for var in keep_vars
            ]
        ).value_counts()
        adata.var["de_occurrences"] = adata.var_names.map(de_occurrences)
        adata.var["variances_norm_bin"] = pd.qcut(adata.var["variances_norm"], 20)
        n_vars_remain = args.n_vars - keep_vars.size
        keep_vars = keep_vars.append(
            adata.var.query("not has_edist")
            .sort_values(["variances_norm_bin", "de_occurrences"], ascending=False)
            .index[:n_vars_remain]
        )
        del adata.var["de_occurrences"], adata.var["variances_norm_bin"]

    var_left = set(keep_vars) | {""}
    obs_mask = adata.obs[interv_col].map(lambda x: not (x - var_left)).astype(bool)
    sub_adata = adata[obs_mask, keep_vars].copy()
    sub_graph = graph.subgraph(sub_adata.var_names)

    args.output_graph.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(sub_graph, args.output_graph)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    sub_adata.obs[interv_col] = sub_adata.obs[interv_col].astype(str)
    sub_adata.write(args.output_data, compression="gzip")


if __name__ == "__main__":
    main(parse_args())
