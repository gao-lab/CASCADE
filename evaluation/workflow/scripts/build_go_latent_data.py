from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import networkx as nx


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--input-graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("-k", type=int, default=20)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input_data)
    graph = nx.read_gml(args.input_graph)

    graph = graph.subgraph(adata.var_names)
    graph = nx.from_pandas_edgelist(
        nx.to_pandas_edgelist(graph)
        .groupby("target")
        .apply(lambda x: x.nlargest(args.k + 1, "weight"))
        .reset_index(drop=True),
        edge_attr=True,
        create_using=nx.DiGraph,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(graph, args.output)


if __name__ == "__main__":
    main(parse_args())
