from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import networkx as nx

from cascade.graph import marginalize


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input-data", type=Path, required=True)
    parser.add_argument("--input-graph", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main(args: Namespace) -> None:
    adata = ad.read_h5ad(args.input_data)
    graphs = []
    for item in args.input_graph:
        graph = nx.read_gml(item)
        graph = marginalize(
            graph.to_directed(),
            adata.var_names,
            max_steps=graph.graph["marginalize_steps"],
        )
        nx.set_edge_attributes(graph, graph.graph["data_source"], "data_source")
        nx.set_edge_attributes(graph, graph.graph["evidence_type"], "evidence_type")
        graphs.append(graph)
    scaffold = nx.compose_all(graphs[::-1])  # Make first graphs take precedence
    scaffold.add_nodes_from(adata.var_names)
    scaffold.remove_edges_from([(v, v) for v in scaffold.nodes])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(scaffold, args.output)


if __name__ == "__main__":
    main(parse_args())
