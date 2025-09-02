import networkx as nx
import pandas as pd


def main():
    go_essential_all = pd.read_csv("go_essential_all/go_essential_all.csv")
    go_essential_all["weight"] = go_essential_all.pop("importance")
    go_essential_all = nx.from_pandas_edgelist(
        go_essential_all, edge_attr=True, create_using=nx.DiGraph
    )
    nx.write_gml(go_essential_all, "go_essential_all.gml.gz")


if __name__ == "__main__":
    main()
