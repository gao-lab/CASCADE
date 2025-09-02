# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
import pandas as pd
import seaborn as sns

# %% [markdown]
# # Read data

# %%
edgelist = pd.read_table(
    "trrust_rawdata.human.tsv", names=["source", "target", "type", "ref"]
)
edgelist

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "prev_symbol", "alias_symbol"],
    dtype=str,
)
hgnc.head()

# %%
hgnc_map = {}
for symbol in hgnc["symbol"]:
    hgnc_map[symbol] = symbol
for symbol, prev_symbol in zip(hgnc["symbol"], hgnc["prev_symbol"]):
    if isinstance(prev_symbol, str):
        for item in prev_symbol.split("|"):
            if item not in hgnc_map:
                hgnc_map[item] = symbol
for symbol, alias_symbol in zip(hgnc["symbol"], hgnc["alias_symbol"]):
    if isinstance(alias_symbol, str):
        for item in alias_symbol.split("|"):
            if item not in hgnc_map:
                hgnc_map[item] = symbol

hgnc_map["SALL4A"] = "SALL4"

# %%
edgelist["source"] = edgelist["source"].map(hgnc_map)
edgelist["target"] = edgelist["target"].map(hgnc_map)

# %% [markdown]
# # Create graph

# %%
graph = nx.from_pandas_edgelist(
    edgelist.dropna(), edge_attr=True, create_using=nx.DiGraph
)
graph.graph["data_source"] = "TRRUST"
graph.graph["evidence_type"] = "Literature"
graph.graph["marginalize_steps"] = 1
graph.number_of_nodes(), graph.number_of_edges()

# %%
nx.write_gml(graph, "trrust.gml.gz")

# %% [markdown]
# # Visualization

# %%
adj = nx.to_pandas_adjacency(graph)

# %%
g = sns.clustermap(
    adj.loc[adj.sum(axis=1) > 0, adj.sum(axis=0) > 0],
    cmap="Blues",
    rasterized=True,
)
g.savefig("adj.pdf")
