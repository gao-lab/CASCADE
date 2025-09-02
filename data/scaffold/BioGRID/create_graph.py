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
from matplotlib import rcParams

from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Read data

# %%
# !unzip BIOGRID-ORGANISM-4.4.239.tab3.zip \
#     BIOGRID-ORGANISM-Homo_sapiens-4.4.239.tab3.txt

# %%
edgelist = pd.read_table(
    "BIOGRID-ORGANISM-Homo_sapiens-4.4.239.tab3.txt",
    usecols=[
        "Entrez Gene Interactor A",
        "Entrez Gene Interactor B",
        "Experimental System",
        "Experimental System Type",
    ],
    dtype=str,
).rename(
    columns={
        "Entrez Gene Interactor A": "entrez_A",
        "Entrez Gene Interactor B": "entrez_B",
        "Experimental System": "experimental_system",
        "Experimental System Type": "experimental_system_type",
    }
)
edgelist.head()

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "entrez_id"],
    dtype=str,
)
hgnc.head()

# %%
hgnc_map = hgnc.set_index("entrez_id")["symbol"].to_dict()
edgelist["hgnc_A"] = edgelist["entrez_A"].map(hgnc_map)
edgelist["hgnc_B"] = edgelist["entrez_B"].map(hgnc_map)
edgelist.head()

# %%
graph = nx.from_pandas_edgelist(
    edgelist.dropna(),
    source="hgnc_A",
    target="hgnc_B",
    edge_attr=["experimental_system", "experimental_system_type"],
)
graph.graph["data_source"] = "BioGRID"
graph.graph["evidence_type"] = "PPI"
graph.graph["marginalize_steps"] = 0
graph.number_of_nodes(), graph.number_of_edges()

# %%
nx.write_gml(graph, "biogrid.gml.gz")

# %% [markdown]
# # Visualization

# %%
adj = nx.to_pandas_adjacency(graph)
top_genes = adj.sum().sort_values(ascending=False).head(5000).index
adj_top = adj.loc[top_genes, top_genes]

# %%
g = sns.clustermap(adj_top, cmap="Blues", rasterized=True, figsize=(10, 10))
g.savefig("adj_top.pdf")

# %%
# interactive_heatmap(
#     adj_top.iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind,
#     ],
#     row_clust=None,
#     col_clust=None,
#     colorscale="blues",
# )
