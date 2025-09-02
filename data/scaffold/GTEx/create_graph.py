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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy.sparse import coo_array

from cascade.plot import set_figure_params
from cascade.utils import spearman_correlation

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Read data

# %%
X = (
    pd.read_table(
        "GTEx_Analysis_v10_RNASeQCv2.4.2_gene_tpm.gct.gz",
        skiprows=2,
    )
    .drop(columns="Description")
    .set_index("Name")
)
X

# %%
X = X.loc[X.sum(axis=1) != 0]
X.shape

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "ensembl_gene_id", "locus_group"],
    dtype=str,
).query("locus_group == 'protein-coding gene'")
hgnc.head()

# %%
hgnc_map = hgnc.set_index("ensembl_gene_id")["symbol"].to_dict()
X.index = X.index.str.replace(r"\.\d+$", "", regex=True).map(hgnc_map)
X = X.loc[X.index.notna()]
X.shape

# %%
X = X.T
X.shape

# %% [markdown]
# # Compute correlation

# %%
corr = pd.DataFrame(
    spearman_correlation(X.to_numpy()).astype(np.float32),
    index=X.columns,
    columns=X.columns,
)
corr.to_pickle("corr.pkl.gz")

# %%
corr = pd.read_pickle("corr.pkl.gz")

# %%
sub_idx = np.random.RandomState(0).choice(corr.shape[0], 5000, replace=False)
corr_sub = corr.iloc[sub_idx, sub_idx]
g = sns.clustermap(
    corr_sub,
    cmap="bwr",
    center=0,
    vmax=0.8,
    vmin=-0.8,
    rasterized=True,
    figsize=(10, 10),
)
g.fig.savefig("corr_sub.pdf")

# %% [markdown]
# # Create graph

# %%
corr_arr = corr.to_numpy()
corr_argsort = np.argsort(corr_arr, axis=1)

# %%
K = 500
top_k = corr_argsort[:, -(K + 1) :]

# %%
adj = np.zeros_like(corr_arr, dtype=bool)
np.put_along_axis(adj, top_k, True, axis=1)
np.fill_diagonal(adj, False)
adj = adj & adj.T
adj = coo_array(adj).multiply(corr_arr)

# %%
graph = nx.from_scipy_sparse_array(adj)
nx.relabel_nodes(graph, {i: v for i, v in enumerate(corr.index)}, copy=False)
graph.graph["data_source"] = "GTEx"
graph.graph["evidence_type"] = "Correlation"
graph.graph["marginalize_steps"] = 0
graph.number_of_nodes(), graph.number_of_edges()

# %%
nx.write_gml(graph, "corr.gml.gz")

# %% [markdown]
# # Visualization

# %%
adj = nx.to_pandas_adjacency(graph)
sub_idx = np.random.RandomState(0).choice(adj.shape[0], 5000, replace=False)
adj_sub = adj.iloc[sub_idx, sub_idx]

# %%
g = sns.clustermap(
    adj_sub,
    cmap="bwr",
    center=0,
    vmax=0.8,
    vmin=-0.8,
    rasterized=True,
    figsize=(10, 10),
)
g.fig.savefig("adj_sub.pdf")

# %%
# from cascade.plot import interactive_heatmap

# interactive_heatmap(
#     adj_top.iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind,
#     ],
#     row_clust=None,
#     col_clust=None,
#     zmin=-0.8,
#     zmax=0.8,
#     colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
# )
