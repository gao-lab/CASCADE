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
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.sparse import coo_array

from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Read components

# %%
kegg = nx.read_gml("../../data/scaffold/KEGG/inferred_kegg_gene_only.gml.gz")
kegg.number_of_nodes(), kegg.number_of_edges()

# %%
tf_target = nx.read_gml("../../data/scaffold/TF-target/TF-target.gml.gz")
tf_target.number_of_nodes(), tf_target.number_of_edges()

# %%
ppi = nx.read_gml("../../data/scaffold/BioGRID/biogrid.gml.gz").to_directed()
ppi.number_of_nodes(), ppi.number_of_edges()

# %%
corr = nx.read_gml("../../data/scaffold/GTEx/corr.gml.gz").to_directed()
corr.number_of_nodes(), corr.number_of_edges()

# %%
type(kegg), type(tf_target), type(ppi), type(corr)

# %% [markdown]
# # Merge components

# %%
genes = pd.Index(sorted(kegg.nodes | tf_target.nodes | ppi.nodes | corr.nodes))
genes.size

# %%
kegg_edgelist = nx.to_pandas_edgelist(kegg)
tf_target_edgelist = nx.to_pandas_edgelist(tf_target)
ppi_edgelist = nx.to_pandas_edgelist(ppi)
corr_edgelist = nx.to_pandas_edgelist(corr)

# %%
kegg_adj = coo_array(
    (
        np.ones(kegg_edgelist.shape[0]),
        np.stack(
            [
                genes.get_indexer(kegg_edgelist["source"]),
                genes.get_indexer(kegg_edgelist["target"]),
            ]
        ),
    ),
    shape=(genes.size, genes.size),
)

# %%
tf_target_adj = coo_array(
    (
        np.ones(tf_target_edgelist.shape[0]),
        np.stack(
            [
                genes.get_indexer(tf_target_edgelist["source"]),
                genes.get_indexer(tf_target_edgelist["target"]),
            ]
        ),
    ),
    shape=(genes.size, genes.size),
)

# %%
ppi_adj = coo_array(
    (
        np.ones(ppi_edgelist.shape[0]),
        np.stack(
            [
                genes.get_indexer(ppi_edgelist["source"]),
                genes.get_indexer(ppi_edgelist["target"]),
            ]
        ),
    ),
    shape=(genes.size, genes.size),
)

# %%
corr_adj = coo_array(
    (
        np.ones(corr_edgelist.shape[0]),
        np.stack(
            [
                genes.get_indexer(corr_edgelist["source"]),
                genes.get_indexer(corr_edgelist["target"]),
            ]
        ),
    ),
    shape=(genes.size, genes.size),
)

# %%
kegg_adj.max(), tf_target_adj.max(), ppi_adj.max(), corr_adj.max()

# %%
combined_adj = kegg_adj + tf_target_adj + ppi_adj + corr_adj
combined = nx.from_scipy_sparse_array(combined_adj, create_using=nx.DiGraph)
nx.relabel_nodes(combined, {i: gene for i, gene in enumerate(genes)}, copy=False)
combined.number_of_nodes(), combined.number_of_edges()

# %%
kegg_adj_df = pd.DataFrame(kegg_adj.toarray(), index=genes, columns=genes)
tf_target_adj_df = pd.DataFrame(tf_target_adj.toarray(), index=genes, columns=genes)
ppi_adj_df = pd.DataFrame(ppi_adj.toarray(), index=genes, columns=genes)
corr_adj_df = pd.DataFrame(corr_adj.toarray(), index=genes, columns=genes)
combined_adj_df = pd.DataFrame(combined_adj.toarray(), index=genes, columns=genes)

# %% [markdown]
# # Visualization

# %%
sub_idx = np.random.RandomState(0).choice(genes.size, 4000, replace=False)

# %%
g = sns.clustermap(
    combined_adj_df.iloc[sub_idx, sub_idx],
    cmap="Blues",
    rasterized=True,
    figsize=(15, 15),
)
g.savefig("combined_adj.pdf")

# %%
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    kegg_adj_df.iloc[sub_idx, sub_idx].iloc[
        g.dendrogram_row.reordered_ind,
        g.dendrogram_col.reordered_ind,
    ],
    cmap="Blues",
    rasterized=True,
    ax=ax,
)
fig.savefig("kegg_adj.pdf")

# %%
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    tf_target_adj_df.iloc[sub_idx, sub_idx].iloc[
        g.dendrogram_row.reordered_ind,
        g.dendrogram_col.reordered_ind,
    ],
    cmap="Blues",
    rasterized=True,
    ax=ax,
)
fig.savefig("tf_target_adj.pdf")

# %%
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    ppi_adj_df.iloc[sub_idx, sub_idx].iloc[
        g.dendrogram_row.reordered_ind,
        g.dendrogram_col.reordered_ind,
    ],
    cmap="Blues",
    rasterized=True,
    ax=ax,
)
fig.savefig("ppi_adj.pdf")

# %%
fig, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(
    corr_adj_df.iloc[sub_idx, sub_idx].iloc[
        g.dendrogram_row.reordered_ind,
        g.dendrogram_col.reordered_ind,
    ],
    cmap="Blues",
    rasterized=True,
    ax=ax,
)
fig.savefig("corr_adj.pdf")

# %%
# from cascade.plot import interactive_heatmap
#
# interactive_heatmap(
#     combined_adj.iloc[sub_idx, sub_idx].iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind,
#     ],
#     row_clust=None,
#     col_clust=None,
#     colorscale="blues",
# )
