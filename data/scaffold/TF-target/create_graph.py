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

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Read data

# %% [markdown]
# ## HGNC

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "ensembl_gene_id", "prev_symbol", "alias_symbol", "locus_group"],
    dtype=str,
).query("locus_group == 'protein-coding gene'")
hgnc.head()

# %%
hgnc_map = {}
for symbol, ensembl_gene_id in zip(hgnc["symbol"], hgnc["ensembl_gene_id"]):
    hgnc_map[symbol] = symbol
    hgnc_map[ensembl_gene_id] = symbol
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
hgnc_map["EZH2phosphoT487"] = "EZH2"
hgnc_map["POLR2AphosphoS2"] = "POLR2A"
hgnc_map["POLR2AphosphoS5"] = "POLR2A"
hgnc_map["DUX1_HUMAN"] = "DUX1"
hgnc_map["DUX3_HUMAN"] = "DUX3"

# %% [markdown]
# ## TF list

# %%
tfs = (
    pd.read_csv(
        "TFs/DatabaseExtract_v_1.01.csv",
        usecols=["Ensembl ID", "HGNC symbol", "Is TF?"],
    )
    .query("`Is TF?` == 'Yes'")
    .drop(columns="Is TF?")
)
tfs.head()


# %%
def tf_rename(row):
    ensembl_id, symbol = row
    return hgnc_map.get(symbol, hgnc_map.get(ensembl_id, np.nan))


tfs = set(tfs.apply(tf_rename, axis=1).dropna())
len(tfs)

# %%
np.savetxt("tfs.txt", sorted(tfs), fmt="%s")

# %% [markdown]
# ## Edges

# %%
edgelist = pd.read_csv(
    "ChIP-based/tf-gene.csv.gz", header=None, names=["source", "target", "count"]
)
edgelist.head()

# %%
edgelist["source"] = edgelist["source"].map(hgnc_map)
edgelist["target"] = edgelist["target"].map(hgnc_map)
edgelist.shape

# %%
edgelist = edgelist.dropna()
edgelist.shape

# %%
edgelist = edgelist.loc[edgelist["source"].isin(tfs)]
edgelist.shape

# %% [markdown]
# # Create graph

# %%
genes = pd.Index(sorted(set(edgelist["source"]) | set(edgelist["target"])))
i = genes.get_indexer(edgelist["source"])
j = genes.get_indexer(edgelist["target"])
adj = coo_array((edgelist["count"], (i, j)), shape=(genes.size, genes.size))
adj = pd.DataFrame(adj.toarray(), index=genes, columns=genes)

# %%
adj = adj.loc[adj.sum(axis=1) > 0]
adj.shape

# %%
sub_idx = np.random.RandomState(0).choice(adj.shape[1], 5000, replace=False)

# %%
g = sns.clustermap(
    np.log1p(adj.iloc[:, sub_idx]), cmap="Blues", rasterized=True, figsize=(12, 8)
)
g.savefig("log_counts.pdf")


# %% [markdown]
# ## Normalize by both rows and columns


# %%
def obs_exp_ratio(df):
    x = df.to_numpy()
    total_sum = x.sum()
    row_prob = x.sum(axis=1, keepdims=True) / total_sum
    col_prob = x.sum(axis=0, keepdims=True) / total_sum
    obs_prob = x / total_sum + 0.1 / total_sum
    exp_prob = row_prob * col_prob + 0.1 / total_sum
    return pd.DataFrame(obs_prob / exp_prob, index=df.index, columns=df.columns)


# %%
adj_norm = obs_exp_ratio(adj)
adj_norm.max().max()

# %%
g = sns.clustermap(
    adj_norm.iloc[:, sub_idx].clip(0, 2), cmap="Blues", rasterized=True, figsize=(12, 8)
)
g.savefig("obs_exp_ratio.pdf")

# %%
adj_rnd = pd.DataFrame(
    np.random.RandomState(42)
    .permutation(adj.to_numpy().ravel())
    .reshape((adj.shape[0], adj.shape[1])),
    index=adj.index,
    columns=adj.columns,
)
adj_rnd_norm = obs_exp_ratio(adj_rnd)

# %%
cutoff = np.quantile(adj_rnd_norm.to_numpy().ravel(), 0.85)
cutoff

# %%
g = sns.displot(np.random.choice(adj_rnd_norm.to_numpy().ravel(), 50000, replace=True))
g.ax.axvline(x=cutoff, c="darkred", ls="--")
g.savefig("rnd_dist.pdf")

# %%
adj_norm_binary = (adj_norm > cutoff).astype(float)

# %%
g = sns.clustermap(
    adj_norm_binary.iloc[:, sub_idx], cmap="Blues", rasterized=True, figsize=(12, 8)
)
g.savefig("binary.pdf")

# %%
edgelist = (
    adj_norm_binary.reset_index()
    .rename(columns={"index": "source"})
    .melt(id_vars=["source"], var_name="target", value_name="weight")
    .query("weight > 0")
)

# %%
graph = nx.from_pandas_edgelist(edgelist, create_using=nx.DiGraph)
graph.graph["data_source"] = "ENCODE-GeneHancer"
graph.graph["evidence_type"] = "TF-target"
graph.graph["marginalize_steps"] = 0
graph.number_of_nodes(), graph.number_of_edges()

# %%
nx.write_gml(graph, "TF-target.gml.gz")

# %%
# from cascade.plot import interactive_heatmap

# interactive_heatmap(
#     adj_norm_binary.iloc[:, sub_idx].iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind,
#     ],
#     row_clust=None,
#     col_clust=None,
#     colorscale="blues",
#     height=640,
#     width=960,
# )
