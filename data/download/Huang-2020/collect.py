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
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread

from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # HGNC

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "ensembl_gene_id", "prev_symbol", "alias_symbol"],
    dtype=str,
)
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


# %% [markdown]
# # Read data


# %%
def qc(adata):
    adata.var["mt"] = adata.var["gene_name"].str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=False, inplace=True)

    adata.obs["log_total_counts"] = np.log(adata.obs["total_counts"])
    log_total_counts_lower = (
        adata.obs["log_total_counts"].mean() - 2 * adata.obs["log_total_counts"].std()
    )
    log_total_counts_upper = (
        adata.obs["log_total_counts"].mean() + 2 * adata.obs["log_total_counts"].std()
    )

    return adata[
        (adata.obs["n_genes_by_counts"] >= 200)
        & (adata.obs["pct_counts_mt"] < 5)
        & (adata.obs["log_total_counts"] > log_total_counts_lower)
        & (adata.obs["log_total_counts"] < log_total_counts_upper)
    ].copy()


# %% [markdown]
# ## UCB2-3

# %% [markdown]
# ### Create

# %%
ucb23_barcodes = np.loadtxt("HP0_filtered_gene_bc_matrices/barcodes.tsv.gz", dtype=str)
ucb23_barcodes.shape

# %%
ucb23_features = pd.read_table(
    "HP0_filtered_gene_bc_matrices/features.tsv.gz",
    names=["gene_id", "gene_name", "modality"],
    index_col="gene_id",
)
ucb23_features.shape

# %%
ucb23_X = mmread("HP0_filtered_gene_bc_matrices/matrix.mtx.gz")
ucb23_X = ucb23_X.T.tocsr().astype(np.float32)
ucb23_X.shape

# %%
ucb23 = ad.AnnData(
    ucb23_X,
    obs=pd.DataFrame(index=ucb23_barcodes).assign(source="UCB", batch="UCB2-3"),
    var=ucb23_features,
)
ucb23

# %% [markdown]
# ### QC

# %%
ucb23_qc = qc(ucb23)
ucb23_qc

# %% [markdown]
# ## UCB1

# %% [markdown]
# ### Create

# %%
ucb1_barcodes = np.loadtxt("HP1_filtered_gene_bc_matrices/barcodes.tsv", dtype=str)
ucb1_barcodes.shape

# %%
ucb1_features = pd.read_table(
    "HP1_filtered_gene_bc_matrices/genes.tsv",
    names=["gene_id", "gene_name"],
    index_col="gene_id",
)
ucb1_features.shape

# %%
ucb1_X = mmread("HP1_filtered_gene_bc_matrices/matrix.mtx")
ucb1_X = ucb1_X.T.tocsr().astype(np.float32)
ucb1_X.shape

# %%
ucb1 = ad.AnnData(
    ucb1_X,
    obs=pd.DataFrame(index=ucb1_barcodes).assign(source="UCB", batch="UCB1"),
    var=ucb1_features,
)
ucb1

# %% [markdown]
# ### QC

# %%
ucb1_qc = qc(ucb1)
ucb1_qc

# %% [markdown]
# ## BM1

# %% [markdown]
# ### Create

# %%
bm1_barcodes = np.loadtxt("HP2_filtered_gene_bc_matrices/barcodes.tsv", dtype=str)
bm1_barcodes.shape

# %%
bm1_features = pd.read_table(
    "HP2_filtered_gene_bc_matrices/genes.tsv",
    names=["gene_id", "gene_name"],
    index_col="gene_id",
)
bm1_features.shape

# %%
bm1_X = mmread("HP2_filtered_gene_bc_matrices/matrix.mtx")
bm1_X = bm1_X.T.tocsr().astype(np.float32)
bm1_X.shape

# %%
bm1 = ad.AnnData(
    bm1_X,
    obs=pd.DataFrame(index=bm1_barcodes).assign(source="BM", batch="BM1"),
    var=bm1_features,
)
bm1

# %% [markdown]
# ### QC

# %%
bm1_qc = qc(bm1)
bm1_qc

# %% [markdown]
# ## BM2

# %% [markdown]
# ### Create

# %%
bm2_barcodes = np.loadtxt("HP3_filtered_gene_bc_matrices/barcodes.tsv", dtype=str)
bm2_barcodes.shape

# %%
bm2_features = pd.read_table(
    "HP3_filtered_gene_bc_matrices/genes.tsv",
    names=["gene_id", "gene_name"],
    index_col="gene_id",
)
bm2_features.shape

# %%
bm2_X = mmread("HP3_filtered_gene_bc_matrices/matrix.mtx")
bm2_X = bm2_X.T.tocsr().astype(np.float32)
bm2_X.shape

# %%
bm2 = ad.AnnData(
    bm2_X,
    obs=pd.DataFrame(index=bm2_barcodes).assign(source="BM", batch="BM2"),
    var=bm2_features,
)
bm2

# %% [markdown]
# ### QC

# %%
bm2_qc = qc(bm2)
bm2_qc

# %% [markdown]
# ## BM3

# %% [markdown]
# ### Create

# %%
bm3_barcodes = np.loadtxt("HP4_filtered_gene_bc_matrices/barcodes.tsv", dtype=str)
bm3_barcodes.shape

# %%
bm3_features = pd.read_table(
    "HP4_filtered_gene_bc_matrices/genes.tsv",
    names=["gene_id", "gene_name"],
    index_col="gene_id",
)
bm3_features.shape

# %%
bm3_X = mmread("HP4_filtered_gene_bc_matrices/matrix.mtx")
bm3_X = bm3_X.T.tocsr().astype(np.float32)
bm3_X.shape

# %%
bm3 = ad.AnnData(
    bm3_X,
    obs=pd.DataFrame(index=bm3_barcodes).assign(source="BM", batch="BM3"),
    var=bm3_features,
)
bm3

# %% [markdown]
# ### QC

# %%
bm3_qc = qc(bm3)
bm3_qc

# %% [markdown]
# # Merge data

# %%
adata = ad.concat(
    [ucb1_qc, ucb23_qc, bm1_qc, bm2_qc, bm3_qc], merge="same", index_unique="-"
)
adata


# %% [markdown]
# # Preprocessing


# %%
def map_name(x):
    author_name = adata.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


sc.pp.filter_genes(adata, min_counts=1)
adata.var_names = adata.var_names.map(map_name)
adata.var_names_make_unique(join=".")
adata

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, layer="counts", n_top_genes=5000, flavor="seurat_v3")

# %%
sc.pp.pca(adata, mask_var=(adata.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(adata, metric="cosine")
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["source", "batch"], wspace=0.4)

# %% [markdown]
# # Write data

# %%
adata.write("../../datasets/Huang-2020.h5ad", compression="gzip")
