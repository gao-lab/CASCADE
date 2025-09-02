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
from scipy.sparse import csr_matrix

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

# %% [markdown]
# ## GSE137864

# %%
barcodes = np.loadtxt("GSE137864_barcodes.tsv.gz", dtype=str)
barcodes.shape

# %%
features = pd.read_table(
    "GSE137864_genes.tsv.gz", names=["gene_id", "gene_name"], index_col="gene_id"
)
features.shape

# %%
X = mmread("GSE137864_matrix.mtx.gz")
X = X.T.tocsr().astype(np.float32)
X.shape

# %%
adata1 = ad.AnnData(X, obs=pd.DataFrame(index=barcodes), var=features)
adata1

# %%
obs_split1 = pd.Series(adata1.obs_names).str.split("_", expand=True)
adata1.obs["cell_type"] = obs_split1[0].to_numpy()
adata1.obs["sample"] = obs_split1[1].to_numpy()
adata1.obs["library"] = obs_split1[2].to_numpy()
adata1.obs["barcode"] = obs_split1[3].to_numpy()
adata1.obs.head()

# %%
adata1.var.head()

# %% [markdown]
# ## GSE149938

# %%
adata2 = ad.AnnData(pd.read_csv("GSE149938_umi_matrix.csv.gz"))
adata2.X = csr_matrix(adata2.X).astype(np.float32)
adata2

# %%
obs_split2 = pd.Series(adata2.obs_names).str.split("_", expand=True)
adata2.obs["cell_type"] = obs_split2[0].to_numpy()
adata2.obs["sample"] = obs_split2[1].to_numpy()
adata2.obs["library"] = obs_split2[2].to_numpy()
adata2.obs["barcode"] = obs_split2[3].to_numpy()
adata2.obs.head()

# %%
adata2.var = adata1.var.copy()  # They are the same
adata2.var.head()

# %% [markdown]
# # Merge data

# %%
adata = ad.concat({"GSE137864": adata1, "GSE149938": adata2}, label="GEO", merge="same")
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
sc.pl.umap(adata, color=["cell_type", "sample", "library", "GEO"], wspace=0.4)

# %% [markdown]
# # Write data

# %%
adata.write("../../datasets/Xie-2021.h5ad", compression="gzip")
