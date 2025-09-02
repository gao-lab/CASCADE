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
import re

import anndata as ad
import pandas as pd
import scanpy as sc

from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Read HGNC

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
adata = ad.read_h5ad("Cao-2020.h5ad")


# %%
def map_names(row):
    gene_id = re.sub(r"\.\d+$", "", row["gene_id"])
    gene_name = row["gene_name"]
    return hgnc_map.get(gene_id, hgnc_map.get(gene_name, gene_name))


adata.var_names = adata.var.apply(map_names, axis=1)
adata.var_names_make_unique(join=".")

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, layer="counts", n_top_genes=5000, flavor="seurat_v3")

# %%
sc.pp.pca(adata, n_comps=100, mask_var="highly_variable")

# %% [markdown]
# # Write data

# %%
adata.write("../../datasets/Cao-2020.h5ad", compression="gzip")
