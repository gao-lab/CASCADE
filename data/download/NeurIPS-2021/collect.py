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
import pandas as pd
import scanpy as sc

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
multiome = ad.read_h5ad(
    "GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
)

# %%
cite = ad.read_h5ad("GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")

# %% [markdown]
# # Split data

# %%
multiome.X = multiome.layers.pop("counts")
cite.X = cite.layers.pop("counts")

# %%
multiome_gex = multiome[:, multiome.var["feature_types"] == "GEX"].copy()
multiome_atac = multiome[:, multiome.var["feature_types"] == "ATAC"].copy()

# %%
cite_gex = cite[:, cite.var["feature_types"] == "GEX"].copy()
cite_adt = cite[:, cite.var["feature_types"] == "ADT"].copy()

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ## Multiome

# %%
multiome_gex.var["gene_id"] = multiome_gex.var["gene_id"].astype(str)
multiome_gex.var = (
    multiome_gex.var.reset_index()
    .rename(columns={"index": "gene_name"})
    .set_index("gene_id", drop=False)
)
multiome_gex


# %%
def map_name(x):
    author_name = multiome_gex.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


sc.pp.filter_genes(multiome_gex, min_counts=1)
multiome_gex.var_names = multiome_gex.var_names.map(map_name)
multiome_gex.var_names.name = None
multiome_gex.var_names_make_unique(join=".")
multiome_gex

# %%
multiome_gex.layers["counts"] = multiome_gex.X.copy()
sc.pp.normalize_total(multiome_gex, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(multiome_gex)

# %%
sc.pp.highly_variable_genes(
    multiome_gex,
    layer="counts",
    batch_key="batch",
    n_top_genes=5000,
    flavor="seurat_v3",
)

# %%
sc.pp.pca(
    multiome_gex, mask_var=(multiome_gex.var["highly_variable_rank"] < 2000).to_numpy()
)
sc.pp.neighbors(multiome_gex, metric="cosine")
sc.tl.umap(multiome_gex)

# %%
sc.pl.umap(multiome_gex, color=["batch", "cell_type"], wspace=0.3)

# %% [markdown]
# ## CITE

# %%
cite_gex.var["gene_id"] = cite_gex.var["gene_id"].astype(str)
cite_gex.var = (
    cite_gex.var.reset_index()
    .rename(columns={"index": "gene_name"})
    .set_index("gene_id", drop=False)
)
cite_gex


# %%
def map_name(x):
    author_name = cite_gex.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


sc.pp.filter_genes(cite_gex, min_counts=1)
cite_gex.var_names = cite_gex.var_names.map(map_name)
cite_gex.var_names.name = None
cite_gex.var_names_make_unique(join=".")
cite_gex

# %%
cite_gex.layers["counts"] = cite_gex.X.copy()
sc.pp.normalize_total(cite_gex, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(cite_gex)

# %%
sc.pp.highly_variable_genes(
    cite_gex,
    layer="counts",
    batch_key="batch",
    n_top_genes=5000,
    flavor="seurat_v3",
)

# %%
sc.pp.pca(cite_gex, mask_var=(cite_gex.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(cite_gex, metric="cosine")
sc.tl.umap(cite_gex)

# %%
sc.pl.umap(cite_gex, color=["batch", "cell_type"], wspace=0.3)

# %% [markdown]
# # Write data

# %%
multiome_gex.write("../../datasets/NeurIPS-2021-Multiome-GEX.h5ad", compression="gzip")

# %%
multiome_atac.write(
    "../../datasets/NeurIPS-2021-Multiome-ATAC.h5ad", compression="gzip"
)

# %%
cite_gex.write("../../datasets/NeurIPS-2021-CITE-GEX.h5ad", compression="gzip")

# %%
cite_adt.write("../../datasets/NeurIPS-2021-CITE-ADT.h5ad", compression="gzip")
