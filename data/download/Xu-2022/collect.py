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

from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # HGNC

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

# %% [markdown]
# # 10x portion

# %% [markdown]
# ## Read data

# %%
cs10_ys = ad.read_loom("CS10_YS.loom")
cs10_ys.layers.clear()
cs10_ys.var_names = cs10_ys.var_names.map(lambda x: hgnc_map.get(x, x))
cs10_ys.var_names_make_unique(join=".")
cs10_ys

# %%
cs11_ys = ad.read_loom("CS11_YS.loom")
cs11_ys.layers.clear()
cs11_ys.var_names = cs11_ys.var_names.map(lambda x: hgnc_map.get(x, x))
cs11_ys.var_names_make_unique(join=".")
cs11_ys

# %%
cs15_ys = ad.read_loom("CS15_YS.loom")
cs15_ys.layers.clear()
cs15_ys.var_names = cs15_ys.var_names.map(lambda x: hgnc_map.get(x, x))
cs15_ys.var_names_make_unique(join=".")
cs15_ys

# %%
fl_ery_1 = ad.read_loom("FL_Ery_1.loom")
fl_ery_1.layers.clear()
fl_ery_1.var_names = fl_ery_1.var_names.map(lambda x: hgnc_map.get(x, x))
fl_ery_1.var_names_make_unique(join=".")
fl_ery_1

# %%
fl_ery_2 = ad.read_loom("FL_Ery_2.loom")
fl_ery_2.layers.clear()
fl_ery_2.var_names = fl_ery_2.var_names.map(lambda x: hgnc_map.get(x, x))
fl_ery_2.var_names_make_unique(join=".")
fl_ery_2

# %%
fl_ery_3 = ad.read_loom("FL_Ery_3.loom")
fl_ery_3.layers.clear()
fl_ery_3.var_names = fl_ery_3.var_names.map(lambda x: hgnc_map.get(x, x))
fl_ery_3.var_names_make_unique(join=".")
fl_ery_3

# %%
ucb_facs_r1 = ad.read_loom("UCB_FACS_R1.loom")
ucb_facs_r1.layers.clear()
ucb_facs_r1.var_names = ucb_facs_r1.var_names.map(lambda x: hgnc_map.get(x, x))
ucb_facs_r1.var_names_make_unique(join=".")
ucb_facs_r1

# %%
ucb_facs_r2 = ad.read_loom("UCB_FACS_R2.loom")
ucb_facs_r2.layers.clear()
ucb_facs_r2.var_names = ucb_facs_r2.var_names.map(lambda x: hgnc_map.get(x, x))
ucb_facs_r2.var_names_make_unique(join=".")
ucb_facs_r2

# %%
ucb_facs_r3 = ad.read_loom("UCB_FACS_R3.loom")
ucb_facs_r3.layers.clear()
ucb_facs_r3.var_names = ucb_facs_r3.var_names.map(lambda x: hgnc_map.get(x, x))
ucb_facs_r3.var_names_make_unique(join=".")
ucb_facs_r3

# %% [markdown]
# ## Merge data

# %%
adata = ad.concat(
    {
        "CS10_YS": cs10_ys,
        "CS11_YS": cs11_ys,
        "CS15_YS": cs15_ys,
        "FL_Ery_1": fl_ery_1,
        "FL_Ery_2": fl_ery_2,
        "FL_Ery_3": fl_ery_3,
        "UCB_FACS_R1": ucb_facs_r1,
        "UCB_FACS_R2": ucb_facs_r2,
        "UCB_FACS_R3": ucb_facs_r3,
    },
    label="batch",
)
adata.obs["Clusters"] = adata.obs["Clusters"].astype("category")
adata.obs["stage"] = adata.obs["batch"].str.replace(
    r"(^CS\d{2}_|_Ery_\d$|_FACS_R\d$)", "", regex=True
)
adata

# %% [markdown]
# ## Preprocessing

# %%
sc.pp.filter_genes(adata, min_counts=1)
adata

# %%
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(
    adata, layer="counts", n_top_genes=5000, batch_key="batch", flavor="seurat_v3"
)

# %%
sc.pp.pca(adata, mask_var=(adata.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(adata, metric="cosine")
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["Clusters", "batch"], wspace=0.4)

# %% [markdown]
# ## Write data

# %%
adata.write("../../datasets/Xu-2022.h5ad", compression="gzip")

# %% [markdown]
# # STRT portion

# %% [markdown]
# ## Read data

# %%
bm = ad.read_h5ad("human_BM_Ery_smart_seq2.h5ad")
del bm.raw
bm.X = bm.X.astype(np.float32)
bm

# %% [markdown]
# ## Preprocessing

# %%
sc.pp.filter_genes(bm, min_counts=1)
bm.var_names = bm.var_names.map(lambda x: hgnc_map.get(x, x))
bm.var_names_make_unique(join=".")
bm

# %%
bm.layers["counts"] = bm.X.copy()
sc.pp.normalize_total(bm, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(bm)

# %%
sc.pp.highly_variable_genes(
    bm, layer="counts", n_top_genes=5000, batch_key="Sample_ID", flavor="seurat_v3"
)

# %%
sc.pp.pca(bm, mask_var=(bm.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(bm, metric="cosine")
sc.tl.umap(bm)

# %%
sc.pl.umap(bm, color=["cluster", "Sample_ID", "Plate_ID"], wspace=0.3)

# %% [markdown]
# ## Write data

# %%
bm.write("../../datasets/Xu-2022-BM.h5ad", compression="gzip")
