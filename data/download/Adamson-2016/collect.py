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
import warnings

import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from pandas.errors import PerformanceWarning
from scipy.sparse import csr_matrix
from scperturb import edist_to_control, equal_subsampling

from cascade.data import filter_unobserved_targets, get_all_targets
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
adata = sc.read_h5ad("AdamsonWeissman2016_GSM2406681_10X010.h5ad")
adata.X = adata.X.tocsr()
adata


# %% [markdown]
# # Rectify meta data

# %% [markdown]
# ## Rectify gene names


# %%
def map_name(x):
    author_name = adata.var.loc[x, "gene_symbol"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


adata.var = adata.var.reset_index().set_index("ensembl_id", drop=False)
adata.var_names = adata.var_names.map(map_name)
adata.var_names.name = None
adata

# %% [markdown]
# ## Rectify perturbation labels

# %%
remap = {"*": "", "62(mod)": "", "63(mod)": "", "Gal4-4(mod)": "", **hgnc_map}
adata.obs["knockdown"] = (
    adata.obs["perturbation"]
    .str.split("_", expand=True)[0]
    .fillna("")
    .map(lambda x: remap.get(x, x))
)

# %% [markdown]
# ## Remove unmeasured perturbations

# %%
sc.pp.filter_genes(adata, min_counts=1)

# %%
perturbed = get_all_targets(adata, "knockdown")
_ = venn2([set(adata.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"])

# %%
adata = filter_unobserved_targets(adata, "knockdown")
sc.pp.filter_genes(adata, min_counts=1)
perturbed = get_all_targets(adata, "knockdown")
assert not perturbed - set(adata.var_names)
adata

# %% [markdown]
# # Normalization and highly-variable gene selection

# %%
adata.layers["counts"] = adata.X.copy()

# %%
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(adata, layer="counts", n_top_genes=5000, flavor="seurat_v3")

# %% [markdown]
# # Mixscape

# %% [markdown]
# ## Compute perturbation signature

# %%
sc.pp.pca(adata, mask_var=(adata.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(adata, metric="cosine")
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["percent_mito", "percent_ribo", "nperts"])

# %%
ms = pt.tl.Mixscape()
adata.X = adata.X.toarray()  # Sparse matrix triggers a bug
ms.perturbation_signature(adata, pert_key="knockdown", control="")
adata.X = csr_matrix(adata.X)

# %%
sc.pp.pca(
    adata,
    layer="X_pert",
    mask_var=(adata.var["highly_variable_rank"] < 2000).to_numpy(),
)
sc.pp.neighbors(adata, metric="cosine")
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["percent_mito", "percent_ribo", "nperts"])

# %% [markdown]
# ## Identify non-perturbed cells

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=adata, control="", labels="knockdown", layer="X_pert")

# %%
ms_head = adata.obs.loc[:, ["knockdown", "mixscape_class_global"]].copy()
ms_head["knockdown"] = ms_head["knockdown"].cat.reorder_categories(
    ms_head["knockdown"].value_counts().index.to_numpy()
)
ms_head = ms_head.loc[
    (ms_head["knockdown"].cat.codes < 30) & (ms_head["knockdown"] != "")
]
ms_head["knockdown"] = ms_head["knockdown"].cat.remove_unused_categories()

# %%
so.Plot(ms_head, y="knockdown", color="mixscape_class_global").add(
    so.Bar(), so.Count(), so.Stack()
).layout(size=(4, 6))

# %% [markdown]
# # Remove non-perturbed cells and rare perturbations

# %%
adata_ms = adata[adata.obs["mixscape_class_global"] != "NP"].copy()
adata_ms

# %%
adata_ms.obs = adata_ms.obs.join(
    adata_ms.obs["knockdown"].value_counts().rename("population"), on="knockdown"
)

# %%
adata_ms = adata_ms[adata_ms.obs["population"] >= 10]
sc.pp.filter_genes(adata_ms, min_counts=1)
perturbed = get_all_targets(adata_ms, "knockdown")
assert not perturbed - set(adata_ms.var_names)
adata_ms

# %%
adata_ms.var["perturbed"] = adata_ms.var_names.isin(perturbed)
adata_ms.var["perturbed"].sum()

# %%
adata_ms.var_names_make_unique(join=".")

# %% [markdown]
# # E-distance

# %%
np.random.seed(42)
adata_ms_eq = equal_subsampling(adata_ms, "knockdown", N_min=100)
sc.pp.highly_variable_genes(
    adata_ms_eq, n_top_genes=2000, flavor="seurat_v3", layer="counts"
)
sc.pp.pca(adata_ms_eq, use_highly_variable=True)
sc.pp.neighbors(adata_ms_eq)

# %%
edist = (
    edist_to_control(adata_ms_eq, obs_key="knockdown", control="", n_jobs=1)
    .rename(columns={"distance": "edist"})
    .sort_values("edist", ascending=False)
)
edist.index = edist.index.reorder_categories(edist.index.to_numpy())
edist.head()

# %%
fig, ax = plt.subplots(figsize=(4, 8))
so.Plot(edist, x="edist", y="knockdown").add(so.Line()).on(ax).plot()
ax.axhline(y=np.where(edist.index == "")[0][0], c="darkred", ls="--")
ax.tick_params(axis="y", labelsize=5)

# %%
sc.pl.umap(
    adata_ms,
    color="knockdown",
    groups=edist.index[-20:].tolist(),
    palette="tab20",
    size=20,
)

# %%
sc.pl.umap(
    adata_ms,
    color="knockdown",
    groups=edist.index[:20].tolist(),
    palette="tab20",
    size=20,
)

# %%
adata_ms.obs = adata_ms.obs.join(edist, on="knockdown")

# %%
so.Plot(
    adata_ms.obs.loc[:, ["edist", "population"]].drop_duplicates(),
    x="edist",
    y="population",
).add(so.Dot(pointsize=3)).scale(y="log")

# %% [markdown]
# # Differential expression

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(
        adata_ms,
        "knockdown",
        groups=edist.index[:1000].to_list(),
        reference="",
        method="t-test",
        rankby_abs=True,
    )

# %% [markdown]
# # Perturbation space

# %%
ps = pt.tl.PseudobulkSpace()
adata_ms.X = adata_ms.X.toarray()  # Sparse matrix triggers a bug
adata_ps = ps.compute(
    adata_ms,
    target_col="knockdown",
    mode="mean",
    min_cells=0,
    min_counts=0,
)
adata_ms.X = csr_matrix(adata_ms.X)

# %%
sc.pp.neighbors(adata_ps)
sc.tl.umap(adata_ps)

# %%
sc.pl.umap(adata_ps)

# %% [markdown]
# ## Covariates

# %% [markdown]
# ### Total counts

# %%
log_ncounts = np.log1p(adata_ms.obs[["ncounts"]]).to_numpy()

# %%
log_ncounts = (log_ncounts - log_ncounts.mean()) / log_ncounts.std()
_ = sns.histplot(log_ncounts)

# %% [markdown]
# ### Is perturbed

# %%
is_perturbed = (adata_ms.obs[["knockdown"]] != "").to_numpy()

# %%
_ = sns.histplot(is_perturbed)

# %% [markdown]
# ### Combine

# %%
adata_ms.obsm["covariate"] = np.concatenate([log_ncounts, is_perturbed], axis=1)

# %% [markdown]
# # Write data

# %%
del adata_ms.uns["mixscape"]
adata_ms.write("../../datasets/Adamson-2016.h5ad", compression="gzip")
