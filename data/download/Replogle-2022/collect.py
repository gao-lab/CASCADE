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
from kneed import KneeLocator
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib_venn import venn2
from pandas.errors import PerformanceWarning
from scperturb import edist_to_control, equal_subsampling
from sklearn.decomposition import TruncatedSVD

from cascade.data import aggregate_obs, filter_unobserved_targets, get_all_targets
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
# # RPE1

# %% [markdown]
# ## Read data

# %%
rpe1 = sc.read_h5ad("ReplogleWeissman2022_rpe1.h5ad")
rpe1


# %% [markdown]
# ## Rectify meta data

# %% [markdown]
# ### Rectify gene names


# %%
def map_name(x):
    author_name = rpe1.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


rpe1.var = rpe1.var.reset_index().set_index("ensembl_id", drop=False)
rpe1.var_names = rpe1.var_names.map(map_name)
rpe1.var_names.name = None
rpe1

# %% [markdown]
# ### Rectify perturbation labels

# %%
remap = {"control": "", **hgnc_map}
rpe1.obs["knockdown"] = rpe1.obs["perturbation"].map(lambda x: remap.get(x, x))

# %% [markdown]
# ### Remove unmeasured perturbations

# %%
sc.pp.filter_genes(rpe1, min_counts=1)

# %%
perturbed = get_all_targets(rpe1, "knockdown")
_ = venn2([set(rpe1.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"])

# %%
rpe1 = filter_unobserved_targets(rpe1, "knockdown")
sc.pp.filter_genes(rpe1, min_counts=1)
perturbed = get_all_targets(rpe1, "knockdown")
assert not perturbed - set(rpe1.var_names)
rpe1

# %% [markdown]
# ## Normalization and highly-variable gene selection

# %%
rpe1.layers["counts"] = rpe1.X.copy()

# %%
sc.pp.normalize_total(rpe1, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(rpe1)

# %%
sc.pp.highly_variable_genes(rpe1, layer="counts", n_top_genes=5000, flavor="seurat_v3")

# %% [markdown]
# ## Mixscape

# %% [markdown]
# ### Compute perturbation signature

# %%
sc.pp.pca(rpe1, mask_var=(rpe1.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(rpe1, metric="cosine")
sc.tl.umap(rpe1)

# %%
sc.pl.umap(rpe1, color=["percent_mito", "percent_ribo", "nperts"])

# %%
ms = pt.tl.Mixscape()
ms.perturbation_signature(rpe1, pert_key="knockdown", control="")

# %%
sc.pp.pca(
    rpe1, layer="X_pert", mask_var=(rpe1.var["highly_variable_rank"] < 2000).to_numpy()
)
sc.pp.neighbors(rpe1, metric="cosine")
sc.tl.umap(rpe1)

# %%
sc.pl.umap(rpe1, color=["percent_mito", "percent_ribo", "nperts"])

# %% [markdown]
# ### Identify non-perturbed cells

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=rpe1, control="", labels="knockdown", layer="X_pert")

# %%
ms_head = rpe1.obs.loc[:, ["knockdown", "mixscape_class_global"]].copy()
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
# ## Remove non-perturbed cells and rare perturbations

# %%
rpe1_ms = rpe1[rpe1.obs["mixscape_class_global"] != "NP"].copy()
rpe1_ms

# %%
rpe1_ms.obs = rpe1_ms.obs.join(
    rpe1_ms.obs["knockdown"].value_counts().rename("population"), on="knockdown"
)

# %%
rpe1_ms = rpe1_ms[rpe1_ms.obs["population"] >= 10]
sc.pp.filter_genes(rpe1_ms, min_counts=1)
perturbed = get_all_targets(rpe1_ms, "knockdown")
assert not perturbed - set(rpe1_ms.var_names)
rpe1_ms

# %%
rpe1_ms.var["perturbed"] = rpe1_ms.var_names.isin(perturbed)
rpe1_ms.var["perturbed"].sum()

# %%
rpe1_ms.var_names_make_unique(join=".")

# %% [markdown]
# ## E-distance

# %%
np.random.seed(42)
rpe1_ms_eq = equal_subsampling(rpe1_ms, "knockdown", N_min=100)
sc.pp.highly_variable_genes(
    rpe1_ms_eq, n_top_genes=2000, flavor="seurat_v3", layer="counts"
)
sc.pp.pca(rpe1_ms_eq, use_highly_variable=True)
sc.pp.neighbors(rpe1_ms_eq)

# %%
edist = (
    edist_to_control(rpe1_ms_eq, obs_key="knockdown", control="", n_jobs=1)
    .rename(columns={"distance": "edist"})
    .sort_values("edist", ascending=False)
)
edist.index = edist.index.reorder_categories(edist.index.to_numpy())
edist.head()

# %%
fig, ax = plt.subplots(figsize=(4, 8))
so.Plot(edist, x="edist", y="knockdown").add(so.Line()).on(ax).plot()
ax.axhline(y=np.where(edist.index == "")[0][0], c="darkred", ls="--")
ax.yaxis.set_major_locator(MaxNLocator(nbins=100))
ax.tick_params(axis="y", labelsize=5)

# %%
sc.pl.umap(
    rpe1_ms,
    color="knockdown",
    groups=edist.index[-80:].tolist(),
    palette="tab20",
    size=20,
)

# %%
sc.pl.umap(
    rpe1_ms,
    color="knockdown",
    groups=edist.index[:80].tolist(),
    palette="tab20",
    size=20,
)

# %%
rpe1_ms.obs = rpe1_ms.obs.join(edist, on="knockdown")

# %%
so.Plot(
    rpe1_ms.obs.loc[:, ["edist", "population"]].drop_duplicates(),
    x="edist",
    y="population",
).add(so.Dot(pointsize=3)).scale(y="log")

# %% [markdown]
# ## Differential expression

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(
        rpe1_ms,
        "knockdown",
        groups=edist.index[:1000].to_list(),
        reference="",
        method="t-test",
        rankby_abs=True,
    )

# %% [markdown]
# ## Perturbation space

# %%
ps = pt.tl.PseudobulkSpace()
rpe1_ps = ps.compute(
    rpe1_ms,
    target_col="knockdown",
    mode="mean",
    min_cells=0,
    min_counts=0,
)

# %%
sc.pp.neighbors(rpe1_ps)
sc.tl.umap(rpe1_ps)

# %%
sc.pl.umap(rpe1_ps)

# %% [markdown]
# ## Covariates

# %% [markdown]
# ### Total counts

# %%
log_ncounts = np.log1p(rpe1_ms.obs[["ncounts"]]).to_numpy()

# %%
log_ncounts = (log_ncounts - log_ncounts.mean()) / log_ncounts.std()
_ = sns.histplot(log_ncounts)

# %% [markdown]
# ### Is perturbed

# %%
is_perturbed = (rpe1_ms.obs[["knockdown"]] != "").to_numpy()

# %%
_ = sns.histplot(is_perturbed)

# %% [markdown]
# ### Batch

# %%
rpe1_batch = aggregate_obs(
    rpe1_ms[:, rpe1_ms.var["highly_variable_rank"] < 2000],
    "batch",
    X_agg="mean",
)

# %%
rpe1_batch = rpe1_batch.to_df()
rpe1_batch = (rpe1_batch - rpe1_batch.mean()) / rpe1_batch.std()

# %%
_ = sns.clustermap(rpe1_batch, cmap="bwr", vmin=-3, vmax=3, figsize=(8, 8))

# %%
n_comps = min(rpe1_batch.shape) - 1
truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
svd = truncated_svd.fit_transform(rpe1_batch)

# %%
argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
svd = svd[:, argsort]
knee = KneeLocator(
    np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
)
ax = sns.lineplot(x=np.arange(n_comps), y=exp_var_ratio)
ax.axvline(x=knee.knee, c="darkred", ls="--")
knee.knee, exp_var_ratio[: knee.knee + 1].sum()

# %%
svd = pd.DataFrame(svd[:, : knee.knee + 1], index=rpe1_batch.index)
svd = svd / svd.iloc[:, 0].std()
_ = svd.hist(figsize=(8, 8))

# %%
svd = svd.reindex(rpe1_ms.obs["batch"].astype(str)).to_numpy()

# %% [markdown]
# ### Combine

# %%
rpe1_ms.obsm["covariate"] = np.concatenate([log_ncounts, is_perturbed, svd], axis=1)

# %% [markdown]
# ## Write data

# %%
del rpe1_ms.uns["mixscape"]
rpe1_ms.write("../../datasets/Replogle-2022-RPE1.h5ad", compression="gzip")

# %% [markdown]
# # K562 essential

# %% [markdown]
# ## Read data

# %%
k562_ess = sc.read_h5ad("ReplogleWeissman2022_K562_essential.h5ad")
k562_ess


# %% [markdown]
# ## Rectify meta data

# %% [markdown]
# ### Rectify gene names


# %%
def map_name(x):
    author_name = k562_ess.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


k562_ess.var = k562_ess.var.reset_index().set_index("ensembl_id", drop=False)
k562_ess.var_names = k562_ess.var_names.map(map_name)
k562_ess.var_names.name = None
k562_ess

# %% [markdown]
# ### Rectify perturbation labels

# %%
remap = {"control": "", **hgnc_map}
k562_ess.obs["knockdown"] = k562_ess.obs["perturbation"].map(lambda x: remap.get(x, x))

# %% [markdown]
# ### Remove unmeasured perturbations

# %%
sc.pp.filter_genes(k562_ess, min_counts=1)

# %%
perturbed = get_all_targets(k562_ess, "knockdown")
_ = venn2(
    [set(k562_ess.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"]
)

# %%
k562_ess = filter_unobserved_targets(k562_ess, "knockdown")
sc.pp.filter_genes(k562_ess, min_counts=1)
perturbed = get_all_targets(k562_ess, "knockdown")
assert not perturbed - set(k562_ess.var_names)
k562_ess

# %% [markdown]
# ## Normalization and highly-variable gene selection

# %%
k562_ess.layers["counts"] = k562_ess.X.copy()

# %%
sc.pp.normalize_total(k562_ess, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(k562_ess)

# %%
sc.pp.highly_variable_genes(
    k562_ess, layer="counts", n_top_genes=5000, flavor="seurat_v3"
)

# %% [markdown]
# ## Mixscape

# %% [markdown]
# ### Compute perturbation signature

# %%
sc.pp.pca(k562_ess, mask_var=(k562_ess.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(k562_ess, metric="cosine")
sc.tl.umap(k562_ess)

# %%
sc.pl.umap(k562_ess, color=["percent_mito", "percent_ribo", "nperts"])

# %%
ms = pt.tl.Mixscape()
ms.perturbation_signature(k562_ess, pert_key="knockdown", control="")

# %%
sc.pp.pca(
    k562_ess,
    layer="X_pert",
    mask_var=(k562_ess.var["highly_variable_rank"] < 2000).to_numpy(),
)
sc.pp.neighbors(k562_ess, metric="cosine")
sc.tl.umap(k562_ess)

# %%
sc.pl.umap(k562_ess, color=["percent_mito", "percent_ribo", "nperts"])

# %% [markdown]
# ### Identify non-perturbed cells

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=k562_ess, control="", labels="knockdown", layer="X_pert")

# %%
ms_head = k562_ess.obs.loc[:, ["knockdown", "mixscape_class_global"]].copy()
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
# ## Remove non-perturbed cells and rare perturbations

# %%
k562_ess_ms = k562_ess[k562_ess.obs["mixscape_class_global"] != "NP"].copy()
k562_ess_ms

# %%
k562_ess_ms.obs = k562_ess_ms.obs.join(
    k562_ess_ms.obs["knockdown"].value_counts().rename("population"), on="knockdown"
)

# %%
k562_ess_ms = k562_ess_ms[k562_ess_ms.obs["population"] >= 10]
sc.pp.filter_genes(k562_ess_ms, min_counts=1)
perturbed = get_all_targets(k562_ess_ms, "knockdown")
assert not perturbed - set(k562_ess_ms.var_names)
k562_ess_ms

# %%
k562_ess_ms.var["perturbed"] = k562_ess_ms.var_names.isin(perturbed)
k562_ess_ms.var["perturbed"].sum()

# %%
k562_ess_ms.var_names_make_unique(join=".")

# %% [markdown]
# ## E-distance

# %%
np.random.seed(42)
k562_ess_ms_eq = equal_subsampling(k562_ess_ms, "knockdown", N_min=100)
sc.pp.highly_variable_genes(
    k562_ess_ms_eq, n_top_genes=2000, flavor="seurat_v3", layer="counts"
)
sc.pp.pca(k562_ess_ms_eq, use_highly_variable=True)
sc.pp.neighbors(k562_ess_ms_eq)

# %%
edist = (
    edist_to_control(k562_ess_ms_eq, obs_key="knockdown", control="", n_jobs=1)
    .rename(columns={"distance": "edist"})
    .sort_values("edist", ascending=False)
)
edist.index = edist.index.reorder_categories(edist.index.to_numpy())
edist.head()

# %%
fig, ax = plt.subplots(figsize=(4, 8))
so.Plot(edist, x="edist", y="knockdown").add(so.Line()).on(ax).plot()
ax.axhline(y=np.where(edist.index == "")[0][0], c="darkred", ls="--")
ax.yaxis.set_major_locator(MaxNLocator(nbins=100))
ax.tick_params(axis="y", labelsize=5)

# %%
sc.pl.umap(
    k562_ess_ms,
    color="knockdown",
    groups=edist.index[-80:].tolist(),
    palette="tab20",
    size=20,
)

# %%
sc.pl.umap(
    k562_ess_ms,
    color="knockdown",
    groups=edist.index[:80].tolist(),
    palette="tab20",
    size=20,
)

# %%
k562_ess_ms.obs = k562_ess_ms.obs.join(edist, on="knockdown")

# %%
so.Plot(
    k562_ess_ms.obs.loc[:, ["edist", "population"]].drop_duplicates(),
    x="edist",
    y="population",
).add(so.Dot(pointsize=3)).scale(y="log")

# %% [markdown]
# ## Differential expression

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(
        k562_ess_ms,
        "knockdown",
        groups=edist.index[:1000].to_list(),
        reference="",
        method="t-test",
        rankby_abs=True,
    )

# %% [markdown]
# ## Perturbation space

# %%
ps = pt.tl.PseudobulkSpace()
k562_ess_ps = ps.compute(
    k562_ess_ms,
    target_col="knockdown",
    mode="mean",
    min_cells=0,
    min_counts=0,
)

# %%
sc.pp.neighbors(k562_ess_ps)
sc.tl.umap(k562_ess_ps)

# %%
sc.pl.umap(k562_ess_ps)

# %% [markdown]
# ## Covariates

# %% [markdown]
# ### Total counts

# %%
log_ncounts = np.log1p(k562_ess_ms.obs[["ncounts"]]).to_numpy()

# %%
log_ncounts = (log_ncounts - log_ncounts.mean()) / log_ncounts.std()
_ = sns.histplot(log_ncounts)

# %% [markdown]
# ### Is perturbed

# %%
is_perturbed = (k562_ess_ms.obs[["knockdown"]] != "").to_numpy()

# %%
_ = sns.histplot(is_perturbed)

# %% [markdown]
# ### Batch

# %%
k562_ess_batch = aggregate_obs(
    k562_ess_ms[:, k562_ess_ms.var["highly_variable_rank"] < 2000],
    "batch",
    X_agg="mean",
)

# %%
k562_ess_batch = k562_ess_batch.to_df()
k562_ess_batch = (k562_ess_batch - k562_ess_batch.mean()) / k562_ess_batch.std()

# %%
_ = sns.clustermap(k562_ess_batch, cmap="bwr", vmin=-3, vmax=3, figsize=(8, 8))

# %%
n_comps = min(k562_ess_batch.shape) - 1
truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
svd = truncated_svd.fit_transform(k562_ess_batch)

# %%
argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
svd = svd[:, argsort]
knee = KneeLocator(
    np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
)
ax = sns.lineplot(x=np.arange(n_comps), y=exp_var_ratio)
ax.axvline(x=knee.knee, c="darkred", ls="--")
knee.knee, exp_var_ratio[: knee.knee + 1].sum()

# %%
svd = pd.DataFrame(svd[:, : knee.knee + 1], index=k562_ess_batch.index)
svd = svd / svd.iloc[:, 0].std()
_ = svd.hist(figsize=(8, 8))

# %%
svd = svd.reindex(k562_ess_ms.obs["batch"].astype(str)).to_numpy()

# %% [markdown]
# ### Combine

# %%
k562_ess_ms.obsm["covariate"] = np.concatenate([log_ncounts, is_perturbed, svd], axis=1)

# %% [markdown]
# ## Write data

# %%
del k562_ess_ms.uns["mixscape"]
k562_ess_ms.write("../../datasets/Replogle-2022-K562-ess.h5ad", compression="gzip")

# %% [markdown]
# # K562 gwps

# %% [markdown]
# ## Read data

# %%
k562_gwps = sc.read_h5ad("ReplogleWeissman2022_K562_gwps.h5ad")
k562_gwps


# %% [markdown]
# ## Rectify meta data

# %% [markdown]
# ### Rectify gene names


# %%
def map_name(x):
    author_name = k562_gwps.var.loc[x, "gene_name"]
    return hgnc_map.get(x, hgnc_map.get(author_name, author_name))


k562_gwps.var = k562_gwps.var.reset_index().set_index("ensembl_id", drop=False)
k562_gwps.var_names = k562_gwps.var_names.map(map_name)
k562_gwps.var_names.name = None
k562_gwps

# %% [markdown]
# ### Rectify perturbation labels

# %%
remap = {"control": "", **hgnc_map}
k562_gwps.obs["knockdown"] = k562_gwps.obs["perturbation"].map(
    lambda x: remap.get(x, x)
)

# %% [markdown]
# ### Remove unmeasured perturbations

# %%
sc.pp.filter_genes(k562_gwps, min_counts=1)

# %%
perturbed = get_all_targets(k562_gwps, "knockdown")
_ = venn2(
    [set(k562_gwps.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"]
)

# %%
k562_gwps = filter_unobserved_targets(k562_gwps, "knockdown")
sc.pp.filter_genes(k562_gwps, min_counts=1)
perturbed = get_all_targets(k562_gwps, "knockdown")
assert not perturbed - set(k562_gwps.var_names)
k562_gwps

# %% [markdown]
# ## Normalization and highly-variable gene selection

# %%
k562_gwps.layers["counts"] = k562_gwps.X.copy()

# %%
sc.pp.normalize_total(k562_gwps, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(k562_gwps)

# %%
sc.pp.highly_variable_genes(
    k562_gwps, layer="counts", n_top_genes=5000, flavor="seurat_v3"
)

# %% [markdown]
# ## Mixscape

# %% [markdown]
# ### Compute perturbation signature

# %%
sc.pp.pca(k562_gwps, mask_var=(k562_gwps.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(k562_gwps, metric="cosine")
sc.tl.umap(k562_gwps)

# %%
sc.pl.umap(k562_gwps, color=["percent_mito", "percent_ribo", "nperts"])

# %%
ms = pt.tl.Mixscape()
ms.perturbation_signature(k562_gwps, pert_key="knockdown", control="")

# %%
sc.pp.pca(
    k562_gwps,
    layer="X_pert",
    mask_var=(k562_gwps.var["highly_variable_rank"] < 2000).to_numpy(),
)
sc.pp.neighbors(k562_gwps, metric="cosine")
sc.tl.umap(k562_gwps)

# %%
sc.pl.umap(k562_gwps, color=["percent_mito", "percent_ribo", "nperts"])

# %% [markdown]
# ### Identify non-perturbed cells

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=k562_gwps, control="", labels="knockdown", layer="X_pert")

# %%
ms_head = k562_gwps.obs.loc[:, ["knockdown", "mixscape_class_global"]].copy()
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
# ## Remove non-perturbed cells and rare perturbations

# %%
k562_gwps_ms = k562_gwps[k562_gwps.obs["mixscape_class_global"] != "NP"].copy()
k562_gwps_ms

# %%
k562_gwps_ms.obs = k562_gwps_ms.obs.join(
    k562_gwps_ms.obs["knockdown"].value_counts().rename("population"), on="knockdown"
)

# %%
k562_gwps_ms = k562_gwps_ms[k562_gwps_ms.obs["population"] >= 10]
sc.pp.filter_genes(k562_gwps_ms, min_counts=1)
perturbed = get_all_targets(k562_gwps_ms, "knockdown")
assert not perturbed - set(k562_gwps_ms.var_names)
k562_gwps_ms

# %%
k562_gwps_ms.var["perturbed"] = k562_gwps_ms.var_names.isin(perturbed)
k562_gwps_ms.var["perturbed"].sum()

# %%
k562_gwps_ms.var_names_make_unique(join=".")

# %% [markdown]
# ## E-distance

# %%
np.random.seed(42)
k562_gwps_ms_eq = equal_subsampling(k562_gwps_ms, "knockdown", N_min=100)
sc.pp.highly_variable_genes(
    k562_gwps_ms_eq, n_top_genes=2000, flavor="seurat_v3", layer="counts"
)
sc.pp.pca(k562_gwps_ms_eq, use_highly_variable=True)
sc.pp.neighbors(k562_gwps_ms_eq)

# %%
edist = (
    edist_to_control(k562_gwps_ms_eq, obs_key="knockdown", control="", n_jobs=1)
    .rename(columns={"distance": "edist"})
    .sort_values("edist", ascending=False)
)
edist.index = edist.index.astype("category").reorder_categories(edist.index.to_numpy())
edist.head()

# %%
fig, ax = plt.subplots(figsize=(4, 8))
so.Plot(edist, x="edist", y="knockdown").add(so.Line()).on(ax).plot()
ax.axhline(y=np.where(edist.index == "")[0][0], c="darkred", ls="--")
ax.yaxis.set_major_locator(MaxNLocator(nbins=100))
ax.tick_params(axis="y", labelsize=5)

# %%
sc.pl.umap(
    k562_gwps_ms,
    color="knockdown",
    groups=edist.index[-80:].tolist(),
    palette="tab20",
    size=20,
)

# %%
sc.pl.umap(
    k562_gwps_ms,
    color="knockdown",
    groups=edist.index[:80].tolist(),
    palette="tab20",
    size=20,
)

# %%
k562_gwps_ms.obs = k562_gwps_ms.obs.join(edist, on="knockdown")

# %%
so.Plot(
    k562_gwps_ms.obs.loc[:, ["edist", "population"]].drop_duplicates(),
    x="edist",
    y="population",
).add(so.Dot(pointsize=3)).scale(y="log")

# %% [markdown]
# ## Differential expression

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(
        k562_gwps_ms,
        "knockdown",
        groups=edist.index[:1000].to_list(),
        reference="",
        method="t-test",
        rankby_abs=True,
    )

# %% [markdown]
# ## Perturbation space

# %%
ps = pt.tl.PseudobulkSpace()
k562_gwps_ps = ps.compute(
    k562_gwps_ms,
    target_col="knockdown",
    mode="mean",
    min_cells=0,
    min_counts=0,
)

# %%
sc.pp.neighbors(k562_gwps_ps)
sc.tl.umap(k562_gwps_ps)

# %%
sc.pl.umap(k562_gwps_ps)

# %% [markdown]
# ## Covariates

# %% [markdown]
# ### Total counts

# %%
log_ncounts = np.log1p(k562_gwps_ms.obs[["ncounts"]]).to_numpy()

# %%
log_ncounts = (log_ncounts - log_ncounts.mean()) / log_ncounts.std()
_ = sns.histplot(log_ncounts)

# %% [markdown]
# ### Is perturbed

# %%
is_perturbed = (k562_gwps_ms.obs[["knockdown"]] != "").to_numpy()

# %%
_ = sns.histplot(is_perturbed)

# %% [markdown]
# ### Batch

# %%
k562_gwps_batch = aggregate_obs(
    k562_gwps_ms[:, k562_gwps_ms.var["highly_variable_rank"] < 2000],
    "batch",
    X_agg="mean",
)

# %%
k562_gwps_batch = k562_gwps_batch.to_df()
k562_gwps_batch = (k562_gwps_batch - k562_gwps_batch.mean()) / k562_gwps_batch.std()

# %%
_ = sns.clustermap(k562_gwps_batch, cmap="bwr", vmin=-3, vmax=3, figsize=(8, 8))

# %%
n_comps = min(k562_gwps_batch.shape) - 1
truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
svd = truncated_svd.fit_transform(k562_gwps_batch)

# %%
argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
svd = svd[:, argsort]
knee = KneeLocator(
    np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
)
ax = sns.lineplot(x=np.arange(n_comps), y=exp_var_ratio)
ax.axvline(x=knee.knee, c="darkred", ls="--")
knee.knee, exp_var_ratio[: knee.knee + 1].sum()

# %%
svd = pd.DataFrame(svd[:, : knee.knee + 1], index=k562_gwps_batch.index)
svd = svd / svd.iloc[:, 0].std()
_ = svd.hist(figsize=(8, 8))

# %%
svd = svd.reindex(k562_gwps_ms.obs["batch"].astype(str)).to_numpy()

# %% [markdown]
# ### Combine

# %%
k562_gwps_ms.obsm["covariate"] = np.concatenate(
    [log_ncounts, is_perturbed, svd], axis=1
)

# %% [markdown]
# ## Write data

# %%
del k562_gwps_ms.uns["mixscape"]
k562_gwps_ms.write("../../datasets/Replogle-2022-K562-gwps.h5ad", compression="gzip")
