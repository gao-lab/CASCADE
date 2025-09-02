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

import anndata as ad
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc
import seaborn as sns
import seaborn.objects as so
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib_venn import venn2
from pandas.errors import PerformanceWarning
from scipy.sparse import csr_matrix
from scperturb import edist, edist_to_control, equal_subsampling
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from cascade.data import filter_unobserved_targets, get_all_targets
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
# # Check TFmap

# %%
tforf = pd.read_excel("1-s2.0-S0092867422014702-mmc2.xlsx", sheet_name="Table S1A")
barcode2name = {b: n.strip() for b, n in zip(tforf["Barcode Sequence"], tforf["Name"])}
barcode2gene = {
    b: g for b, g in zip(tforf["Barcode Sequence"], tforf["RefSeq Gene Name"])
}
barcode2isoform = {b: g for b, g in zip(tforf["Barcode Sequence"], tforf["Isoform ID"])}
name2barcode = {n: b for b, n in barcode2name.items()}

# %%
TFmaps = []
suffix_map = {
    "S01": ",P1.22-0-0",
    "S02": ",P1.30-0-0",
    "S03": ",P1.38-0-0",
    "S04": ",P1.46-0-0",
    "S05": ",P1.54-0-1-0",
    "S06": ",P1.62-1-1-0",
    "S07": ",P1.07-2-1-0",
    "S08": ",P1.15-3-1-0",
    "S09": ",P1.23-0-2-0",
    "S10": ",P1.31-1-2-0",
    "S11": ",P1.39-2-2-0",
    "S12": ",P1.47-3-2-0",
    "S13": ",P1.55-0-3-0",
    "S14": ",P1.63-1-3-0",
    "S15": ",P1.08-2-3-0",
    "S16": ",P1.16-3-3-0",
    "S17": ",P1.24-0-4-0",
    "S18": ",P1.32-1-4-0",
    "S19": ",P1.40-2-4-0",
    "S20": ",P1.48-3-4-0",
    "S21": "-0-1",  # P1.22
    "S22": "-1-1",  # P1.30
    "S23": "-2-1",  # P1.38
    "S24": "-3-1",  # P1.62
}
for s in tqdm(range(1, 21)):
    TFmap = pd.read_csv(
        f"GSM67199{69+s}_210322_TFmap_S{s:02d}.csv.gz",
        names=["cell_barcode", "TF_barcode", "TF_count"],
    )
    TFmap["cell_barcode"] += suffix_map[f"S{s:02d}"]
    TFmap["TFORF"] = TFmap["TF_barcode"].map(barcode2name)
    TFmap["TF_gene"] = TFmap["TF_barcode"].map(barcode2gene)
    TFmap["TF_isoform"] = TFmap["TF_barcode"].map(barcode2isoform)
    TFmaps.append(TFmap)
for s in tqdm(range(21, 25)):
    TFmap = pd.read_csv(
        f"210322_TFmap_S{s:02d}_v2.csv.gz",
        names=["cell_barcode", "TFORF", "TF_count"],
    )
    TFmap["cell_barcode"] += suffix_map[f"S{s:02d}"]
    TFmap["TFORF"] = TFmap["TFORF"].str.split(r"(?<=TFORF\d{4})-", expand=True)[0]
    TFmap["TF_barcode"] = TFmap["TFORF"].map(name2barcode)
    TFmap["TF_gene"] = TFmap["TF_barcode"].map(barcode2gene)
    TFmap["TF_isoform"] = TFmap["TF_barcode"].map(barcode2isoform)
    TFmaps.append(TFmap)
TFmaps = pd.concat(TFmaps, ignore_index=True).set_index("cell_barcode")
TFmaps

# %% [markdown]
# # TF atlas full

# %%
adata = ad.read_h5ad("GSE217460_210322_TFAtlas.h5ad")
adata.X = csr_matrix(adata.X, dtype=np.float32).expm1()
normalizers = np.vectorize(min)(adata.X.tolil().data)
adata.X = adata.X.multiply(1 / normalizers[:, np.newaxis])
adata.X.data = np.round(adata.X.data, 0)
adata.X = adata.X.tocsr()
adata

# %% [markdown]
# ## Rectify meta data

# %% [markdown]
# ### Rectify gene names

# %%
adata.var_names = adata.var_names.map(lambda x: hgnc_map.get(x, x))

# %% [markdown]
# ### Rectify perturbation labels

# %%
remap = {"GFP": "", "mCherry": "", **hgnc_map}
adata.obs = adata.obs.join(TFmaps)
adata.obs["knockup"] = adata.obs["TF_gene"].map(lambda x: remap.get(x, x))

# %% [markdown]
# ### Remove unmeasured perturbations

# %%
sc.pp.filter_genes(adata, min_counts=1)

# %%
perturbed = get_all_targets(adata, "knockup")
_ = venn2([set(adata.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"])

# %%
adata = filter_unobserved_targets(adata, "knockup")
sc.pp.filter_genes(adata, min_counts=1)
perturbed = get_all_targets(adata, "knockup")
assert not perturbed - set(adata.var_names)
adata

# %% [markdown]
# ## Normalization and highly-variable gene selection

# %%
adata.layers["counts"] = adata.X.copy()

# %%
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(
    adata, layer="counts", n_top_genes=5000, flavor="seurat_v3", batch_key="batch"
)

# %% [markdown]
# ## Mixscape

# %% [markdown]
# ### Compute perturbation signature

# %%
sc.pp.pca(adata, mask_var=(adata.var["highly_variable_rank"] < 2000).to_numpy())
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color=["n_counts", "louvain", "batch"])

# %%
adata.write("../../datasets/Joung-2023.h5ad", compression="gzip")

# %% [markdown]
# # TF atlas subsample

# %%
adata = ad.read_h5ad("GSE217460_210322_TFAtlas_subsample_raw.h5ad")
adata.X = csr_matrix(adata.X, dtype=np.float32)
adata

# %% [markdown]
# ## Rectify meta data

# %% [markdown]
# ### Rectify gene names

# %%
adata.var_names = adata.var_names.map(lambda x: hgnc_map.get(x, x))

# %% [markdown]
# ### Rectify perturbation labels

# %%
remap = {"GFP": "", "mCherry": "", **hgnc_map}
adata.obs = adata.obs.join(TFmaps)
adata.obs["knockup"] = adata.obs["TF_gene"].map(lambda x: remap.get(x, x))

# %% [markdown]
# ### Remove unmeasured perturbations

# %%
sc.pp.filter_genes(adata, min_counts=1)

# %%
perturbed = get_all_targets(adata, "knockup")
_ = venn2([set(adata.var_names), set(perturbed)], set_labels=["Measured", "Perturbed"])

# %%
adata = filter_unobserved_targets(adata, "knockup")
sc.pp.filter_genes(adata, min_counts=1)
perturbed = get_all_targets(adata, "knockup")
assert not perturbed - set(adata.var_names)
adata

# %% [markdown]
# ## Standard preprocessing

# %%
adata.layers["counts"] = adata.X.copy()

# %%
sc.pp.normalize_total(adata, target_sum=1e4, key_added="ncounts")
sc.pp.log1p(adata)

# %%
sc.pp.highly_variable_genes(
    adata, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key="batch"
)

# %%
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
sc.pp.regress_out(adata_hvg, ["n_counts", "percent_mito"])
sc.pp.scale(adata_hvg)

# %%
sc.tl.pca(adata_hvg, n_comps=50)
sc.pp.neighbors(adata_hvg, n_neighbors=20)
sc.tl.umap(adata_hvg)

# %%
sc.pl.umap(adata_hvg, color=["n_counts", "louvain", "batch"])

# %%
adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
adata.obsp["distances"] = adata_hvg.obsp["distances"]
adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]

# %% [markdown]
# ## Mixscape

# %% [markdown]
# Only batch 1 involves control, which is a problem for mixscape. Here we will use
# batch 1 to identify a set of weak perturbations, and use these perturbations as
# the "pseudo control" for batch 0.

# %% [markdown]
# ### Use batch 1 to identify weak perturbations

# %% [markdown]
# Control cells in differentiated clusters are discarded.

# %%
adata_b1 = adata[
    (adata.obs["batch"] == "1")
    & ((adata.obs["knockup"] != "") | ~adata.obs["louvain"].isin({"6", "7", "8"}))
].copy()

# %%
adata_b1.obs = adata_b1.obs.join(
    adata_b1.obs["knockup"].value_counts().rename("population"), on="knockup"
)
adata_b1 = adata_b1[adata_b1.obs["population"] >= 3].copy()
adata_b1

# %%
np.random.seed(42)
adata_b1_eq = equal_subsampling(adata_b1, "knockup", N_min=100)
estat = (
    edist_to_control(adata_b1_eq, obs_key="knockup", control="", n_jobs=1)
    .rename(columns={"distance": "edist"})
    .sort_values("edist", ascending=False)
)
estat.index = estat.index.reorder_categories(estat.index.to_numpy())
estat.head()

# %%
fig, ax = plt.subplots(figsize=(4, 8))
so.Plot(estat, x="edist", y="knockup").add(so.Line()).on(ax).plot()
ax.axhline(y=np.where(estat.index == "")[0][0], c="darkred", ls="--")
ax.yaxis.set_major_locator(MaxNLocator(nbins=100))
ax.tick_params(axis="y", labelsize=5)

# %%
weak_tfs = estat.loc[estat["edist"].abs() < 0.5]
weak_tfs

# %%
weak_tfs.shape[0]

# %% [markdown]
# ### Treat weak perturbations as ctrl in batch 0

# %%
adata_b0 = adata[adata.obs["batch"] == "0"].copy()

# %%
adata_b0.obs = adata_b0.obs.join(
    adata_b0.obs["knockup"].value_counts().rename("population"), on="knockup"
)
adata_b0 = adata_b0[adata_b0.obs["population"] >= 3].copy()
adata_b0

# %%
np.random.seed(42)
adata_b0_eq = equal_subsampling(adata_b0, "knockup", N_min=100)
adata_b0_eq_weak = adata_b0_eq[adata_b0_eq.obs["knockup"].isin(weak_tfs.index)].copy()
adata_b0_eq_weak.obs["knockup"].value_counts()

# %%
estat = edist(adata_b0_eq_weak, obs_key="knockup", n_jobs=1)
sns.clustermap(estat, cmap="bwr", center=0, figsize=(5, 5))

# %%
adata_b0.obs["knockup_pseudoctrl"] = (
    adata_b0.obs["knockup"]
    .astype(str)
    .replace(
        {
            "TP63": "",
            "ZNF563": "",
            "NFATC4": "",
            "TAFAZZIN": "",
            "ZNF394": "",
        }
    )
    .astype("category")
)
(adata_b0.obs["knockup_pseudoctrl"] == "").sum()

# %%
adata_b0 = adata_b0[
    (adata_b0.obs["knockup_pseudoctrl"] != "")
    | ~adata_b0.obs["louvain"].isin({"6", "7", "8"})
].copy()

# %% [markdown]
# ### Compute perturbation signature for batch 1

# %%
ms = pt.tl.Mixscape()
adata_b1.X = adata_b1.X.toarray()  # Sparse matrix triggers a bug
ms.perturbation_signature(adata_b1, pert_key="knockup", control="")
adata_b1.X = csr_matrix(adata_b1.X)

# %%
sc.pp.pca(adata_b1, n_comps=50, layer="X_pert", mask_var="highly_variable")
sc.pp.neighbors(adata_b1, n_neighbors=20)
sc.tl.umap(adata_b1)

# %%
sc.pl.umap(adata_b1, color=["n_counts", "louvain"])

# %% [markdown]
# ### Identify non-perturbed cells for batch 1

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=adata_b1, control="", labels="knockup", layer="X_pert")

# %%
ms_head = adata_b1.obs.loc[:, ["knockup", "mixscape_class_global"]].copy()
ms_head["knockup"] = ms_head["knockup"].cat.reorder_categories(
    ms_head["knockup"].value_counts().index.to_numpy()
)
ms_head = ms_head.loc[(ms_head["knockup"].cat.codes < 30) & (ms_head["knockup"] != "")]
ms_head["knockup"] = ms_head["knockup"].cat.remove_unused_categories()

# %%
so.Plot(ms_head, y="knockup", color="mixscape_class_global").add(
    so.Bar(), so.Count(), so.Stack()
).layout(size=(4, 6))

# %% [markdown]
# ### Remove non-perturbed cells and rare perturbations for batch 1

# %% [markdown]
# Cells in differentiated clusters are always kept regardless of the mixscape call.

# %%
adata_b1_ms = adata_b1[
    (adata_b1.obs["mixscape_class_global"] != "NP")
    | adata_b1.obs["louvain"].isin({"6", "7", "8"})
].copy()
adata_b1_ms

# %% [markdown]
# ### Compute perturbation signature for batch 0

# %%
ms = pt.tl.Mixscape()
adata_b0.X = adata_b0.X.toarray()  # Sparse matrix triggers a bug
ms.perturbation_signature(adata_b0, pert_key="knockup_pseudoctrl", control="")
adata_b0.X = csr_matrix(adata_b0.X)

# %%
sc.pp.pca(adata_b0, n_comps=50, layer="X_pert", mask_var="highly_variable")
sc.pp.neighbors(adata_b0, n_neighbors=20)
sc.tl.umap(adata_b0)

# %%
sc.pl.umap(adata_b0, color=["n_counts", "louvain"])

# %% [markdown]
# ### Identify non-perturbed cells for batch 0

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.simplefilter("ignore", category=PerformanceWarning)
    ms.mixscape(adata=adata_b0, control="", labels="knockup_pseudoctrl", layer="X_pert")

# %%
ms_head = adata_b0.obs.loc[:, ["knockup_pseudoctrl", "mixscape_class_global"]].copy()
ms_head["knockup_pseudoctrl"] = ms_head["knockup_pseudoctrl"].cat.reorder_categories(
    ms_head["knockup_pseudoctrl"].value_counts().index.to_numpy()
)
ms_head = ms_head.loc[
    (ms_head["knockup_pseudoctrl"].cat.codes < 30)
    & (ms_head["knockup_pseudoctrl"] != "")
]
ms_head["knockup_pseudoctrl"] = ms_head[
    "knockup_pseudoctrl"
].cat.remove_unused_categories()

# %%
so.Plot(ms_head, y="knockup_pseudoctrl", color="mixscape_class_global").add(
    so.Bar(), so.Count(), so.Stack()
).layout(size=(4, 6))

# %% [markdown]
# ### Remove non-perturbed cells and rare perturbations for batch 0

# %% [markdown]
# Cells in differentiated clusters are always kept regardless of the mixscape call.

# %%
adata_b0.obs["mixscape_class_global"].value_counts()

# %%
adata_b0_ms = adata_b0[
    (adata_b0.obs["mixscape_class_global"] == "KO")
    | adata_b0.obs["louvain"].isin({"6", "7", "8"})
].copy()
adata_b0_ms

# %% [markdown]
# ### Combine batches

# %%
adata_ms = adata[
    adata.obs_names.isin(set(adata_b0_ms.obs_names) | set(adata_b1_ms.obs_names))
].copy()
adata_ms

# %%
adata_ms.obs = adata_ms.obs.join(
    adata_ms.obs["knockup"].value_counts().rename("population"), on="knockup"
)
adata_ms = adata_ms[adata_ms.obs["population"] >= 3].copy()
adata_ms

# %%
sc.pp.filter_genes(adata_ms, min_counts=1)
adata_ms = filter_unobserved_targets(adata_ms, "knockup")
sc.pp.filter_genes(adata_ms, min_counts=1)
perturbed = get_all_targets(adata_ms, "knockup")
assert not perturbed - set(adata_ms.var_names)
adata_ms

# %%
adata_ms.var["perturbed"] = adata_ms.var_names.isin(perturbed)
adata_ms.var["perturbed"].sum()

# %%
adata_ms.var_names_make_unique(join=".")

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
# ### Percent mito

# %%
pct_mito = adata_ms.obs[["percent_mito"]].to_numpy()

# %%
pct_mito = (pct_mito - pct_mito.mean()) / pct_mito.std()
_ = sns.histplot(pct_mito)

# %% [markdown]
# ### Is perturbed

# %%
is_perturbed = (adata_ms.obs[["knockup"]] != "").to_numpy()

# %%
_ = sns.histplot(is_perturbed)

# %% [markdown]
# ### Batch

# %%
batch = OneHotEncoder().fit_transform(adata_ms.obs[["batch"]]).toarray()

# %% [markdown]
# ### Combine

# %%
adata_ms.obsm["covariate"] = np.concatenate(
    [log_ncounts, pct_mito, is_perturbed, batch], axis=1
)

# %% [markdown]
# ## Write data

# %%
adata_ms.write("../../datasets/Joung-2023-subsample.h5ad", compression="gzip")

# %% [markdown]
# # TF atlas differentiated

# %%
adata = ad.read_h5ad("GSE217460_210322_TFAtlas_differentiated_raw.h5ad")
adata.X = csr_matrix(adata.X, dtype=np.float32)
adata

# %%
adata.obs = adata.obs.join(TFmaps)

# %%
TFs = set(adata.obs["TF_gene"])
TFs - set(adata.var_names)

# %%
adata.obs["knockup"] = adata.obs["TF_gene"].replace({"GFP": "", "mCherry": ""})

# %%
adata = adata[~adata.obs["knockup"].isin({"DND1", "SHOX"})].copy()
adata

# %%
adata.write("../../datasets/Joung-2023-diff.h5ad", compression="gzip")
