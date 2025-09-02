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

# %% editable=true slideshow={"slide_type": ""}
import warnings
from pathlib import Path
from shutil import rmtree

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import rcParams
from matplotlib_venn import venn3
from pandas.errors import PerformanceWarning
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from cascade.data import Targets, neighbor_impute
from cascade.graph import marginalize
from cascade.plot import set_figure_params
from cascade.utils import config

# %% editable=true slideshow={"slide_type": ""}
config.LOG_LEVEL = "DEBUG"
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Read data

# %% editable=true slideshow={"slide_type": ""}
joung = ad.read_h5ad("../../data/datasets/Joung-2023-subsample.h5ad")
joung.obs["cell_type"] = "H1"
joung


# %%
cao = ad.read_h5ad("../../data/datasets/Cao-2020.h5ad")
cao.obs["knockup"] = ""
cao

# %%
kegg = nx.read_gml("../../data/scaffold/KEGG/inferred_kegg_gene_only.gml.gz")
tf_target = nx.read_gml("../../data/scaffold/TF-target/TF-target.gml.gz")
ppi = nx.read_gml("../../data/scaffold/BioGRID/biogrid.gml.gz").to_directed()
corr = nx.read_gml("../../data/scaffold/GTEx/corr.gml.gz").to_directed()

# %% [markdown]
# # Subsample target and merge with source

# %%
rnd = np.random.RandomState(42)
subsample_idx = []
for ct, idx in cao.obs.groupby("cell_type", observed=True).indices.items():
    subsample_idx.append(rnd.choice(idx, min(5000, idx.size), replace=False))
subsample_idx = np.concatenate(subsample_idx)
cao_sub = cao[subsample_idx].copy()
cao_sub

# %%
combined = ad.concat(
    {"Joung-2023": joung, "Cao-2020": cao_sub},
    label="dataset",
    index_unique="-",
)
combined

# %% [markdown]
# # Find common DEGs across cell types as batch effect

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(combined, "cell_type", reference="H1")

# %%
ct_de = {}
for ct in cao_sub.obs["cell_type"].cat.categories:
    ct_de[ct] = sc.get.rank_genes_groups_df(combined, ct)

# %%
deg_ct_counts = pd.concat(
    de_df.query("scores > 25 | scores < -25")["names"] for de_df in ct_de.values()
).value_counts()
common_degs = deg_ct_counts[
    deg_ct_counts > cao_sub.obs["cell_type"].cat.categories.size / 2
].index
common_degs

# %%
combined.var["common_degs"] = combined.var_names.isin(common_degs)

# %% [markdown]
# # Determine gene set

# %%
joung_diff = joung[
    joung.obs["louvain"].isin({"6", "7", "8"}) | (joung.obs["knockup"] == "")
].copy()
joung_diff

# %%
sc.pp.highly_variable_genes(
    joung_diff, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key="batch"
)

# %%
var_merge = pd.merge(
    joung_diff.var,
    cao.var,
    left_index=True,
    right_index=True,
    suffixes=["_joung", "_cao"],
).join(combined.var)

# %%
_ = venn3(
    [
        set(var_merge.query("highly_variable_nbatches > 0").index),
        set(var_merge.query("highly_variable_cao").index),
        set(var_merge.query("common_degs").index),
    ],
    set_labels=["Joung HVGs", "Cao HVGs", "Common DEGs"],
)

# %%
selected_genes = set(
    var_merge.query(
        "((highly_variable_nbatches > 0 | highly_variable_cao) & "
        "~common_degs & "
        "means_joung > 0.0125) | perturbed"
    ).index
)
len(selected_genes)

# %% [markdown]
# # Remove unselected perturbations

# %%
knockups = joung.obs["knockup"].unique()
knockups = pd.Series(knockups, index=knockups).map(
    lambda x: not (Targets(x) - selected_genes)
)

# %%
joung_use = joung[knockups.loc[joung.obs["knockup"]]].copy()
joung_use

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=PerformanceWarning)
    sc.tl.rank_genes_groups(combined, "dataset", reference="Joung-2023")
ds_de = sc.get.rank_genes_groups_df(combined, "Cao-2020")

# %%
stable_genes = ds_de.query("pvals_adj > 0.05")["names"]
stable_genes.size

# %%
nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(
    joung_use[:, stable_genes].X.toarray()
)
nni = nearest_neighbors.kneighbors(
    cao_sub[:, stable_genes].X.toarray(), return_distance=False
).ravel()
cao_sub.obsm["covariate"] = joung_use.obsm["covariate"][nni]
cao_sub

# %% [markdown]
# # Covariates

# %%
joung_use.obsm["covariate"] = np.concatenate(
    [joung_use.obsm["covariate"], np.zeros((joung_use.n_obs, 1))], axis=1
)

# %%
cao_sub.obsm["covariate"][:, 2] = 1
cao_sub.obsm["covariate"][:, 3:] = 0
cao_sub.obsm["covariate"] = np.concatenate(
    [cao_sub.obsm["covariate"], np.ones((cao_sub.n_obs, 1))], axis=1
)

# %% [markdown]
# # Impute data

# %%
combined.obs.groupby("dataset", observed=True).mean(numeric_only=True)

# %%
joung_use = joung_use[:, [g in selected_genes for g in joung_use.var_names]].copy()
joung_use

# %%
joung_use.X = np.expm1(joung_use.X)
joung_use_imp = neighbor_impute(
    joung_use,
    k=20,
    use_rep="X_pca",
    use_batch="knockup",
    X_agg="mean",
    obs_agg={"ncounts": "sum"},
    obsm_agg={"covariate": "mean"},
    layers_agg={"counts": "sum"},
)
joung_use_imp.X = np.log1p(joung_use_imp.X)
joung_use.X = np.log1p(joung_use.X)

# %%
cao_sub = cao_sub[:, [g in selected_genes for g in cao_sub.var_names]].copy()
cao_sub

# %%
cao_sub.X = np.expm1(cao_sub.X)
cao_sub_imp = neighbor_impute(
    cao_sub,
    k=80,
    use_rep="X_pca",
    use_batch="cell_type",
    X_agg="mean",
    obs_agg={"ncounts": "sum"},
    obsm_agg={"covariate": "mean"},
    layers_agg={"counts": "sum"},
)
cao_sub_imp.X = np.log1p(cao_sub_imp.X)
cao_sub.X = np.log1p(cao_sub.X)

# %% [markdown]
# # Save data

# %%
adata_use_imp = ad.concat(
    {
        "Joung-2023": joung_use_imp,
        "Cao-2020": cao_sub_imp[rnd.choice(cao_sub_imp.n_obs, 50000, replace=False)],
    },
    label="dataset",
    index_unique="-",
)
adata_use_imp.write("adata.h5ad", compression="gzip")

# %%
ctrl = joung_use_imp[joung_use_imp.obs["knockup"] == ""].copy()
ctrl.write("ctrl.h5ad", compression="gzip")

# %%
adata_use_imp.write("adata.h5ad", compression="gzip")
ctrl.write("ctrl.h5ad", compression="gzip")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Construct scaffold

# %%
kegg_marg = marginalize(
    kegg, adata_use_imp.var_names, max_steps=kegg.graph["marginalize_steps"]
)
tf_target_marg = marginalize(
    tf_target, adata_use_imp.var_names, max_steps=tf_target.graph["marginalize_steps"]
)
ppi_marg = marginalize(
    ppi, adata_use_imp.var_names, max_steps=ppi.graph["marginalize_steps"]
)
corr_marg = marginalize(
    corr, adata_use_imp.var_names, max_steps=corr.graph["marginalize_steps"]
)
(
    kegg_marg.number_of_edges(),
    tf_target_marg.number_of_edges(),
    ppi_marg.number_of_edges(),
    corr_marg.number_of_edges(),
)

# %%
nx.set_edge_attributes(kegg_marg, kegg_marg.graph["data_source"], "data_source")
nx.set_edge_attributes(kegg_marg, kegg_marg.graph["evidence_type"], "evidence_type")
nx.set_edge_attributes(
    tf_target_marg, tf_target_marg.graph["data_source"], "data_source"
)
nx.set_edge_attributes(
    tf_target_marg, tf_target_marg.graph["evidence_type"], "evidence_type"
)
nx.set_edge_attributes(ppi_marg, ppi_marg.graph["data_source"], "data_source")
nx.set_edge_attributes(ppi_marg, ppi_marg.graph["evidence_type"], "evidence_type")
nx.set_edge_attributes(corr_marg, corr_marg.graph["data_source"], "data_source")
nx.set_edge_attributes(corr_marg, corr_marg.graph["evidence_type"], "evidence_type")

# %%
scaffold = nx.compose_all([corr_marg, ppi_marg, tf_target_marg, kegg_marg])
scaffold.add_nodes_from(adata_use_imp.var_names)
scaffold.remove_edges_from([(v, v) for v in scaffold.nodes])

# %%
g = sns.clustermap(
    nx.to_pandas_adjacency(scaffold, weight=None), cmap="Blues", rasterized=True
)

# %%
# from cascade.plot import interactive_heatmap

# interactive_heatmap(
#     nx.to_pandas_adjacency(scaffold, weight=None).iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind,
#     ],
#     row_clust=None,
#     col_clust=None,
#     colorscale="blues",
# )

# %% editable=true slideshow={"slide_type": ""}
nx.write_gml(scaffold, "scaffold.gml.gz")

# %% [markdown]
# # Latent data

# %%
latent_data = pd.read_csv("../../data/function/GO/gene2gos_lsi.csv.gz", index_col=0)
latent_data = latent_data.reindex(adata_use_imp.var_names).dropna()
latent_data

# %%
latent_data.to_csv("go_lsi.csv.gz")

# %% [markdown]
# # Design targets

# %%
with open("markers.yaml") as f:
    markers = yaml.load(f, Loader=yaml.Loader)

# %%
target_dir = Path("targets")
if target_dir.exists():
    rmtree(target_dir)
target_dir.mkdir()

# %%
for ct in tqdm(markers):
    target = cao_sub_imp[cao_sub_imp.obs["cell_type"] == ct].copy()
    marker_mask = target.var_names.isin(markers[ct])
    target.var["weight"] = 0.0
    target.var.loc[~marker_mask, "weight"] = marker_mask.size / (~marker_mask).sum()
    target.var.loc[marker_mask, "weight"] = marker_mask.size / marker_mask.sum()
    target.write(target_dir / f"{ct.replace(' ', '_')}.h5ad", compression="gzip")
    de_df = ct_de[ct].set_index("names").join(joung_use.var["perturbed"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        de_df["perturbed"] = de_df["perturbed"].fillna(False)
    np.savetxt(
        target_dir / f"{ct.replace(' ', '_')}.txt",
        de_df.query("perturbed").query("scores > 0").index,
        fmt="%s",
    )
