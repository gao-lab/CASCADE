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
import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import rcParams
from matplotlib_venn import venn3
from sklearn.neighbors import NearestNeighbors

from cascade.data import Targets, get_all_targets, neighbor_impute
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
norman = ad.read_h5ad("../../data/datasets/Norman-2019.h5ad")


# %%
xu = ad.read_h5ad("../../data/datasets/Xu-2022.h5ad")
xu_bm = ad.read_h5ad("../../data/datasets/Xu-2022-BM.h5ad")
huang = ad.read_h5ad("../../data/datasets/Huang-2020.h5ad")
xie = ad.read_h5ad("../../data/datasets/Xie-2021.h5ad")
neurips_multiome = ad.read_h5ad("../../data/datasets/NeurIPS-2021-Multiome-GEX.h5ad")
neurips_cite = ad.read_h5ad("../../data/datasets/NeurIPS-2021-CITE-GEX.h5ad")

# %%
xie = xie[xie.obs["cell_type"] == "ery"]
neurips_multiome = neurips_multiome[
    neurips_multiome.obs["cell_type"].isin(
        {"Proerythroblast", "Erythroblast", "Normoblast"}
    )
]
neurips_cite = neurips_cite[
    neurips_cite.obs["cell_type"].isin(
        {"Proerythroblast", "Erythroblast", "Normoblast", "Reticulocyte"}
    )
]

# %%
kegg = nx.read_gml("../../data/scaffold/KEGG/inferred_kegg_gene_only.gml.gz")
tf_target = nx.read_gml("../../data/scaffold/TF-target/TF-target.gml.gz")
ppi = nx.read_gml("../../data/scaffold/BioGRID/biogrid.gml.gz").to_directed()
corr = nx.read_gml("../../data/scaffold/GTEx/corr.gml.gz").to_directed()

# %%
with open("markers.yaml") as f:
    markers = yaml.load(f, Loader=yaml.Loader)["Ery"]
len(markers)

# %% [markdown]
# # Merge data

# %%
combined = ad.concat(
    {
        "Norman-2019": norman,
        "Xu-2022": xu,
        "Xu-2022-BM": xu_bm,
        "Huang-2020": huang,
        "Xie-2021": xie,
        "NeurIPS-2021-Multiome": neurips_multiome,
        "NeurIPS-2021-CITE": neurips_cite,
    },
    label="dataset",
    index_unique="-",
)
combined.obs["role"] = combined.obs["dataset"].map(
    lambda x: "source" if x == "Norman-2019" else "target"
)
combined

# %% [markdown]
# # Determine gene set

# %% [markdown]
# ## Differentially expressed HVGs

# %%
sc.tl.rank_genes_groups(combined, "role", reference="source")
de_df = (
    sc.get.rank_genes_groups_df(combined, "target")
    .set_index("names")
    .assign(
        highly_variable_rank=norman.var["highly_variable_rank"],
        perturbed=norman.var["perturbed"],
    )
)
de_df

# %%
top_de_hvgs = de_df.query("highly_variable_rank < 5000 & (scores > 75 | scores < -75)")
top_de_hvgs

# %% [markdown]
# ## Perturbed genes

# %%
perturbed = get_all_targets(norman, "knockup")
perturbed

# %%
tfs = {g for g in tf_target.nodes if tf_target.out_degree(g)}
_ = venn3(
    [set(top_de_hvgs.index), tfs, set(perturbed)],
    set_labels=["DE HVGs", "TFs", "Perturbed"],
)

# %%
selected_genes = set(top_de_hvgs.index) | (set(combined.var_names) & perturbed)
len(selected_genes)

# %%
knockups = norman.obs["knockup"].unique()
knockups = pd.Series(knockups, index=knockups).map(
    lambda x: not (Targets(x) - selected_genes)
)

# %%
norman_use = norman[
    knockups.loc[norman.obs["knockup"]],
    [g in selected_genes for g in norman.var_names],
]
norman_use

# %% [markdown]
# # Impute data

# %%
norman_use.X = np.expm1(norman_use.X)
norman_use_imp = neighbor_impute(
    norman_use,
    k=20,
    use_rep="X_pca",
    use_batch="knockup",
    X_agg="mean",
    obs_agg={"ncounts": "sum"},
    obsm_agg={"covariate": "mean"},
    layers_agg={"counts": "sum"},
)
norman_use_imp.X = np.log1p(norman_use_imp.X)
norman_use.X = np.log1p(norman_use.X)

# %% [markdown]
# # Save data

# %% editable=true slideshow={"slide_type": ""}
norman_use_imp.write("norman.h5ad", compression="gzip")

# %%
ctrl = norman_use_imp[norman_use_imp.obs["knockup"] == ""]
ctrl.write("ctrl.h5ad", compression="gzip")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Construct scaffold

# %%
kegg_marg = marginalize(
    kegg, norman_use_imp.var_names, max_steps=kegg.graph["marginalize_steps"]
)
tf_target_marg = marginalize(
    tf_target, norman_use_imp.var_names, max_steps=tf_target.graph["marginalize_steps"]
)
ppi_marg = marginalize(
    ppi, norman_use_imp.var_names, max_steps=ppi.graph["marginalize_steps"]
)
corr_marg = marginalize(
    corr, norman_use_imp.var_names, max_steps=corr.graph["marginalize_steps"]
)
(
    kegg_marg.number_of_edges(),
    tf_target_marg.number_of_edges(),
    ppi_marg.number_of_edges(),
    corr_marg.number_of_edges(),
)

# %%
nx.set_edge_attributes(kegg_marg, kegg_marg.graph["data_source"], name="data_source")
nx.set_edge_attributes(
    kegg_marg, kegg_marg.graph["evidence_type"], name="evidence_type"
)
nx.set_edge_attributes(
    tf_target_marg, tf_target_marg.graph["data_source"], name="data_source"
)
nx.set_edge_attributes(
    tf_target_marg, tf_target_marg.graph["evidence_type"], name="evidence_type"
)
nx.set_edge_attributes(ppi_marg, ppi_marg.graph["data_source"], name="data_source")
nx.set_edge_attributes(ppi_marg, ppi_marg.graph["evidence_type"], name="evidence_type")
nx.set_edge_attributes(corr_marg, corr_marg.graph["data_source"], name="data_source")
nx.set_edge_attributes(
    corr_marg, corr_marg.graph["evidence_type"], name="evidence_type"
)

# %%
scaffold = nx.compose_all([corr_marg, ppi_marg, tf_target_marg, kegg_marg])
scaffold.add_nodes_from(norman_use_imp.var_names)
scaffold.remove_edges_from([(v, v) for v in scaffold.nodes])

# %%
g = sns.clustermap(
    nx.to_pandas_adjacency(scaffold, weight=None), cmap="Blues", rasterized=True
)

# %% editable=true slideshow={"slide_type": ""}
nx.write_gml(scaffold, "scaffold.gml.gz")

# %% [markdown]
# # Latent data

# %%
latent_data = pd.read_csv("../../data/function/GO/gene2gos_lsi.csv.gz", index_col=0)
latent_data = latent_data.reindex(norman_use_imp.var_names).dropna()
latent_data

# %%
latent_data.to_csv("go_lsi.csv.gz")

# %% [markdown]
# # Design targets

# %%
stable_genes = de_df.query("pvals_adj > 0.05").index
stable_genes.size

# %%
target = combined[combined.obs["role"] == "target"].copy()

# %%
nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(
    norman[:, stable_genes].X.toarray()
)
nni = nearest_neighbors.kneighbors(
    target[:, stable_genes].X.toarray(), return_distance=False
).ravel()
target.obsm["covariate"] = norman.obsm["covariate"][nni]

# %%
target_use = target[:, norman_use_imp.var_names]
target_use

# %%
marker_mask = target_use.var_names.isin(markers)
target_use.var["weight"] = 0.0
target_use.var.loc[~marker_mask, "weight"] = marker_mask.size / (~marker_mask).sum()
target_use.var.loc[marker_mask, "weight"] = marker_mask.size / marker_mask.sum()

# %%
target_use.write("target.h5ad", compression="gzip")

# %% [markdown]
# # Generate design vars

# %%
candidates = de_df.query("scores > 0 & perturbed").index
candidates

# %%
np.savetxt("candidates.txt", candidates, fmt="%s")
