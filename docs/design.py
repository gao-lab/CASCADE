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

# %% [markdown]
# # Stage 4: Intervention design
#
# In this tutorial, we'll walk through how to use the CASCADE model trained in
# [stage 2](training.ipynb) to perform targeted intervention design, using
# K562-to-erythroid differentiation as an example.

# %%
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix

from cascade.data import (
    configure_dataset,
    encode_regime,
    get_all_targets,
    get_configuration,
)
from cascade.model import CASCADE, IntervDesign
from cascade.plot import plot_design_error_curve, plot_design_scores, set_figure_params

# %%
set_figure_params()

# %% [markdown]
# ## Prepare source data
#
# First, we will extract the source state, i.e., unperturbed K562 cells from our
# preprocessed data:

# %%
adata = sc.read_h5ad("adata.h5ad")
adata

# %%
source = adata[adata.obs["knockup"] == ""].copy()
source

# %% [markdown]
# ## Prepare target data
#
# Next, we use an erythroid scRNA-seq dataset from
# [Xu, et al. (2022)](https://doi.org/10.1038/s41590-022-01245-8) as the
# target state. The data file can be downloaded from here:
#
# - http://ftp.cbi.pku.edu.cn/pub/cascade-download/Xu-2022.h5ad

# %%
target = sc.read_h5ad("Xu-2022.h5ad")
target

# %% [markdown]
# Some genes in our source dataset could be missing from the target data, so we
# manually expand the target data with zero-paddings:

# %%
common_vars = [i for i in adata.var_names if i in target.var_names]
missing_vars = [i for i in adata.var_names if i not in target.var_names]
len(common_vars), len(missing_vars)

# %%
empty = ad.AnnData(
    X=csr_matrix((target.n_obs, len(missing_vars)), dtype=target.X.dtype),
    obs=target.obs,
    var=pd.DataFrame(index=missing_vars),
)
target = ad.concat([target, empty], axis=1, merge="first")
target

# %% [markdown]
# Similar to the training data, we backup the raw counts and log-normalize the
# dataset.

# %%
target.layers["counts"] = target.X.copy()
sc.pp.normalize_total(target, target_sum=1e4)
sc.pp.log1p(target)

# %% [markdown]
# We'd also need to configure the target dataset. Of note, the target data
# configuration should provide the same covariate as the training data. However,
# it obviously does not have batch info compatible with the training data.
#
# Here we are using a rough approach of simply taking the covariate
# of random training cells. Another possibility is to take the covariate of
# training cells most similar to the target cells, based on the expression levels
# of housekeeping genes.

# %%
target.obsm["covariate"] = source.obsm["covariate"][
    np.random.choice(source.n_obs, target.n_obs)
]

# %%
configure_dataset(
    target, use_covariate="covariate", use_size="ncounts", use_layer="counts"
)
get_configuration(target)

# %% [markdown]
# ## Define gene weights
#
# Given that the source and target data were produced with different experimental
# protocols in different studies, batch effect would be a substantial problem.
# To mitigate this issue, we give higher weight to known erythroid markers when
# comparing counterfactual states with the target state during intervention
# design, which helps avoid learning batch effect between source and target.

# %%
markers = [
    "AHSP",
    "ALAS2",
    "ALDOA",
    "BLVRB",
    "BPGM",
    "CLIC1",
    "ENO1",
    "GYPA",
    "GYPB",
    "HAMP",
    "HBA1",
    "HBA2",
    "HBB",
    "HBD",
    "HBE1",
    "HBG1",
    "HBG2",
    "HBZ",
    "HEMGN",
    "LDHA",
    "PRDX1",
    "PRDX2",
    "SLC25A37",
    "SLC4A1",
    "SMIM1",
]
assert not set(markers) - set(adata.var_names)
assert not set(markers) - set(target.var_names)
len(markers)

# %%
non_markers = [i for i in common_vars if i not in markers]
len(non_markers)

# %%
target.var["weight"] = 0.0
target.var.loc[non_markers, "weight"] = len(common_vars) / len(non_markers) / 2
target.var.loc[markers, "weight"] = len(common_vars) / len(markers) / 2

# %%
target.var.loc[non_markers, "weight"].head()

# %%
target.var.loc[markers, "weight"].head()

# %% [markdown]
# In this case we scaled the gene weights so that the total weight of marker genes
# equals and the total weight of non-marker genes, and genes that were zero-padded
# in the target data were given zero weight, so they will not bias the result.
#
# ## Specify candidate genes
#
# We can also provide a candidate gene pool, e.g., we'll just use genes perturbed
# in the CRISPRa dataset that show higher expression in the target cells:

# %%
all_targets = sorted(get_all_targets(adata, "knockup"))
target_mask = (
    target[:, all_targets].to_df().mean() > adata[:, all_targets].to_df().mean()
)
candidates = target_mask.index[target_mask].to_list()
candidates

# %% [markdown]
# ## Run intervention design
#
# > (Estimated time: 10 min – 20 min, depending on computation device)
#
# Finally, we can load our trained model for intervention design:

# %%
cascade = CASCADE.load("tune.pt")

# %% [markdown]
# For the sake of speed, we will subsample both the source and target data to
# 5,000 cells.

# %%
sc.pp.subsample(source, n_obs=5000)
sc.pp.subsample(target, n_obs=5000)

# %% [markdown]
# CASCADE model provides a dedicated [design](api/cascade.model.CASCADE.design.rst)
# method for intervention design, which uses differentiable optimization to
# optimize interventions that produces effects more similar to the target state.
#
# We'll need to pass the source and target datasets, along with a candidate gene
# pool, a maximal combination order (`design_size=1` for designing single-gene
# perturbations), as well as the target gene weight we just assigned.

# %%
scores, design = cascade.design(
    source, target, pool=candidates, design_size=1, target_weight="weight"
)

# %% [markdown]
# The `design` method returns two objects:
#
# - The `scores` object is a data frame containing scores of candidate genes
#   (or gene combinations if `design_size` was larger than 1).
# - The `design` object is an [IntervDesign](api/cascade.nn.IntervDesign.rst)
#   module that contains both the scores and optimized interventional scales and
#   biases for the designed interventions, which can also be saved and loaded just
#   like the CASCADE model.

# %%
scores.to_csv("design.csv")
scores.head()

# %%
design.save("design.pt")
design = IntervDesign.load("design.pt")

# %% [markdown]
# The same can also be achieved using the
# [command line interface](cli.rst#intervention-design)
# with the following command.
#
# ```sh
# cascade design -d source.h5ad -m tune.pt -t target.h5ad \
#     --pool candidates.txt -o design.csv -u design.pt \
#     --design-size 1 --target-weight weight [other options]
# ```

# %% [markdown]
# ## Determine design score cutoff

# %%
curve, cutoff = cascade.design_error_curve(source, target, design, n_cells=5000)

# %%
plot_design_error_curve(curve, cutoff=cutoff)

# %% [markdown]
# We can also visualize the scores with a scatter plot:

# %%
plot_design_scores(scores, cutoff=cutoff)

# %% [markdown]
# ## Verify design with counterfactual prediction
#
# The designed perturbation can certainly be passed back to the
# [counterfactual](api/cascade.model.CASCADE.counterfactual.rst) method to verify
# whether the target markers are up-regulated, just like what we did in
# [stage 3](counterfactual.ipynb).
#
# One notable difference is that we need to specify the `design` argument of the
# [counterfactual](api/cascade.model.CASCADE.counterfactual.rst) method, which
# tells the model to use the designed interventional scales and biases
# (instead of those from the training set) when making counterfactual
# predictions.

# %%
source.obs["my_pert"] = "KLF1"
encode_regime(source, "ctfact", key="my_pert")

# %%
configure_dataset(source, use_regime="ctfact")
ctfact = cascade.counterfactual(source, design=design, sample=True)

# %%
configure_dataset(source, use_regime="interv")
nil = cascade.counterfactual(source, design=design, sample=True)

# %%
combined = ad.concat({"nil": nil, "ctfact": ctfact}, label="role", index_unique="-")
combined.X = np.log1p(combined.X * (1e4 / combined.obs[["ncounts"]].to_numpy()))
combined

# %%
sc.tl.rank_genes_groups(combined, "role", reference="nil", rankby_abs=True, pts=True)
de_df = sc.get.rank_genes_groups_df(combined, "ctfact").query("pct_nz_group > 0.05")
de_df["-logfdr"] = -np.log10(de_df["pvals_adj"]).clip(lower=-350)
de_df["is_marker"] = de_df["names"].isin(markers)
de_df.head()

# %%
_ = sns.scatterplot(
    data=de_df, x="logfoldchanges", y="-logfdr", hue="is_marker", edgecolor=None, s=10
)
