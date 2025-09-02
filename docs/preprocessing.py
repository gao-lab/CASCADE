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
# # Stage 1: Data preprocessing
#
# In this tutorial, we'll first walk through how to prepare the datasets for use in
# CASCADE, using the [Norman, et al. (2019)](https://doi.org/10.1126/science.aax4438)
# dataset as an example. This dataset contains single- and double-gene CRISPRa
# perturbations.

# %%
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cascade.data import (
    configure_dataset,
    encode_regime,
    filter_unobserved_targets,
    get_all_targets,
    get_configuration,
    neighbor_impute,
)
from cascade.graph import assemble_scaffolds

# %% [markdown]
# ## Read data
#
# First, we need to prepare the dataset into `AnnData` objects. See the
# [documentation](https://anndata.readthedocs.io/) for more details if you are
# unfamiliar, including how to construct `AnnData` objects from scratch, and how
# to read data in other formats (csv, mtx, loom, etc.) into `AnnData` objects.
#
# Here we just load existing `h5ad` files, which is the native file format for
# `AnnData`. The `h5ad` file used in this tutorial can be downloaded from here:
#
# - http://ftp.cbi.pku.edu.cn/pub/cascade-download/Norman-2019.h5ad

# %%
adata = sc.read_h5ad("Norman-2019.h5ad")
adata

# %% [markdown]
# ## Data format requirements
#
# CASCADE requires the following data format:
#
# - Raw counts in `adata.X`;
# - Total count in `adata.obs`, which would be used when fitting data with
#   the negative binomial distribution;
# - HGNC gene symbols as `adata.var_names`;
# - Perturbation label in `adata.obs` that specifies which genes are perturbed
#   in each cell:
#   - For control cells with no perturbation, the value **MUST** be an empty
#     string `""`;
#   - For cells with multiple perturbations, the perturbed genes should be
#     comma-separated, e.g., `"CEBPB,KLF1"`;
#   - Name of perturbed genes must match those in `adata.var_names`.
#
# In this case, we can verify that the expression matrix contains raw counts:

# %%
adata.X, adata.X.data

# %% [markdown]
# The total counts are stored as `"ncounts"` in `adata.obs`:

# %%
adata.obs["ncounts"]

# %% [markdown]
# And perturbation labels are stored as `"knockup"` in `adata.obs`:

# %%
adata.obs["knockup"]

# %% [markdown]
# Before any further processing, we back up the raw UMI counts in a layer called
# `“counts”`, which will be used later during model training.

# %%
adata.layers["counts"] = adata.X.copy()

# %% [markdown]
# ## Cell and gene selection
#
# Since CASCADE can only model perturbations in measured genes, we first filter out
# any perturbation that was missing from the readout. A utility function called
# [filter_unobserved_targets](api/cascade.data.filter_unobserved_targets.rst) is
# provided for this purpose.
#
# In this case no cell was filtered:

# %%
filter_unobserved_targets(adata, "knockup")

# %% [markdown]
# Next, we identify highly variable genes using the `"seurat_v3"` method,
# to allow the model to focus on informative genes:

# %%
sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat_v3")

# %% [markdown]
# Again, as CASCADE can only model perturbations in measured genes, we expand this
# highly variable gene set to incorporating all perturbed genes (via a utility
# function [get_all_targets](api/cascade.data.get_all_targets.rst)) to avoid
# discarding useful perturbation information:

# %%
all_targets = get_all_targets(adata, key="knockup")
all_targets

# %%
adata.var["selected"] = adata.var["highly_variable"] | adata.var_names.isin(all_targets)
adata.var["selected"].sum()

# %% [markdown]
# ## Encode intervention regime
#
# CASCADE represents genetic perturbations as a cell-by-gene binary matrix,
# which can be encoded from the `adata.obs["knockup"]` column using the
# [encode_regime](api/cascade.data.encode_regime.rst) function. The function
# stores the encoded regime matrix in a layer with user-specified name,
# here using the name `"interv"`.

# %%
encode_regime(adata, "interv", key="knockup")
adata.layers["interv"]

# %% [markdown]
# ## Encode technical covariates
#
# To minimize the effect of technical confounding on the causal discovery process,
# it is recommended to add all possible confounding factors into a covariate matrix
# in `adata.obsm`.
#
# Here we will add the one-hot encoded batch label (`"gemgroup"`) and log-centered
# total counts as the covariate:

# %%
batch = OneHotEncoder().fit_transform(adata.obs[["gemgroup"]]).toarray()
batch

# %%
log_ncounts = StandardScaler().fit_transform(np.log10(adata.obs[["ncounts"]]))
log_ncounts

# %%
adata.obsm["covariate"] = np.concatenate([batch, log_ncounts], axis=1)
adata.obsm["covariate"].shape

# %% [markdown]
# ## Data normalization
#
# Next, we follow the standard scRNA-seq preprocessing approach in `scanpy`
# to normalize the expression matrix in `adata.X`. You may visit its
# [documentation](https://scanpy.readthedocs.io/) if unfamiliar.

# %%
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# %% [markdown]
# Now we can subset the dataset to retain the selected genes only:

# %%
adata = adata[:, adata.var["selected"]].copy()
adata

# %% [markdown]
# ## Neighbor-based imputation
#
# Given that scRNA-seq data can be sparse, we recommend conducting a lightweight
# neighbor-based data imputation before model training. This is done by aggregating
# similar cells in the PCA space with the same perturbation. We provide a utility
# function called [neighbor_impute](api/cascade.data.neighbor_impute.rst) for this
# purpose:

# %%
sc.pp.pca(adata)

# %%
adata = neighbor_impute(
    adata,
    k=20,
    use_rep="X_pca",
    use_batch="knockup",
    X_agg="mean",
    obs_agg={"ncounts": "sum"},
    obsm_agg={"covariate": "mean"},
    layers_agg={"counts": "sum"},
)

# %% [markdown]
# Note that we used the `"sum"` aggregation for `adata.obs["ncounts"]` and
# `adata.layers["counts"]`, which maintains their count-based nature.
#
# ## Configure dataset
#
# Now we can use the function
# [configure_dataset](api/cascade.data.configure_dataset.rst) to tell CASCADE
# where the expression matrix, intervention regime, covariates and total counts
# are stored:

# %%
configure_dataset(
    adata,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)
get_configuration(adata)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Construct scaffold graph
#
# Next, we need to construct a scaffold graph to guide the causal discovery process.
#
# The following 4 pre-built human gene scaffolds are available for download:
#
# - KEGG pathways:
#   http://ftp.cbi.pku.edu.cn/pub/cascade-download/inferred_kegg_gene_only.gml.gz
# - TF-target predictions:
#   http://ftp.cbi.pku.edu.cn/pub/cascade-download/TF-target.gml.gz
# - BioGRID protein-protein interactions:
#   http://ftp.cbi.pku.edu.cn/pub/cascade-download/biogrid.gml.gz
# - GTEx correlated genes:
#   http://ftp.cbi.pku.edu.cn/pub/cascade-download/corr.gml.gz

# %%
kegg = nx.read_gml("inferred_kegg_gene_only.gml.gz")
tf_target = nx.read_gml("TF-target.gml.gz")
biogrid = nx.read_gml("biogrid.gml.gz")
corr = nx.read_gml("corr.gml.gz")

# %% [markdown]
# The individual scaffold components can be assembled into a hybrid scaffold using
# the [assemble_scaffolds](api/cascade.graph.assemble_scaffolds.rst) function,
# which also marginalizes these components with regard to the genes being
# modeled here:

# %%
scaffold = assemble_scaffolds(corr, biogrid, tf_target, kegg, nodes=adata.var_names)
scaffold.number_of_nodes(), scaffold.number_of_edges()

# %% [markdown] editable=true slideshow={"slide_type": ""}
# ## Prepare gene function embeddings
#
# Lastly, we also fetch relevant entries from gene embeddings pre-computed using
# LSI of their GO annotations, which can be downloaded here:
#
# - http://ftp.cbi.pku.edu.cn/pub/cascade-download/gene2gos_lsi.csv.gz
#
# This will serve as the input of the interventional latent variable in CASCADE:

# %%
latent_emb = pd.read_csv("gene2gos_lsi.csv.gz", index_col=0)
latent_emb = latent_emb.reindex(adata.var_names).dropna()
latent_emb.shape

# %% [markdown]
# ## Save processed data files
#
# Finally, save the preprocessed data files for use in [stage 2](training.ipynb).

# %%
adata.write("adata.h5ad", compression="gzip")

# %%
nx.write_gml(scaffold, "scaffold.gml.gz")

# %%
latent_emb.to_csv("latent_emb.csv.gz")

# %% [markdown]
# ## Afterwords
#
# Described above is the minimal preprocessing for running CASCADE. Additional
# steps such as filtering non-perturbed cells using
# [mixscape](https://pertpy.readthedocs.io/en/latest/tutorials/notebooks/mixscape.html)
# may also be useful depending on the data at hand.
