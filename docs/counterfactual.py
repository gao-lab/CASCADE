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
# # Stage 3: Counterfactual prediction
#
# In this tutorial, we will walk through how to use the CASCADE model trained
# in [stage 2](training.ipynb) to conduct counterfactual inference.

# %%
import anndata as ad
import networkx as nx
import numpy as np
import scanpy as sc
import seaborn as sns

from cascade.data import configure_dataset, encode_regime, get_configuration
from cascade.graph import annotate_explanation, core_explanation_graph, prep_cytoscape
from cascade.model import CASCADE
from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# ## Read data and model

# %%
adata = sc.read_h5ad("adata.h5ad")

# %%
cascade = CASCADE.load("tune.pt")

# %%
scaffold = nx.read_gml("scaffold.gml.gz")
graph = nx.read_gml("discover.gml.gz")

# %% [markdown]
# ## Specify counterfactual condition
#
# Suppose we want to predict the counterfactual effect of triple gene perturbation
# `"CEBPB,KLF1,MAPK1"` for the negative control cells, we'll need to first extract
# some control cells, and then specify the perturbation in a column in `adata.obs`
# (e.g., `"my_pert"`), in the same comma-separated format as the `"knockup"` column:

# %%
ctrl = adata[adata.obs["knockup"] == ""]
sc.pp.subsample(ctrl, n_obs=1000)
ctrl.obs["my_pert"] = "CEBPB,KLF1,MAPK1"

# %% [markdown]
# Then we call [encode_regime](api/cascade.data.encode_regime.rst) again to
# encode this counterfactual perturbation into a binary regime matrix,
# here in a new layer called `"ctfact"`:

# %%
encode_regime(ctrl, "ctfact", key="my_pert")

# %% [markdown]
# We'd also need to call [configure_dataset](api/cascade.data.configure_dataset.rst)
# again to let the model use this new regime:

# %%
configure_dataset(ctrl, use_regime="ctfact")
get_configuration(ctrl)

# %% [markdown]
# ## Run counterfactual prediction
#
# Now we use the `counterfactual` method to perform counterfactual prediction
# with this newly specified perturbation:

# %%
ctfact = cascade.counterfactual(ctrl, sample=True)

# %% [markdown]
# Here we specified `sample=True` to make the model output random samples from the
# counterfactual negative binomial distribution, which would better represent
# the distribution than a simple mean.
#
# The prediction will be saved in both `ctfact.X` and `ctfact.layers["X_ctfact"]`,
# where `ctfact.X` is the average prediction across SVGD particles, and
# `ctfact.layers["X_ctfact"]` contains the per-particle predictions with shape
# `(n_obs, n_vars, n_particles)`. Note that both of these are in raw count scale.

# %%
ctfact.X

# %%
ctfact.layers["X_ctfact"].shape

# %% [markdown]
# > For counterfactual prediction of [CASCADE designs](design.ipynb), you would
# > also need to specify the `design` argument to the `counterfactual` method.
#
# Please visit the documentation of
# [counterfactual](api/cascade.model.CASCADE.counterfactual.rst)
# for more details.
#
# The same can also be achieved using the
# [command line interface](cli.rst#counterfactual-deduction),
# with the following command:
#
# ```sh
# cascade counterfactual -d ctrl.h5ad -m tune.pt -p ctfact.h5ad [other options]
# ```
#
# ## Counterfactual differential expression comparison
#
# To check for counterfactual effects, we are expected to compare the predicted
# dataset (`ctfact`) with the input dataset (`ctrl`). However, to avoid artifacts
# caused by model prediction biases, it is recommended to compare the predicted
# dataset (`ctfact`) with a "nil prediction", i.e., model prediction with the
# original perturbation labels.
#
# Here, we can go back to use the `"interv"` regime to obtain the "nil prediction":

# %%
configure_dataset(ctrl, use_regime="interv")
get_configuration(ctrl)

# %%
nil = cascade.counterfactual(ctrl, sample=True)

# %% [markdown]
# Now we combine and log-normalize both predictions to perform differential
# expression analysis:

# %%
combined = ad.concat({"nil": nil, "ctfact": ctfact}, label="role", index_unique="-")
combined.X = np.log1p(combined.X * (1e4 / combined.obs[["ncounts"]].to_numpy()))
combined

# %%
sc.tl.rank_genes_groups(combined, "role", reference="nil", rankby_abs=True, pts=True)
de_df = sc.get.rank_genes_groups_df(combined, "ctfact").query("pct_nz_group > 0.05")
de_df["-logfdr"] = -np.log10(de_df["pvals_adj"]).clip(lower=-350)
de_df.head()

# %%
ax = sns.scatterplot(data=de_df, x="logfoldchanges", y="-logfdr", edgecolor=None, s=5)

# %% [markdown]
# ## Explain counterfactual prediction
#
# To understand why CASCADE made the above prediction, we can use the `explain`
# method to decompose the contribution into individual components in the model.
#
# The method needs both a factual dataset (`ctrl`) and a counterfactual
# prediction (`ctfact`), properly configured as below:

# %%
configure_dataset(ctrl, use_regime="interv")
get_configuration(ctrl)

# %%
configure_dataset(ctfact, use_layer="X_ctfact")
get_configuration(ctfact)

# %%
explanation = cascade.explain(ctrl, ctfact)
explanation

# %% [markdown]
# The explanation is also an `AnnData` dataset with predictions from individual
# model components saved in separate layers.
#
# We can pass this explanation dataset to the
# [annotate_explanation](api/cascade.graph.annotate_explanation.rst) function
# to annotate the contributions on graph nodes and edges:

# %%
explanation_graph = annotate_explanation(
    graph, explanation, cascade.export_causal_map()
)
explanation_graph.number_of_nodes(), explanation_graph.number_of_edges()

# %% [markdown]
# Note that `annotate_explanation` has a `cutoff` argument (by default 0.2)
# that you may want to adjust, which specifies an cutoff of predicted effect
# (in log-normalized expression space), below which the effect is deemed too
# small to explain.
#
# We can then extract a core subgraph that explains expression changes for a
# specific list of genes (e.g., top 10 genes with the most prominent changes),
# using the [core_explanation_graph](api/cascade.graph.core_explanation_graph.rst)
# function:

# %%
response = de_df["names"].head(20).to_list()
response

# %%
core_subgraph = core_explanation_graph(explanation_graph, response)
core_subgraph.number_of_nodes(), core_subgraph.number_of_edges()

# %% [markdown]
# This core subgraph can be exported to [Cytoscape](https://cytoscape.org/)
# for visualization using the utility function
# [prep_cytoscape](api/cascade.graph.prep_cytoscape.rst):

# %%
nx.write_gml(
    prep_cytoscape(core_subgraph, scaffold, ["CEBPB", "KLF1", "MAPK1"], response),
    "cytoscape.gml",
)

# %% [markdown]
# You may download a template Cytoscape file containing corresponding styles from:
#
# - http://ftp.cbi.pku.edu.cn/pub/cascade-download/template.cys
#
# Click here to apply style:
#
# ![style](style.png)
#
# Below is an example visualization of the above core explanation graph:
#
# ![cytoscape](cytoscape.png)
