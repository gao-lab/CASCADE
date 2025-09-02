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
# # Stage 2: Model training
#
# In this tutorial, we will walk through how to train the CASCADE model using the
# preprocessed data from [stage 1](preprocessing.ipynb).

# %%
import networkx as nx
import pandas as pd
import scanpy as sc

from cascade.graph import acyclify, demultiplex, filter_edges, multiplex
from cascade.model import CASCADE

# %% [markdown]
# ## Read preprocessed data

# %%
adata = sc.read_h5ad("adata.h5ad")

# %%
scaffold = nx.read_gml("scaffold.gml.gz")

# %%
latent_emb = pd.read_csv("latent_emb.csv.gz", index_col=0)

# %% [markdown]
# ## Build the CASCADE model
#
# The first step is to build a CASCADE model:

# %%
cascade = CASCADE(
    vars=adata.var_names,
    n_covariates=adata.obsm["covariate"].shape[1],
    scaffold_graph=scaffold,
    latent_data=latent_emb,
    log_dir="log_dir",
)

# %% [markdown]
# This creates a CASCADE model under the default setting. For advanced options,
# visit the documentation of [CASCADE](api/cascade.model.CASCADE.rst) to find out
# more about tunable hyperparameters, modules and their usage.
#
# ## Run causal discovery
#
# > (Estimated time: 30 min – 1 hour, depending on computation device)
#
# To run causal discovery using the CASCADE model, use the `discover` method:

# %% editable=true slideshow={"slide_type": ""}
cascade.discover(adata)
cascade.save("discover.pt")

# %% [markdown]
# This runs CASCADE causal discovery under the default setting. For advanced
# options, visit the documentation of
# [discover](api/cascade.model.CASCADE.discover.rst) for more details.
#
# The same can also be achieved using the
# [command line interface](cli.rst#causal-discovery),
# with the following command:
#
# ```sh
# cascade discover -d adata.h5ad -m discover.pt \
#     --scaffold-graph scaffold.gml.gz \
#     --latent-data latent_emb.csv.gz [other options]
# ```
#
# > You may use `tensorboard --logdir .` to monitor the training process.
#
# ## Remove remaining cycles
#
# Due to numerical limitations, some cycles may still remain in the resulting model.
# We further use graph utility functions to ensure directed acyclic graphs, which
# is required for downstream inferences.

# %%
graph = cascade.export_causal_graph()
graph = multiplex(*[acyclify(filter_edges(g, cutoff=0.5)) for g in demultiplex(graph)])
nx.write_gml(graph, "discover.gml.gz")

# %% [markdown]
# The same can also be achieved using the
# [command line interface](cli.rst#graph-acyclification),
# with the following command:
#
# ```sh
# cascade acyclify -m discover.pt -g discover.gml.gz [other options]
# ```
#
# ## Model tuning
#
# > (Estimated time: 15 min – 30 min, depending on computation device)
#
# Next, we reimport the acyclified graph back into the model:

# %%
cascade.import_causal_graph(graph)

# %% [markdown] editable=true slideshow={"slide_type": ""}
# Now we can fine tune the structural equations in the model using the `tune`
# method to adapt for removed edges during the acyclification step. It is also
# recommended to enable the counterfactual tuning mode, where the tuning process
# is specifically optimized for counterfactual prediction.

# %% editable=true slideshow={"slide_type": ""}
cascade.tune(adata, tune_ctfact=True)
cascade.save("tune.pt")

# %% [markdown]
# For advanced options, visit the documentation of
# [tune](api/cascade.model.CASCADE.tune.rst) for more details.
#
# The same can also be achieved using the
# [command line interface](cli.rst#model-tuning),
# using the following command:
#
# ```sh
# cascade tune -d adata.h5ad -g discover.gml.gz -m discover.pt -o tune.pt \
#     --tune-ctfact [other options]
# ```
#
# Now this tuned model is ready for counterfactual prediction in
# [stage 3](counterfactual.ipynb) and intervention design in
# [stage 4](design.ipynb).
