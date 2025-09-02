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
from matplotlib import pyplot as plt

from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Read data

# %%
path = (
    "model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
    "lam=0.1-alp=0.5-run_sd=1/design/size=1/"
)

# %%
ctfact = ad.read_h5ad(f"{path}/ctfact.h5ad")
ctfact.X = np.log1p(ctfact.X * (1e4 / ctfact.obs[["ncounts"]].to_numpy()))
ctfact.obs["knockup"] = ctfact.obs["knockup"].cat.rename_categories({"": "Control"})
ctfact.obs["dataset"] = "Norman-2019"
ctfact

# %%
target = ad.read_h5ad("target.h5ad")
target.obs["knockup"] = "Target"
target

# %%
rnd = np.random.RandomState(0)
sub_idx = rnd.choice(target.n_obs, ctfact.n_obs, replace=False)
target_sub = target[sub_idx].copy()
target_sub

# %%
combined = ad.concat({"ctfact": ctfact, "target": target_sub}, label="role")
combined.obs["knockup"] = pd.Categorical(
    combined.obs["knockup"],
    categories=["Control", "BPGM", "MAML2", "PRDM1", "ZBTB1", "KLF1", "Target"],
)

# %%
sc.pp.pca(combined, n_comps=2)

# %%
fig, ax = plt.subplots()
plot_idx = rnd.permutation(np.where(combined.obs["dataset"] != "Xu-2022")[0])
sc.pl.pca(
    combined[plot_idx].copy(),
    color="knockup",
    palette={
        "Control": "lightgrey",
        "BPGM": "#aec7e8",
        "MAML2": "#98df8a",
        "PRDM1": "#c5b0d5",
        "ZBTB1": "#dbdb8d",
        "KLF1": "#ff9896",
        "Target": "#d62728",
    },
    annotate_var_explained=True,
    title="",
    ax=ax,
)
fig.savefig(f"{path}/pca.pdf")
