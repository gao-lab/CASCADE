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
# %load_ext autoreload
# %autoreload 2

# %%
import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt

from cascade.data import configure_dataset, encode_regime, get_configuration
from cascade.model import CASCADE, IntervDesign
from cascade.plot import plot_design_error_curve, plot_design_scores, set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Read data

# %%
ctrl = ad.read_h5ad("ctrl.h5ad")
target = ad.read_h5ad("target.h5ad")

# %% [markdown]
# # Read design

# %%
RUN = "nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-lam=0.1-alp=0.5-run_sd=1"

# %%
model = CASCADE.load(f"model/{RUN}/tune.pt")

# %%
design = IntervDesign.load(f"model/{RUN}/design/size=1/design.pt")

# %%
scores = pd.read_csv(
    f"model/{RUN}/design/size=1/design.csv",
    index_col=0,
    keep_default_na=False,
)
scores.head()

# %% [markdown]
# # Design error curve

# %%
encode_regime(ctrl, "interv", key="knockup")
configure_dataset(
    ctrl,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)
get_configuration(ctrl)

# %%
configure_dataset(
    target, use_covariate="covariate", use_size="ncounts", use_layer="counts"
)
get_configuration(target)

# %%
curve, cutoff = model.design_error_curve(ctrl, target, design, n_cells=1000)

# %% [markdown]
# # Visualization

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
plot_design_error_curve(curve, cutoff=cutoff, ax=ax)
fig.savefig(f"model/{RUN}/design/size=1/design_error_curve.pdf")

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
plot_design_scores(scores, cutoff=cutoff, ax=ax)
fig.savefig(f"model/{RUN}/design/size=1/design_scores.pdf")
