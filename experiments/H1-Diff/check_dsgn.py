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
from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt

from cascade.data import configure_dataset, encode_regime, get_configuration
from cascade.model import CASCADE, IntervDesign
from cascade.plot import plot_design_error_curve, plot_design_scores, set_figure_params
from cascade.utils import is_notebook

# %%
set_figure_params()

# %% [markdown]
# # Parametrization

# %%
if is_notebook():
    ct = "Erythroblasts"
    size = 2
    model_path = Path(
        "model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.1-alp=0.5-run_sd=1"
    )
    design_path = model_path / "design" / f"target={ct}-size={size}"
    args = Namespace(
        ctrl=Path("ctrl.h5ad"),
        target=Path(f"targets/{ct}.h5ad"),
        model=model_path / "tune.pt",
        scores=design_path / "design.csv",
        design=design_path / "design.pt",
        design_error_curve=design_path / "curve.pdf",
        design_scores=design_path / "scores.pdf",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--design-error-curve", type=Path, required=True)
    parser.add_argument("--design-scores", type=Path, required=True)
    args = parser.parse_args()
vars(args)

# %% [markdown]
# # Read data

# %%
ctrl = ad.read_h5ad(args.ctrl)
target = ad.read_h5ad(args.target)

# %% [markdown]
# # Read design

# %%
model = CASCADE.load(args.model)
design = IntervDesign.load(args.design)

# %%
scores = pd.read_csv(args.scores, index_col=0, keep_default_na=False)

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
curve, cutoff = model.design_error_curve(ctrl, target, design)

# %% [markdown]
# # Visualization

# %%
fig, ax = plt.subplots()
plot_design_error_curve(curve, cutoff=cutoff, ax=ax)
fig.savefig(args.design_error_curve)

# %%
fig, ax = plt.subplots()
plot_design_scores(scores, n_scatter=1000, n_label=10, cutoff=cutoff, ax=ax)
fig.savefig(args.design_scores)
