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
from argparse import ArgumentParser, Namespace
from functools import reduce
from operator import and_
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker

from cascade.data import configure_dataset, encode_regime
from cascade.graph import demultiplex, filter_edges
from cascade.model import CASCADE
from cascade.plot import set_figure_params
from cascade.utils import config, is_notebook

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()

# %%
if is_notebook():
    model_dir = Path(
        "model/nptc=4-dz=8-drop=0.2-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.01-dec=0.01-run_sd=0"
    )
    args = Namespace(
        ctrl=Path("ctrl.h5ad"),
        model=model_dir / "tune.pt",
        batch_size=128,
        n_devices=1,
        discover=model_dir / "discover.gml.gz",
        repressors="repressors.txt",
        activators="activators.txt",
        jacobian=model_dir / "jacobian_sub.h5ad",
        mean_std_cells=model_dir / "mean_std_across_cells.pdf",
        mean_std_particles=model_dir / "mean_std_across_particles.pdf",
        repressor_hist=model_dir / "repressor_hist.pdf",
        repressor_hist_conf=model_dir / "repressor_hist_conf.pdf",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-devices", type=int, default=1)
    parser.add_argument("--discover", type=Path, required=True)
    parser.add_argument("--repressors", type=Path, required=True)
    parser.add_argument("--activators", type=Path, required=True)
    parser.add_argument("--jacobian", type=Path, required=True)
    parser.add_argument("--mean-std-cells", type=Path, required=True)
    parser.add_argument("--mean-std-particles", type=Path, required=True)
    parser.add_argument("--repressor-hist", type=Path, required=True)
    parser.add_argument("--repressor-hist-conf", type=Path, required=True)
    args = parser.parse_args()
vars(args)

# %% [markdown]
# # Read data

# %%
model = CASCADE.load(args.model)

# %%
ctrl = ad.read_h5ad(args.ctrl)

# %% [markdown]
# # Compute Jacobian matrix

# %%
encode_regime(ctrl, "interv", key="knockup")
configure_dataset(
    ctrl,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)

# %%
jacobian = model.jacobian(
    ctrl[np.random.choice(ctrl.n_obs, 2000, replace=False)],
    batch_size=args.batch_size,
    n_devices=args.n_devices,
)

# %%
# jacobian.write(args.jacobian, compression="gzip")

# %%
# jacobian = ad.read_h5ad(args.jacobian)

# %% [markdown]
# # General checks

# %%
discover = nx.read_gml(args.discover)

# %%
discovers = demultiplex(discover)

# %% [markdown]
# ## Whether the gradients match the scaffold

# %%
discovers = [filter_edges(particle, cutoff=0.5) for particle in discovers]
tuple(particle.number_of_edges() for particle in discovers)

# %%
tuple((jacobian.layers["X_jac"][0, :, :, i] != 0).sum() for i in range(len(discovers)))

# %%
discover_masks = [
    nx.to_numpy_array(particle, nodelist=jacobian.var_names, dtype=bool)
    for particle in discovers
]
jac_masks = [jacobian.layers["X_jac"][0, :, :, i] != 0 for i in range(len(discovers))]

# %%
assert all((m1 == m2.T).all() for m1, m2 in zip(discover_masks, jac_masks))


# %% [markdown]
# Note that discover adjacencies have shape *(in_genes, out_genes)* while jacobian
# matrices have shape *(out_genes, in_genes)*. They are ***transposed***!


# %%
def extract_jac_edgelist(graph, jac, index, rdim, cdim):
    edgelist = nx.to_pandas_edgelist(graph)
    ridx = index.get_indexer(edgelist["source"])
    cidx = index.get_indexer(edgelist["target"])
    jac = np.moveaxis(jac, (rdim, cdim), (0, 1))
    edgelist["jac"] = [row for row in jac[(ridx, cidx)]]
    return edgelist


# %% [markdown]
# ## Whether the gradients are consistent across cells

# %%
jac_edgelists = []
for i, particle in enumerate(discovers):
    jac_edgelist = extract_jac_edgelist(
        particle, jacobian.layers["X_jac"][..., i], jacobian.var_names, 2, 1
    )
    jac_edgelist["jac_mean"] = jac_edgelist["jac"].map(np.mean)  # Across cells
    jac_edgelist["jac_std"] = jac_edgelist["jac"].map(np.std)  # Across cells
    jac_edgelists.append(jac_edgelist)

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(8, 8), gridspec_kw={"hspace": 0.4, "wspace": 0.4}
)
for i, (jac_edgelist, ax) in enumerate(zip(jac_edgelists, axes.ravel())):
    ax.errorbar(
        x="jac_mean",
        y="jac_mean",
        yerr="jac_std",
        fmt=".",
        alpha=0.5,
        rasterized=True,
        data=jac_edgelist,
    )
    ax.axhline(y=0, c="darkred", ls="--")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.set_xlabel("Gradient mean across cells")
    ax.set_ylabel("Gradient mean±std across cells")
    ax.set_title(f"Particle {i}")
fig.savefig(args.mean_std_cells)

# %% [markdown]
# ## Whether the gradients are consistent across particles

# %%
jac_mean = jacobian.layers["X_jac"].mean(axis=0)  # Mean across cells

# %%
discover_common = nx.DiGraph(
    reduce(and_, (set(particle.edges) for particle in discovers))
)

# %%
jac_mean_edgelist = extract_jac_edgelist(
    discover_common, jac_mean, jacobian.var_names, 1, 0
)
jac_mean_edgelist["jac_mean"] = jac_mean_edgelist["jac"].map(
    np.mean
)  # Across particles
jac_mean_edgelist["jac_std"] = jac_mean_edgelist["jac"].map(np.std)  # Across particles
jac_mean_edgelist.head()

# %%
fig, ax = plt.subplots()
ax.errorbar(
    x="jac_mean",
    y="jac_mean",
    yerr="jac_std",
    fmt=".",
    alpha=0.5,
    rasterized=True,
    data=jac_mean_edgelist,
)
ax.axhline(y=0, c="darkred", ls="--")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.set_xlabel("Gradient mean across particles")
ax.set_ylabel("Gradient mean±std across particles")
fig.savefig(args.mean_std_particles)

# %% [markdown]
# # Biological checks

# %%
activators = set(np.loadtxt(args.activators, dtype=str))
repressors = set(np.loadtxt(args.repressors, dtype=str))


# %%
def annotate(x):
    if x in activators:
        return "activator"
    if x in repressors:
        return "repressor"
    return "unknown"


# %% [markdown]
# ## Whether edges from repressors tend to be negative


# %%
def back_to_back_hist(
    left,
    right,
    left_bins=10,
    right_bins=10,
    left_label="left",
    right_label="right",
    ax=None,
):
    ax = plt.gca() if ax is None else ax
    hist_left, bins_left = np.histogram(left, bins=left_bins, density=True)
    hist_right, bins_right = np.histogram(right, bins=right_bins, density=True)
    bin_centers_left = 0.5 * (bins_left[1:] + bins_left[:-1])
    bin_centers_right = 0.5 * (bins_right[1:] + bins_right[:-1])
    ax.barh(
        bin_centers_left,
        -np.log1p(hist_left),
        height=np.diff(bins_left),
        color="blue",
        alpha=0.6,
        label=left_label,
        align="center",
    )
    ax.barh(
        bin_centers_right,
        np.log1p(hist_right),
        height=np.diff(bins_right),
        color="red",
        alpha=0.6,
        label=right_label,
        align="center",
    )
    ax.set_xlabel("Log density")
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(int(tick))}" for tick in xticks])
    ax.legend()
    return ax


# %% [markdown]
# ### Single particles

# %%
for i in range(len(discovers)):
    jac_edgelists[i]["function"] = jac_edgelists[i]["source"].map(annotate)
    jac_edgelists[i]["jac_mean_abs"] = jac_edgelists[i]["jac_mean"].abs()

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(8, 8), gridspec_kw={"hspace": 0.4, "wspace": 0.4}
)
for i, (jac_edgelist, ax) in enumerate(zip(jac_edgelists, axes.ravel())):
    ax = back_to_back_hist(
        jac_edgelist.query("function == 'repressor'")["jac_mean"],
        jac_edgelist.query("function == 'activator'")["jac_mean"],
        left_bins=100,
        right_bins=100,
        left_label="From repressors",
        right_label="From activators",
        ax=ax,
    )
    ax.set_ylabel("Gradient")
    ax.set_title(f"Particle {i}")

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(8, 8), gridspec_kw={"hspace": 0.4, "wspace": 0.4}
)
for i, (jac_edgelist, ax) in enumerate(zip(jac_edgelists, axes.ravel())):
    ax = back_to_back_hist(
        jac_edgelist.query("(function == 'repressor') and (jac_mean_abs > 0.05)")[
            "jac_mean"
        ],
        jac_edgelist.query("(function == 'activator') and (jac_mean_abs > 0.05)")[
            "jac_mean"
        ],
        left_bins=15,
        right_bins=40,
        left_label="From repressors",
        right_label="From activators",
        ax=ax,
    )
    ax.set_ylabel("Gradient")
    ax.set_title(f"Particle {i}")
fig.savefig(args.repressor_hist)

# %% [markdown]
# ### Confident edges across particles

# %%
jac_edgelist_conf = pd.concat(jac_edgelists, ignore_index=True)
jac_edgelist_conf = (
    jac_edgelist_conf.groupby(["source", "target"])["jac_mean"]
    .agg(jac_mean="mean", n_particles="count")
    .reset_index()
    .query("n_particles > 0")
)
jac_edgelist_conf["jac_mean_abs"] = jac_edgelist_conf["jac_mean"].abs()
jac_edgelist_conf["function"] = jac_edgelist_conf["source"].map(annotate)
jac_edgelist_conf

# %%
fig, ax = plt.subplots()
ax = back_to_back_hist(
    jac_edgelist_conf.query("(function == 'repressor') and (jac_mean_abs > 0.05)")[
        "jac_mean"
    ],
    jac_edgelist_conf.query("(function == 'activator') and (jac_mean_abs > 0.05)")[
        "jac_mean"
    ],
    left_bins=25,
    right_bins=40,
    left_label="From repressors",
    right_label="From activators",
    ax=ax,
)
ax.set_ylabel("Gradient")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.savefig(args.repressor_hist_conf)

# %% [markdown]
# ## Whether gradients from the same gene tends to be of the same sign

# %%
jac_sourcelist_conf = (
    jac_edgelist_conf.groupby("source")["jac_mean"]
    .agg(jac_mean="mean", jac_std="std", n_targets="count")
    .reset_index()
    .dropna(subset="jac_std")
)
jac_sourcelist_conf["function"] = jac_sourcelist_conf["source"].map(annotate)
jac_sourcelist_conf

# %%
fig, ax = plt.subplots()
ax = sns.scatterplot(
    x="n_targets",
    y="jac_mean",
    hue="function",
    data=jac_sourcelist_conf.query("(function != 'unknown') and (n_targets > 10)"),
    edgecolor=None,
    s=15,
    alpha=0.5,
    ax=ax,
)
ax.errorbar(
    x="n_targets",
    y="jac_mean",
    yerr="jac_std",
    fmt="none",
    data=jac_sourcelist_conf.query("(function != 'unknown') and (n_targets > 10)"),
    alpha=0.5,
)
