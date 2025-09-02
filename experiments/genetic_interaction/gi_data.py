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
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import colors
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from cascade.data import aggregate_obs
from cascade.plot import set_figure_params

# %%
set_figure_params()


# %% [markdown]
# # Preparation


# %%
def build_paths(div):
    dat_pat = "ds=Norman-2019/imp=20/n_vars=2000"
    div_pat = f"kg=0.9-kc=0.75-div_sd={div}"
    return {
        "true": Path(f"../../evaluation/dat/{dat_pat}/{div_pat}/test.h5ad"),
        "train": Path(f"../../evaluation/dat/{dat_pat}/{div_pat}/train.h5ad"),
        "nc": Path(f"../../evaluation/dat/{dat_pat}/{div_pat}/ctfact_test.h5ad"),
        "linear": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/nil/{div_pat}/"
            f"linear/dim=10-lam=0.1-run_sd=0/ctfact_test.h5ad"
        ),
        "cpa": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/lsi/{div_pat}/"
            f"cpa/run_sd=0/ctfact_test.h5ad"
        ),
        "biolord": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/lsi/{div_pat}/"
            f"biolord/run_sd=0/ctfact_test.h5ad"
        ),
        "gears": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/go/{div_pat}/"
            f"gears/hidden_size=64-epochs=20-run_sd=0/ctfact_test.h5ad"
        ),
        "scgpt": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/go/{div_pat}/"
            f"scgpt/epochs=20-run_sd=0/ctfact_test.h5ad"
        ),
        "scfoundation": Path(
            f"../../evaluation/inf/{dat_pat}/cpl/go/{div_pat}/"
            f"scfoundation/hidden_size=512-epochs=20-run_sd=0/ctfact_test.h5ad",
        ),
        "cascade": Path(
            f"../../evaluation/inf/{dat_pat}/kegg+tf+ppi+corr/lsi/{div_pat}/"
            f"cascade/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
            f"lam=0.1-alp=0.5-run_sd=0-tune_ct=True-ablt=none/ctfact_test.h5ad"
        ),
    }


# %%
def read_data(paths):
    true = ad.read_h5ad(paths["true"])
    true = true[true.obs["category"] == "0/2 unseen"]
    true = aggregate_obs(true, "knockup", X_agg="mean").to_df()
    train = ad.read_h5ad(paths["train"])
    train = train[train.obs["category"].isin({"0 seen", "1 seen"})]
    train = aggregate_obs(train, "knockup", X_agg="mean").to_df()
    ctrl = train.loc[""]
    diff = train.sub(ctrl, axis="columns")
    additive = pd.DataFrame.from_dict(
        {x: ctrl + diff.loc[x.split(",")].sum(axis=0) for x in true.index},
        orient="index",
    )
    nc = ad.read_h5ad(paths["nc"])
    nc = aggregate_obs(nc, "knockup", X_agg="mean").to_df().loc[true.index]
    linear = ad.read_h5ad(paths["linear"]).to_df().loc[true.index]
    cpa = ad.read_h5ad(paths["cpa"])
    cpa = aggregate_obs(cpa, "knockup", X_agg="mean").to_df().loc[true.index]
    biolord = ad.read_h5ad(paths["biolord"])
    biolord = aggregate_obs(biolord, "knockup", X_agg="mean").to_df().loc[true.index]
    gears = ad.read_h5ad(paths["gears"]).to_df().loc[true.index]
    scgpt = ad.read_h5ad(paths["scgpt"]).to_df().loc[true.index]
    scfoundation = ad.read_h5ad(paths["scfoundation"]).to_df().loc[true.index]
    cascade = ad.read_h5ad(paths["cascade"])
    cascade.X = np.log1p(cascade.X * (1e4 / cascade.obs[["ncounts"]].to_numpy()))
    cascade = aggregate_obs(cascade, "knockup", X_agg="mean").to_df().loc[true.index]
    return (
        true,
        ctrl,
        additive,
        nc,
        linear,
        cpa,
        biolord,
        gears,
        scgpt,
        scfoundation,
        cascade,
    )


# %%
def cmp_scatter(pred, name):
    true_dev = true - additive
    pred_dev = pred - additive
    cmp = pd.DataFrame(
        {
            "true_dev": true_dev.to_numpy().ravel(),
            "pred_dev": pred_dev.to_numpy().ravel(),
            "additive": additive.sub(ctrl, axis="columns").to_numpy().ravel(),
        }
    )
    corr = cmp.corr().loc["true_dev", "pred_dev"]
    rmse = (cmp["true_dev"] - cmp["pred_dev"]).pow(2).mean() ** 0.5
    hue_norm = colors.TwoSlopeNorm(
        vmin=cmp["additive"].min(), vcenter=0, vmax=cmp["additive"].max()
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.scatterplot(
        data=cmp,
        x="true_dev",
        y="pred_dev",
        hue="additive",
        palette="bwr",
        hue_norm=hue_norm,
        edgecolor=None,
        s=5,
        rasterized=True,
        ax=ax,
    )
    ax.axline((0, 0), slope=1, color="darkred", ls="--")
    ax.set_title(f"Corr = {corr:.3f}, RMSE = {rmse:.3f}")
    ax.set_xlabel("True – Additive")
    ax.set_ylabel(f"{name} – Additive")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Additive effect")
    plt.close()
    return fig, corr, rmse


# %% [markdown]
# # Prediction errors

# %%
corr, rmse = defaultdict(dict), defaultdict(dict)
for div in tqdm(range(5)):
    paths = build_paths(div)
    (
        true,
        ctrl,
        additive,
        nc,
        linear,
        cpa,
        biolord,
        gears,
        scgpt,
        scfoundation,
        cascade,
    ) = read_data(paths)
    fig, corr["No change"][div], rmse["No change"][div] = cmp_scatter(nc, "No change")
    fig.savefig(paths["nc"].parent / "gi.pdf")
    fig, corr["Linear"][div], rmse["Linear"][div] = cmp_scatter(linear, "Linear")
    fig.savefig(paths["linear"].parent / "gi.pdf")
    fig, corr["CPA"][div], rmse["CPA"][div] = cmp_scatter(cpa, "CPA")
    fig.savefig(paths["cpa"].parent / "gi.pdf")
    fig, corr["Biolord"][div], rmse["Biolord"][div] = cmp_scatter(biolord, "Biolord")
    fig.savefig(paths["biolord"].parent / "gi.pdf")
    fig, corr["GEARS"][div], rmse["GEARS"][div] = cmp_scatter(gears, "GEARS")
    fig.savefig(paths["gears"].parent / "gi.pdf")
    fig, corr["scGPT"][div], rmse["scGPT"][div] = cmp_scatter(scgpt, "scGPT")
    fig.savefig(paths["scgpt"].parent / "gi.pdf")
    fig, corr["scFoundation"][div], rmse["scFoundation"][div] = cmp_scatter(
        scfoundation, "scFoundation"
    )
    fig.savefig(paths["scfoundation"].parent / "gi.pdf")
    fig, corr["CASCADE"][div], rmse["CASCADE"][div] = cmp_scatter(cascade, "CASCADE")
    fig.savefig(paths["cascade"].parent / "gi.pdf")

# %% [markdown]
# # Summarizing plot

# %%
with open("../../evaluation/config/display.yaml") as f:
    palette = yaml.load(f, Loader=yaml.Loader)["palette"]["methods"]

# %%
df = pd.merge(
    pd.DataFrame(corr)
    .reset_index()
    .rename(columns={"index": "div"})
    .melt(id_vars="div", var_name="method", value_name="Corr"),
    pd.DataFrame(rmse)
    .reset_index()
    .rename(columns={"index": "div"})
    .melt(id_vars="div", var_name="method", value_name="RMSE"),
)
df.head()

# %%
fig, ax = plt.subplots(figsize=(3, 4))
ax = sns.boxplot(
    data=df,
    x="method",
    y="Corr",
    hue="method",
    showmeans=True,
    palette=palette,
    width=0.75,
    flierprops={
        "marker": ".",
        "markerfacecolor": "black",
        "markeredgecolor": "none",
        "markersize": 5,
    },
    meanprops={
        "marker": "^",
        "markerfacecolor": "#EEEEEE",
        "markeredgecolor": "black",
        "markersize": 5,
    },
)
plt.xticks(rotation=90)
ax.set_xlabel("Method")
ax.set_ylabel("Non-additive residual correlation")
fig.savefig("corr.pdf")

# %%
fig, ax = plt.subplots(figsize=(3, 4))
ax = sns.boxplot(
    data=df,
    x="method",
    y="RMSE",
    hue="method",
    showmeans=True,
    palette=palette,
    width=0.75,
    flierprops={
        "marker": ".",
        "markerfacecolor": "black",
        "markeredgecolor": "none",
        "markersize": 5,
    },
    meanprops={
        "marker": "^",
        "markerfacecolor": "#EEEEEE",
        "markeredgecolor": "black",
        "markersize": 5,
    },
)
plt.xticks(rotation=90)
ax.set_xlabel("Method")
ax.set_ylabel("Non-additive residual RMSE")
fig.savefig("rmse.pdf")
