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
from pathlib import Path

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

from cascade.data import aggregate_obs
from cascade.model import CASCADE
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
        "discover": Path(
            f"../../evaluation/inf/{dat_pat}/kegg+tf+ppi+corr/lsi/{div_pat}/"
            f"cascade/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
            f"lam=0.1-alp=0.5-run_sd=0/discover.gml.gz"
        ),
        "model": Path(
            f"../../evaluation/inf/{dat_pat}/kegg+tf+ppi+corr/lsi/{div_pat}/"
            f"cascade/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
            f"lam=0.1-alp=0.5-run_sd=0-tune_ct=True/tune.pt"
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
        train,
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


# %% [markdown]
# # Read data

# %%
div = 4
paths = build_paths(div)

# %%
(
    true,
    train,
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

# %%
discover = nx.read_gml(
    f"../../evaluation/inf/ds=Norman-2019/imp=20/n_vars=2000/kegg+tf+ppi+corr/"
    f"lsi/kg=0.9-kc=0.75-div_sd={div}/cascade/nptc=4-dz=16-beta=0.1-"
    f"sps=L1-acyc=SpecNorm-lik=NegBin-lam=0.1-alp=0.5-run_sd=0/discover.gml.gz"
)

# %%
model = CASCADE.load(
    f"../../evaluation/inf/ds=Norman-2019/imp=20/n_vars=2000/kegg+tf+ppi+corr/"
    f"lsi/kg=0.9-kc=0.75-div_sd={div}/cascade/nptc=4-dz=16-beta=0.1-"
    f"sps=L1-acyc=SpecNorm-lik=NegBin-lam=0.1-alp=0.5-run_sd=0-tune_ct=True/"
    f"tune.pt"
)

# %% [markdown]
# # Link non-additive effect with causal graph

# %%
additive_melt = (
    (additive - ctrl)
    .reset_index()
    .rename(columns={"index": "knockup"})
    .melt(id_vars=["knockup"], var_name="response", value_name="additive")
)
residual_melt = (
    (true - additive)
    .reset_index()
    .rename(columns={"index": "knockup"})
    .melt(id_vars=["knockup"], var_name="response", value_name="residual")
)
error_melt = (
    (cascade - true)
    .reset_index()
    .rename(columns={"index": "knockup"})
    .melt(id_vars=["knockup"], var_name="response", value_name="error")
)
df = additive_melt.merge(residual_melt).merge(error_melt)
df["additive_sign"] = np.sign(df["additive"])
df["residual_sign"] = np.sign(df["residual"])
df

# %%
df_non_additive = df.query(
    "(residual > 0.5 | residual < -0.5) & "
    "(additive > 0.5 | additive < -0.5) & "
    "residual_sign == additive_sign"
).copy()
df_non_additive.shape[0]

# %%
df_additive = df.query(
    "(residual < 0.01 & residual > -0.01) & (additive > 0.5 | additive < -0.5)"
).copy()
df_additive.shape[0]

# %%
connectivity = []
for _, row in df_non_additive.iterrows():
    (i, j), k = row["knockup"].split(","), row["response"]
    connectivity.append(nx.has_path(discover, i, k) and nx.has_path(discover, j, k))
df_non_additive["connectivity"] = connectivity
df_non_additive["connectivity"].mean()

# %%
connectivity = []
for _, row in df_additive.iterrows():
    (i, j), k = row["knockup"].split(","), row["response"]
    connectivity.append(nx.has_path(discover, i, k) and nx.has_path(discover, j, k))
df_additive["connectivity"] = connectivity
df_additive["connectivity"].mean()

# %% [markdown]
# # Identify significant and well-predicted non-additive effects

# %%
candidates = df.query(
    "(residual > 0.2 | residual < -0.2) & "
    "(additive > 0.1 | additive < -0.1) & "
    "(error < 0.1 & error > -0.1) & "
    "residual_sign == additive_sign"
).sort_values("knockup")
candidates

# %%
candidates.to_csv("candidates.csv", index=False)
