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
from glob import glob
from os.path import basename

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from cascade.data import configure_dataset, encode_regime
from cascade.model import CASCADE
from cascade.nn import IntervDesign
from cascade.plot import set_figure_params
from cascade.utils import densify

# %%
set_figure_params()

# %% [markdown]
# # Read data

# %%
with open("config/display.yaml") as f:
    display = yaml.load(f, Loader=yaml.Loader)
meth_lut = {v: k for k, v in display["naming"]["methods"].items()}


# %%
def examine(ctrl, target, true, model, mod=None):
    tidx = rnd.choice(ctrl.n_obs, target.n_obs)
    prep = ctrl[tidx].copy()
    prep.obs_names_make_unique()
    prep.obsm["covariate"] = target.obsm["covariate"]
    prep.obs["interv"] = ",".join(true.split("+"))
    encode_regime(prep, "interv", key="interv")
    configure_dataset(
        prep,
        use_regime="interv",
        use_covariate="covariate",
        use_size="ncounts",
        use_layer="counts",
    )
    ctfact = model.counterfactual(prep, design=mod)
    ctfact.X = np.log1p(ctfact.X * (1e4 / ctfact.obs[["ncounts"]].to_numpy()))

    ctrl_vals = densify(ctrl[:, true.split("+")].X).mean(axis=0)
    target_vals = densify(target[:, true.split("+")].X).mean(axis=0)
    ctfact_vals = ctfact[:, true.split("+")].X.mean(axis=0)
    return ctrl_vals, target_vals, ctfact_vals


# %%
dsgn_dict = defaultdict(list)
rnd = np.random.RandomState(0)
for ds in tqdm(
    [
        "Adamson-2016",
        "Replogle-2022-RPE1",
        "Replogle-2022-K562-ess",
        "Replogle-2022-K562-gwps",
        "Norman-2019",
    ]
):
    for div in tqdm(range(5)):
        model_path = (
            f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
            f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
            f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
            f"lam=0.1-alp=0.5-run_sd=0-tune_ct=True"
        )
        ctrl = ad.read_h5ad(f"dat/ds={ds}/imp=20/n_vars=2000/ctrl.h5ad")
        model = CASCADE.load(f"{model_path}/tune.pt")
        for dsgn_path in glob(f"{model_path}-dsgn=bf/dsgn_test/*"):
            true = basename(dsgn_path)
            target = ad.read_h5ad(
                f"dat/ds={ds}/imp=20/n_vars=2000/kg=0.9-kc=0.75-div_sd={div}/"
                f"dsgn_test/{true}.h5ad"
            )
            dsgn = pd.read_csv(
                f"{dsgn_path}/dsgn.csv", index_col=0, keep_default_na=False
            )
            rank = dsgn["votes"].rank(ascending=False)
            qtl = rank.loc[",".join(true.split("+"))] / rank.shape[0]
            ctrl_vals, target_vals, ctfact_vals = examine(ctrl, target, true, model)

            dsgn_dict["ds"].append(ds)
            dsgn_dict["div"].append(div)
            dsgn_dict["meth"].append("cascade_bf")
            dsgn_dict["true"].append(true)
            dsgn_dict["ctrl"].append(ctrl_vals)
            dsgn_dict["target"].append(target_vals)
            dsgn_dict["ctfact"].append(ctfact_vals)
            dsgn_dict["qtl"].append(qtl)
        for dsgn_path in glob(f"{model_path}-dsgn=sb/dsgn_test/*"):
            true = basename(dsgn_path)
            target = ad.read_h5ad(
                f"dat/ds={ds}/imp=20/n_vars=2000/kg=0.9-kc=0.75-div_sd={div}/"
                f"dsgn_test/{true}.h5ad"
            )
            mod = IntervDesign.load(f"{dsgn_path}/design.pt")
            dsgn = pd.read_csv(
                f"{dsgn_path}/dsgn.csv", index_col=0, keep_default_na=False
            )
            rank = dsgn["score"].rank(ascending=False)
            qtl = rank.loc[",".join(true.split("+"))] / rank.shape[0]
            ctrl_vals, target_vals, ctfact_vals = examine(
                ctrl, target, true, model, mod=mod
            )

            dsgn_dict["ds"].append(ds)
            dsgn_dict["div"].append(div)
            dsgn_dict["meth"].append("cascade_sb")
            dsgn_dict["true"].append(true)
            dsgn_dict["ctrl"].append(ctrl_vals)
            dsgn_dict["target"].append(target_vals)
            dsgn_dict["ctfact"].append(ctfact_vals)
            dsgn_dict["qtl"].append(qtl)

# %%
summary_df = pd.DataFrame(dsgn_dict)
summary_df["dev"] = (summary_df["ctfact"] - summary_df["target"]).map(
    np.linalg.norm
) / (summary_df["target"] - summary_df["ctrl"]).map(np.linalg.norm)
summary_df.to_csv("dsgn_sb.csv", index=False)
summary_df.head()

# %%
paired_df = pd.merge(
    summary_df.query("meth == 'cascade_bf'")
    .drop(columns=["meth"])
    .set_index(["ds", "div", "true"])
    .add_suffix("_bf"),
    summary_df.query("meth == 'cascade_sb'")
    .drop(columns=["meth"])
    .set_index(["ds", "div", "true"])
    .add_suffix("_sb"),
    left_index=True,
    right_index=True,
).reset_index()
paired_df["dev_diff"] = paired_df["dev_bf"] - paired_df["dev_sb"]
paired_df["qtl_diff"] = paired_df["qtl_bf"] - paired_df["qtl_sb"]
paired_df.head()

# %%
paired_df["dev_diff_bin"] = pd.cut(
    paired_df["dev_diff"], [-1, -0.2, -0.05, 0.05, 0.2, 1]
)
fig, ax = plt.subplots(figsize=(4, 4))
ax = sns.boxplot(
    data=paired_df,
    x="dev_diff_bin",
    y="qtl_diff",
    width=0.7,
    flierprops={
        "marker": ".",
        "markerfacecolor": "black",
        "markeredgecolor": "none",
        "markersize": 5,
    },
    ax=ax,
)
plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
ax.axhline(y=0, c="darkred", ls="--")
ax.set_xlabel("Improvement in intervention deviation\n(optim vs exhaust)")
ax.set_ylabel("Improvement in design quantile\n(optim vs exhaust)")
fig.savefig("dsgn_sb.pdf")
