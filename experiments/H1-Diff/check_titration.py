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
from pathlib import Path
from subprocess import run

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from cascade.data import Targets, configure_dataset, encode_regime
from cascade.model import CASCADE, IntervDesign
from cascade.plot import set_figure_params
from cascade.utils import is_notebook

# %%
set_figure_params()

# %% [markdown]
# # Parametrize

# %%
if is_notebook():
    ct = "Retinal_progenitors_and_Muller_glia"
    pert = "SIX3"
    size = len(pert.split(","))
    model_path = Path(
        "model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.1-alp=0.5-run_sd=1"
    )
    design_path = model_path / "design" / f"target={ct}-size={size}"
    args = Namespace(
        ctrl=Path("ctrl.h5ad"),
        data=Path("adata.h5ad"),
        target=Path(f"targets/{ct}.h5ad"),
        markers=Path("markers.yaml"),
        model=model_path / "tune.pt",
        scaffold=Path("scaffold.gml.gz"),
        discover=model_path / "discover.gml.gz",
        design=design_path / "design.pt",
        pert=pert,
        volcano=design_path / f"volcano_{pert}.pdf",
        gsea_dir=design_path / f"gsea_{pert}",
        explain=design_path / f"explain_{pert}.gml",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--discover", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--pert", type=str, required=True)
    parser.add_argument("--volcano", type=Path, required=True)
    parser.add_argument("--gsea-dir", type=Path, default=None)
    parser.add_argument("--explain", type=Path, required=True)
    args = parser.parse_args()
vars(args)

# %%
args.design_path = args.design.parent

# %% [markdown]
# # Read data

# %%
ctrl = ad.read_h5ad(args.ctrl)
ctrl

# %%
adata = ad.read_h5ad(args.data)
adata

# %%
target = ad.read_h5ad(args.target)
target

# %%
with open(args.markers) as f:
    markers = set(yaml.load(f, Loader=yaml.Loader)[ct.replace("_", " ")])
    target_ctrl_diff = target.to_df().mean() - ctrl.to_df().mean()
    markers = [i for i in markers if target_ctrl_diff.get(i, 0) > 0]
print(f"n_markers = {len(markers)}")

# %%
model = CASCADE.load(args.model)
design = IntervDesign.load(args.design)

# %%
pert_set = Targets(args.pert)
pert_list = sorted(pert_set)
pert_set, pert_list

# %% [markdown]
# # Counterfactual prediction

# %%
n_obs = 1000
rnd = np.random.RandomState(0)
ctrl_idx = rnd.choice(ctrl.n_obs, n_obs, replace=True)
target_idx = rnd.choice(target.n_obs, n_obs, replace=True)

# %%
prep = ctrl[ctrl_idx].copy()
prep.obs_names_make_unique()

# %%
prep.obsm["covariate"] = target.obsm["covariate"][target_idx]
prep.obs["knockup"] = args.pert

# %% [markdown]
# ## Covariate only

# %%
encode_regime(prep, "interv")
configure_dataset(
    prep,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)

# %%
ctfact_cov = model.counterfactual(prep, design=design, sample=True)
ctfact_cov.X = np.log1p(ctfact_cov.X * (1e4 / ctfact_cov.obs[["ncounts"]].to_numpy()))

# %% [markdown]
# ## Covariate + knockup

# %%
encode_regime(prep, "interv", key="knockup")
configure_dataset(
    prep,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)

# %%
ctfact_design = model.counterfactual(prep, design=design, sample=True)
ctfact_design.X = np.log1p(
    ctfact_design.X * (1e4 / ctfact_design.obs[["ncounts"]].to_numpy())
)

# %% [markdown]
# # Prepare input datasets of different fold changes

# %%
target_tp10k = np.expm1(target[:, pert_list].X).mean(axis=0).A1
target_tp10k

# %%
nil_tp10k = np.expm1(ctrl[:, pert_list].X).mean(axis=0).A1
nil_tp10k

# %%
design_tp10k = np.expm1(ctfact_design[:, pert_list].X).mean(axis=0)
design_tp10k

# %%
steps = np.linspace(0, 5, 21).round(1)
steps

# %%
for step in tqdm(steps):
    prep_nil_fix = prep.copy()
    prep_nil_fix.obs["knockup"] = ""
    prep_design_fix = prep.copy()
    fix_tp10k = np.expm1(prep_design_fix[:, pert_list].X).mean(axis=0).A1
    prep_design_fix.layers["counts"][
        :, prep_design_fix.var_names.get_indexer(pert_list)
    ] = (fix_tp10k + step * (design_tp10k - fix_tp10k)) * (
        prep_design_fix.obs[["ncounts"]].to_numpy() / 1e4
    )
    prep_fix = ad.concat([prep_nil_fix, prep_design_fix], merge="first")
    prep_fix.obs_names_make_unique(join=":")
    prep_fix.write(
        args.design_path / f"prep-{args.pert}-step={step}.h5ad", compression="gzip"
    )

# %% [markdown]
# # Run step-wise counterfactual inference

# %%
for step in tqdm(steps):
    run(
        [
            "cascade",
            "counterfactual",
            "-d",
            args.design_path / f"prep-{args.pert}-step={step}.h5ad",
            "-m",
            args.model,
            "-u",
            args.design,
            "-p",
            args.design_path / f"ctfact-{args.pert}-step={step}.h5ad",
            "--interv-key",
            "knockup",
            "--use-covariate",
            "covariate",
            "--use-size",
            "ncounts",
            "--use-layer",
            "counts",
            "--fixed-genes",
            args.pert,
            "--batch-size",
            "128",
        ]
    )

# %%
ctfact_dict = {}
for step in tqdm(steps):
    ctfact_dict[step] = ad.read_h5ad(
        args.design_path / f"ctfact-{args.pert}-step={step}.h5ad"
    )
    ctfact_dict[step].X = np.log1p(
        ctfact_dict[step].X * (1e4 / ctfact_dict[step].obs[["ncounts"]].to_numpy())
    )

# %% [markdown]
# # Summarize results

# %%
ctrl_mean = pd.Series(ctrl.X.mean(axis=0).A1, index=ctrl.var_names)
actual_mean = pd.Series(
    adata[(adata.obs["dataset"] == "Joung-2023") & adata.obs["knockup"].isin(pert_set)]
    .X.mean(axis=0)
    .A1,
    index=adata.var_names,
)
nil_mean = pd.Series(
    ctfact_dict[step][ctfact_dict[step].obs["knockup"] == ""].X.mean(axis=0),
    index=ctfact_dict[step].var_names,
)  # Constant across steps
ctfact_dict_mean = {
    step: pd.Series(
        ctfact_dict[step][ctfact_dict[step].obs["knockup"] == args.pert].X.mean(axis=0),
        index=ctfact_dict[step].var_names,
    )
    for step in steps
}
target_mean = pd.Series(target.X.mean(axis=0).A1, index=target.var_names)

# %%
cmp = pd.DataFrame.from_dict(
    {
        "Control": ctrl_mean,
        "Actual": actual_mean,
        "Nil": np.maximum(nil_mean, 0),
        "Target": target_mean,
        "Weight": target.var["weight"],
    }
    | {f"Design (step={step})": np.maximum(ctfact_dict_mean[step], 0) for step in steps}
)
cmp.tail()


# %%
def weighted_mse(x, y, weight):
    weight = weight.size * weight / weight.sum()
    return (np.square(x - y) * weight).mean()


# %%
mse_df = []
for step in steps:
    mse_df.append(
        {
            "step": step,
            "nil_mse": weighted_mse(cmp["Target"], cmp["Nil"], cmp["Weight"]),
            "design_mse": weighted_mse(
                cmp["Target"], cmp[f"Design (step={step})"], cmp["Weight"]
            ),
        }
    )
mse_df = pd.DataFrame.from_records(mse_df)
mse_df["exp_pct"] = (
    100 * (mse_df["design_mse"] - mse_df["nil_mse"]) / (mse_df["nil_mse"])
)

tpm_df = []
for step in steps:
    tpm_df.append(
        {"step": step}
        | {g: 100 * np.expm1(cmp[f"Design (step={step})"][g]) for g in pert_set}
    )
tpm_df = pd.DataFrame.from_records(tpm_df)

fc_df = []
for step in steps:
    fc_df.append(
        {"step": step}
        | {
            g: np.expm1(cmp[f"Design (step={step})"][g]) / np.expm1(cmp["Nil"][g])
            for g in pert_set
        }
    )
fc_df = pd.DataFrame.from_records(fc_df)

# %%
tpm_df = tpm_df.melt(id_vars="step", var_name="Gene", value_name="tpm")
fc_df = fc_df.melt(id_vars="step", var_name="Gene", value_name="fc")
combined_df = pd.merge(mse_df, pd.merge(tpm_df, fc_df))
combined_df.head()

# %% [markdown]
# # Plotting

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3.5), gridspec_kw={"wspace": 0.45})
ax1 = sns.lineplot(data=combined_df, x="tpm", y="exp_pct", hue="Gene", ax=ax1)
ax1 = sns.scatterplot(
    data=combined_df, x="tpm", y="exp_pct", hue="Gene", legend=False, ax=ax1
)
ax1.set_xlabel("TPM")
ax1.set_ylabel("Explained MSE%")
ax2 = sns.lineplot(data=combined_df, x="fc", y="exp_pct", hue="Gene", ax=ax2)
ax2 = sns.scatterplot(
    data=combined_df, x="fc", y="exp_pct", hue="Gene", legend=False, ax=ax2
)
ax2.set_xlabel("Fold change")
ax2.set_ylabel("Explained MSE%")
fig.savefig(args.design_path / f"titration-{args.pert}.pdf")

# %% [markdown]
# # Intervened genes

# %%
cmp.loc[pert_list]

# %% [markdown]
# # Responsive genes

# %%
responsive_genes = ["PAX6", "RAX", "LHX2"]
cmp.loc[responsive_genes]

# %%
response_df = []
for step in steps:
    response_df.append(
        {"step": step}
        | {
            g: 100 * np.expm1(cmp[f"Design (step={step})"][g])
            for g in [*responsive_genes, *pert_list]
        }
    )
response_df = pd.DataFrame.from_records(response_df)
response_df = response_df.melt(
    id_vars=["step", pert_list[0]], var_name="gene", value_name="TPM"
)

# %%
fig, ax = plt.subplots()
ax = sns.lineplot(data=response_df, x=pert_list[0], y="TPM", hue="gene", ax=ax)

# %% [markdown]
# # qPCR genes

# %%
qPCR_genes = ["VSX2", "PAX6", "SIX6", "SLC1A3", "SOX9"]
cmp.loc[qPCR_genes]

# %%
qPCR_df = []
for step in steps:
    qPCR_df.append(
        {"step": step}
        | {
            g: 100 * np.expm1(cmp[f"Design (step={step})"][g])
            for g in [*qPCR_genes, *pert_list]
        }
    )
qPCR_df = pd.DataFrame.from_records(qPCR_df)
qPCR_df = qPCR_df.melt(
    id_vars=["step", pert_list[0]], var_name="gene", value_name="TPM"
)

# %%
fig, ax = plt.subplots()
ax = sns.lineplot(data=qPCR_df, x=pert_list[0], y="TPM", hue="gene", ax=ax)

# %% [markdown]
# # Apoptotic genes

# %%
# fmt: off
apoptotic_genes = [
    "BIRC3", "ACTB", "CASP6", "PRF1", "PIDD1", "TUBA3E", "CHUK", "DAB2IP",
    "GADD45A", "TRAF1", "PIK3R1", "CASP8", "KRAS", "CTSH", "CFLAR", "PIK3R2",
    "LMNB1", "FASLG", "EIF2AK3", "CASP3", "TNF", "CAPN1", "HTRA2", "ENDOG",
    "EIF2S1", "ATF4", "DAXX", "TRADD", "AKT2", "NGF", "AIFM1", "TNFRSF10A",
    "BID", "NFKBIA", "PIK3R3", "IKBKB", "RELA", "NFKB1", "CTSS", "CTSO",
    "MAP2K1", "CASP10", "PTPN13", "CTSD", "SPTA1", "TUBA4A", "LMNB2", "IL3",
    "PARP4", "PIK3CB"
]
# fmt: on
apoptotic_genes = [g for g in apoptotic_genes if g in ctrl.var_names]
len(apoptotic_genes)

# %%
cmp.loc[apoptotic_genes].mean()

# %%
