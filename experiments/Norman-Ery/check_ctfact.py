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

# %% editable=true slideshow={"slide_type": ""}
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib import rcParams

from cascade.data import Targets
from cascade.plot import pair_grid, set_figure_params
from cascade.utils import config, is_notebook

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()
rcParams["axes.grid"] = True

# %% [markdown]
# # Parametrize

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
if is_notebook():
    design_dir = Path(
        "model/nptc=4-dz=8-drop=0.2-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.01-dec=0.01-run_sd=0/design/size=3"
    )
    args = Namespace(
        markers=Path("markers.yaml"),
        data=Path("norman.h5ad"),
        ctrl=Path("ctrl.h5ad"),
        target=Path("target.h5ad"),
        ctfact=design_dir / "ctfact.h5ad",
        expr_scatter=design_dir / "expr_scatter.pdf",
        logfc_scatter=design_dir / "logfc_scatter.pdf",
        logfc_violin=design_dir / "logfc_violin.pdf",
        gsea_dir=design_dir / "gsea",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--ctfact", type=Path, required=True)
    parser.add_argument("--expr-scatter", type=Path, required=True)
    parser.add_argument("--logfc-scatter", type=Path, required=True)
    parser.add_argument("--logfc-violin", type=Path, required=True)
    parser.add_argument("--gsea-dir", type=Path, required=True)
    args = parser.parse_args()
vars(args)

# %% [markdown]
# # Read data

# %%
with args.markers.open("r") as f:
    markers = set(yaml.load(f, Loader=yaml.Loader)["Ery"])
len(markers)

# %% editable=true slideshow={"slide_type": ""}
adata = ad.read_h5ad(args.data)
ctrl = ad.read_h5ad(args.ctrl)
target = ad.read_h5ad(args.target)
ctfact = ad.read_h5ad(args.ctfact)
if "NegBin" in args.ctfact.as_posix():
    ctfact.X = np.log1p(ctfact.X * (1e4 / ctfact.obs[["ncounts"]].to_numpy()))

# %%
ctfact_unique = ctfact.obs.loc[ctfact.obs["prep"] == "design", "knockup"].unique()
assert ctfact_unique.size == 1
design = ctfact_unique[0]
design_set = Targets(design.split(","))
design_set

# %%
ctrl_mean = pd.Series(ctrl.X.mean(axis=0).A1, index=ctrl.var_names)
actual_mean = pd.Series(
    adata[adata.obs["knockup"].isin(design_set)].X.mean(axis=0).A1,
    index=adata.var_names,
)
nil_mean = pd.Series(
    ctfact[ctfact.obs["prep"] == "nil"].X.mean(axis=0), index=ctfact.var_names
)
ctfact_mean = pd.Series(
    ctfact[ctfact.obs["prep"] == "design"].X.mean(axis=0), index=ctfact.var_names
)
target_mean = pd.Series(target.X.mean(axis=0).A1, index=target.var_names)

# %% [markdown]
# # Visualization

# %%
cmp = pd.DataFrame.from_dict(
    {
        "Control": ctrl_mean,
        "Actual": actual_mean,
        "Nil": np.maximum(nil_mean, 0),
        "Design": np.maximum(ctfact_mean, 0),
        "Target": target_mean,
        "Weight": target.var["weight"],
        "Role": "Others",
    }
)
cmp["Actual logFC"] = np.log((cmp["Actual"] + 0.1) / (cmp["Control"] + 0.1))
cmp["Design logFC"] = np.log((cmp["Design"] + 0.1) / (cmp["Nil"] + 0.1))
cmp["Target logFC"] = np.log((cmp["Target"] + 0.1) / (cmp["Nil"] + 0.1))
cmp.loc[cmp.index.isin(markers) & (cmp["Control"] < cmp["Target"]), "Role"] = "Marker"
cmp.loc[cmp.index.isin(design_set), "Role"] = "Design"
cmp["Role"] = pd.Categorical(cmp["Role"], categories=["Others", "Marker", "Design"])
cmp = cmp.sort_values("Role")
cmp.tail()

# %%
palette = {"Others": "#D3D3D3", "Marker": "#ff7e28", "Design": "#1f77b4"}

# %% [markdown]
# ## Absolute expression value


# %%
def annotate_outlier(x, y, color=None, label=None, hue=None, **kwargs):
    logfc = np.log((x + 0.1) / (y + 0.1))
    df = pd.DataFrame({"x": x, "y": y, "logfc": logfc, "hue": hue})
    marker_df = df.query("hue == 'Marker'").sort_values("logfc")
    design_df = df.query("hue == 'Design'")
    df = pd.concat([marker_df.head(5), marker_df.tail(5), design_df])
    adjust_text(
        [
            plt.text(row["x"], row["y"], index, fontsize="xx-small")
            for index, row in df.iterrows()
        ],
        arrowprops={"arrowstyle": "-"},
        min_arrow_len=5,
    )


g = pair_grid(
    cmp,
    vars=["Control", "Actual", "Nil", "Design", "Target"],
    hue="Role",
    palette=palette,
    weight="Weight",
    scatter_kws={"rasterized": True},
    hist_kws={"bins": 20},
)
g.map_lower(annotate_outlier)
g.savefig(args.expr_scatter)


# %% [markdown]
# ## logFC


# %%
def text_label(x, y, color=None, label=None, hue=None, **kwargs):
    df = pd.DataFrame({"x": x, "y": y, "hue": hue}).query("hue != 'Others'")
    adjust_text(
        [
            plt.text(row["x"], row["y"], index, fontsize="xx-small")
            for index, row in df.iterrows()
        ],
        arrowprops={"arrowstyle": "-"},
        min_arrow_len=5,
    )


g = pair_grid(
    cmp,
    vars=["Actual logFC", "Target logFC", "Design logFC"],
    hue="Role",
    palette=palette,
    weight="Weight",
    scatter_kws={"rasterized": True},
    hist_kws={"bins": 20},
    height=3,
)
g.map_lower(text_label)
g.savefig(args.logfc_scatter)

# %%
cmp_melt = (
    cmp[["Role", "Actual logFC", "Design logFC", "Target logFC"]]
    .reset_index()
    .rename(
        columns={
            "Actual logFC": "Actual vs Ctrl",
            "Design logFC": "Design vs Ctrl",
            "Target logFC": "Target vs Ctrl",
        }
    )
    .melt(id_vars=["index", "Role"], var_name="Comparison", value_name="logFC")
)

g = sns.FacetGrid(
    cmp_melt.query("Role != 'Design'"),
    col="Comparison",
    hue="Role",
    palette=palette,
    sharey=False,
)
g.map(sns.violinplot, "Role", "logFC", order=["Marker", "Others"])
g.savefig(args.logfc_violin)


# %% [markdown]
# ## GSEA

# %%
if args.gsea_dir.exists():
    shutil.rmtree(args.gsea_dir)

# %%
(args.gsea_dir / "c2").mkdir(parents=True, exist_ok=True)
prerank = gp.prerank(
    rnk=cmp["Design logFC"].sort_values(),
    gene_sets="../../data/function/MSigDB/c2.cp.v2023.2.Hs.symbols.gmt",
    outdir=args.gsea_dir / "c2",
    threads=8,
    seed=0,
)

# %%
(args.gsea_dir / "c5").mkdir(parents=True, exist_ok=True)
prerank = gp.prerank(
    rnk=cmp["Design logFC"].sort_values(),
    gene_sets="../../data/function/MSigDB/c5.go.bp.v2023.2.Hs.symbols.gmt",
    outdir=args.gsea_dir / "c5",
    threads=8,
    seed=0,
)

# %%
(args.gsea_dir / "c8").mkdir(parents=True, exist_ok=True)
prerank = gp.prerank(
    rnk=cmp["Design logFC"].sort_values(),
    gene_sets="../../data/function/MSigDB/c8.all.v2023.2.Hs.symbols.gmt",
    outdir=args.gsea_dir / "c8",
    threads=8,
    seed=0,
)
