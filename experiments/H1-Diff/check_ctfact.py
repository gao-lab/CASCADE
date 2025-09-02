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
import gc
from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm

from cascade.data import aggregate_obs
from cascade.plot import pair_grid, set_figure_params
from cascade.utils import config, is_notebook

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()
plt.ioff()

# %% [markdown]
# # Parametrize

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
if is_notebook():
    ct = "Astrocytes"
    size = 1
    design_dir = Path(
        f"model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        f"lam=0.1-alp=0.5-run_sd=1/design/target={ct}-size={size}"
    )
    args = Namespace(
        markers=Path("markers.yaml"),
        ctrl=Path("ctrl.h5ad"),
        target=Path(f"targets/{ct}.h5ad"),
        design=design_dir / "design.csv",
        ctfact=design_dir / "ctfact.h5ad",
        top_designs=3,
        explained=design_dir / "explained.csv",
        expr_scatter=design_dir / "expr_scatter.pdf",
        logfc_scatter=design_dir / "logfc_scatter.pdf",
        logfc_violin=design_dir / "logfc_violin.pdf",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--ctrl", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--ctfact", type=Path, required=True)
    parser.add_argument("--top-designs", type=int, default=10)
    parser.add_argument("--explained", type=Path, required=True)
    parser.add_argument("--expr-scatter", type=Path, required=True)
    parser.add_argument("--logfc-scatter", type=Path, required=True)
    parser.add_argument("--logfc-violin", type=Path, required=True)
    args = parser.parse_args()
vars(args)

# %% [markdown]
# # Read data

# %%
ct = args.target.stem
ct

# %%
with args.markers.open("r") as f:
    markers = set(yaml.load(f, Loader=yaml.Loader)[ct.replace("_", " ")])
len(markers)

# %% editable=true slideshow={"slide_type": ""}
ctrl = ad.read_h5ad(args.ctrl)
target = ad.read_h5ad(args.target)
ctfact = ad.read_h5ad(args.ctfact)
if "NegBin" in args.ctfact.as_posix():
    ctfact.X = np.log1p(ctfact.X * (1e4 / ctfact.obs[["ncounts"]].to_numpy()))

# %%
design = pd.read_csv(args.design, index_col=0, nrows=100, keep_default_na=False)
design.head()

# %%
design_used = design

# %%
cmp = (
    aggregate_obs(ctfact, "knockup", X_agg="mean")
    .to_df()
    .transpose()
    .assign(
        Control=pd.Series(ctrl.X.mean(axis=0).A1, index=ctrl.var_names),
        Target=pd.Series(target.X.mean(axis=0).A1, index=target.var_names),
        Weight=target.var["weight"],
    )
)
cmp.head()

# %%
del ctrl, target, ctfact
gc.collect()

# %% [markdown]
# # Visualizations

# %%
palette = {"Others": "#D3D3D3", "Marker": "#ff7e28", "Design": "#1f77b4"}


def extract_combo(cmp, combo):
    split = combo.split(",")
    size = len(split)
    additive = cmp[split].sum(axis=1) - cmp[""] * (size - 1)
    cmp_abs = cmp[["", *split, combo, "Target", "Weight"]].copy()
    cmp_abs = cmp_abs.rename(columns={"": "None"})
    cmp_abs.insert(size + 1, "Additive", additive)
    cmp_abs["Role"] = "Others"
    cmp_abs.loc[
        cmp_abs.index.isin(markers) & (cmp["Control"] < cmp["Target"]),
        "Role",
    ] = "Marker"
    cmp_abs.loc[cmp_abs.index.isin(split), "Role"] = "Design"
    cmp_logfc = cmp_abs[["None", *split, "Additive", combo, "Target"]].sub(
        cmp_abs["None"], axis="index"
    )
    cmp_logfc = pd.concat([cmp_logfc, cmp_abs[["Weight", "Role"]]], axis=1)
    cmp_abs = cmp_abs.loc[:, ~cmp_abs.columns.duplicated()]  # Single may dup
    cmp_logfc = cmp_logfc.loc[:, ~cmp_logfc.columns.duplicated()]  # Single may dup
    return cmp_abs, cmp_logfc


def weighted_mse(x, y, weight):
    weight = weight.size * weight / weight.sum()
    return (np.square(x - y) * weight).mean().item()


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


# %%
with PdfPages(args.expr_scatter) as pdf:
    for combo in tqdm(design.index[: args.top_designs]):
        cmp_abs, _ = extract_combo(cmp, combo)
        split = combo.split(",")
        use_vars = [*split, "Additive", combo] if len(split) > 1 else [combo]
        g = pair_grid(
            cmp_abs,
            vars=["None", *use_vars, "Target"],
            hue="Role",
            palette=palette,
            weight="Weight",
            scatter_kws={"rasterized": True},
            hist_kws={"bins": 20},
        )
        g.map_lower(text_label)
        g.fig.suptitle(ct, y=1.01)
        pdf.savefig(g.fig)
        plt.close()
        gc.collect()

# %%
with PdfPages(args.logfc_scatter) as pdf:
    for combo in tqdm(design.index[: args.top_designs]):
        _, cmp_logfc = extract_combo(cmp, combo)
        split = combo.split(",")
        use_vars = [*split, "Additive", combo] if len(split) > 1 else [combo]
        g = pair_grid(
            cmp_logfc,
            vars=[*use_vars, "Target"],
            hue="Role",
            palette=palette,
            weight="Weight",
            scatter_kws={"rasterized": True},
            hist_kws={"bins": 20},
        )
        g.map_lower(text_label)
        g.fig.suptitle(ct, y=1.01)
        pdf.savefig(g.fig)
        plt.close()
        gc.collect()

# %%
with PdfPages(args.logfc_violin) as pdf:
    for combo in tqdm(design.index[: args.top_designs]):
        _, cmp_logfc = extract_combo(cmp, combo)
        split = combo.split(",")
        use_vars = [*split, "Additive", combo] if len(split) > 1 else [combo]
        g = sns.FacetGrid(
            cmp_logfc[[*use_vars, "Target", "Role"]]
            .reset_index()
            .melt(id_vars=["index", "Role"], var_name="Condition", value_name="logFC")
            .query("Role != 'Design'"),
            col="Condition",
            hue="Role",
            palette=palette,
        )
        g.map(sns.violinplot, "Role", "logFC", order=["Marker", "Others"])
        g.fig.suptitle(ct, y=1.05)
        pdf.savefig(g.fig)
        plt.close()
        gc.collect()

# %%
explained = {}
for combo in tqdm(design.index[: args.top_designs]):
    cmp_abs, _ = extract_combo(cmp, combo)
    split = combo.split(",")
    mse = {
        item: weighted_mse(cmp_abs[item], cmp_abs["Target"], cmp_abs["Weight"])
        for item in ["None", *split, "Additive", combo]
    }
    exp = {
        "individual_exp": tuple(
            (mse["None"] - mse[item]) / mse["None"] for item in split
        ),
        "additive_exp": (mse["None"] - mse["Additive"]) / mse["None"],
        "combo_exp": (mse["None"] - mse[combo]) / mse["None"],
        "synergy_exp": (mse["Additive"] - mse[combo]) / mse["None"],
    }
    explained[combo] = exp

explained = pd.DataFrame.from_dict(explained, orient="index")
explained.to_csv(args.explained)
explained
