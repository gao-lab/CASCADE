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

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import leaves_list, linkage

from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Read perturbation choices

# %%
with open("choice.yaml") as f:
    choice = yaml.load(f, Loader=yaml.Loader)

# %%
gsea_collection = {}
for ct, pert in choice["design"].items():
    size = len(pert.split(","))
    try:
        gsea_collection[ct] = (
            pd.read_csv(
                f"model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
                f"lam=0.1-alp=0.5-run_sd=1/design/target={ct}-size={size}/"
                f"gsea_{pert}/gseapy.gene_set.prerank.report.csv"
            )
            # .query("`FDR q-val` < 0.05 & NES > 0")
            .query("NES > 0")
            .assign(Target=ct)
            .sort_values("NES", ascending=False)
        )
    except FileNotFoundError:
        print(f"{ct} not found")
len(gsea_collection)

# %%
gsea_combined = pd.concat(gsea_collection.values())
gsea_combined["-log10 FDR"] = (-np.log10(gsea_combined["FDR q-val"])).clip(
    lower=0, upper=4.5
)  # Finite max is 4.48

# %%
with pd.option_context("display.max_colwidth", None):
    display(
        gsea_combined.loc[
            gsea_combined["Target"] == "Excitatory_neurons",
            ["Target", "Term", "NES", "FDR q-val"],
        ].head(20)
    )

# %% [markdown]
# # All terms

# %%
nes_mat = gsea_combined.pivot(index="Term", columns="Target", values="NES").fillna(0)
nes_mat.shape

# %%
row_link = linkage(nes_mat, method="ward")
row_leaves = leaves_list(row_link)
gsea_combined["Term_Display"] = pd.Categorical(
    gsea_combined["Term"], categories=nes_mat.index[row_leaves]
)

# %%
col_link = linkage(nes_mat.T, method="ward")
col_leaves = leaves_list(col_link)
gsea_combined["Target_Display"] = pd.Categorical(
    gsea_combined["Target"], categories=nes_mat.columns[col_leaves]
)
gsea_combined["Target_Display"] = gsea_combined["Target_Display"].cat.rename_categories(
    lambda x: f"{x.replace('_', ' ')} ({choice['design'][x]})"
)

# %%
gsea_combined["NES"] = gsea_combined["NES"].clip(lower=1.5)

# %%
fig, ax = plt.subplots(figsize=(25, 18))
ax = sns.scatterplot(
    data=gsea_combined.sort_values("NES"),
    x="Term_Display",
    y="Target_Display",
    size="NES",
    hue="-log10 FDR",
    palette="BuPu",
    edgecolor=None,
    sizes=(1, 300),
    ax=ax,
)
ax.margins(x=0.02, y=0.02)
locs, labels = ax.get_xticks(), ax.get_xticklabels()
ax.set_xticks(locs[::10])
ax.set_xticklabels(labels[::10], rotation=60, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell type signature gene set", fontsize="x-large", fontweight="bold")
ax.set_ylabel(
    "Target cell type (Representative design)", fontsize="x-large", fontweight="bold"
)
ax.legend(
    fontsize="large", handletextpad=1, loc="upper right", bbox_to_anchor=(-0.2, -0.05)
)
fig.savefig("overview_all.pdf")

# %% [markdown]
# # Selected terms (large)

# %%
term2target = defaultdict(list)
for k, v in choice["gsea"].items():
    term2target[v].append(k)

# %%
gsea_combined_choice = gsea_combined.loc[gsea_combined["Term"].isin(term2target)].copy()
gsea_combined_choice

# %%
nes_mat_choice = gsea_combined_choice.pivot(
    index="Term", columns="Target", values="NES"
).fillna(0)
nes_mat_choice.shape

# %%
row_link = linkage(nes_mat_choice, method="ward")
row_leaves = leaves_list(row_link)
gsea_combined_choice["Term_Display"] = pd.Categorical(
    gsea_combined_choice["Term"], categories=nes_mat_choice.index[row_leaves]
)

# %%
gsea_combined_choice["Target_Display"] = pd.Categorical(
    gsea_combined_choice["Target"],
    categories=[
        target
        for term in nes_mat_choice.index[row_leaves]
        for target in term2target[term]
    ],
)
gsea_combined_choice["Target_Display"] = gsea_combined_choice[
    "Target_Display"
].cat.rename_categories(lambda x: f"{x.replace('_', ' ')} ({choice['design'][x]})")

# %%
fig, ax = plt.subplots(figsize=(18, 18))
ax = sns.scatterplot(
    data=gsea_combined_choice.sort_values("NES"),
    x="Term_Display",
    y="Target_Display",
    size="NES",
    hue="-log10 FDR",
    palette="BuPu",
    edgecolor=None,
    sizes=(1, 350),
    ax=ax,
)
ax.margins(x=0.02, y=0.02)
locs, labels = ax.get_xticks(), ax.get_xticklabels()
ax.set_xticks(locs)
ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell type signature gene set", fontsize="x-large", fontweight="bold")
ax.set_ylabel(
    "Target cell type (Representative design)", fontsize="x-large", fontweight="bold"
)
ax.legend(
    fontsize="large", handletextpad=1, loc="upper right", bbox_to_anchor=(-0.2, -0.05)
)
fig.savefig("overview_selected.pdf")

# %% [markdown]
# # Selected terms (small)

# %%
select_targets = [
    "Intestinal_epithelial_cells",
    "Vascular_endothelial_cells",
    "Erythroblasts",
    "Metanephric_cells",
    "Myeloid_cells",
    "Adrenocortical_cells",
    "Squamous_epithelial_cells",
    "Visceral_neurons",
    "Amacrine_cells",
]

# %%
term2target = defaultdict(list)
for k, v in choice["gsea"].items():
    if k in select_targets:
        term2target[v].append(k)

# %%
gsea_combined_choice = gsea_combined.loc[
    gsea_combined["Term"].isin(term2target)
    & gsea_combined["Target"].isin(select_targets)
].copy()
gsea_combined_choice["Target_Display"] = gsea_combined_choice[
    "Target_Display"
].cat.remove_unused_categories()
gsea_combined_choice

# %%
nes_mat_choice = gsea_combined_choice.pivot(
    index="Term", columns="Target", values="NES"
).fillna(0)
nes_mat_choice.shape

# %%
row_link = linkage(nes_mat_choice, method="ward")
row_leaves = leaves_list(row_link)
gsea_combined_choice["Term_Display"] = pd.Categorical(
    gsea_combined_choice["Term"], categories=nes_mat_choice.index[row_leaves]
)

# %%
gsea_combined_choice["Target_Display"] = pd.Categorical(
    gsea_combined_choice["Target"],
    categories=[
        target
        for term in nes_mat_choice.index[row_leaves]
        for target in term2target[term]
    ],
)
gsea_combined_choice["Target_Display"] = gsea_combined_choice[
    "Target_Display"
].cat.rename_categories(lambda x: f"{x.replace('_', ' ')} ({choice['design'][x]})")

# %%
fig, ax = plt.subplots(figsize=(3.5, 3.5))
ax = sns.scatterplot(
    data=gsea_combined_choice.sort_values("NES"),
    x="Term_Display",
    y="Target_Display",
    size="NES",
    hue="-log10 FDR",
    palette="BuPu",
    edgecolor=None,
    sizes=(1, 500),
    ax=ax,
)
ax.margins(x=0.02, y=0.02)
locs, labels = ax.get_xticks(), ax.get_xticklabels()
ax.set_xticks(locs)
ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
ax.set_xlabel("Cell type signatures", fontsize="large", fontweight="bold")
ax.set_ylabel(
    "Target cell type\n(Representative design)", fontsize="large", fontweight="bold"
)
ax.legend(
    fontsize="large", handletextpad=1, loc="upper right", bbox_to_anchor=(-0.85, -0.05)
)
ax.margins(x=0.1, y=0.1)
fig.savefig("overview_selected_small.pdf")
