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
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick
from tqdm.auto import tqdm

from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Read scaffolds

# %%
corr = nx.read_gml("../data/scaffold/GTEx/corr.gml.gz")
ppi = nx.read_gml("../data/scaffold/BioGRID/biogrid.gml.gz")
kegg = nx.read_gml("../data/scaffold/KEGG/inferred_kegg_gene_only.gml.gz")
tf_target = nx.read_gml("../data/scaffold/TF-target/TF-target.gml.gz")

# %% [markdown]
# # Evidence retain ratio

# %% [markdown]
# ## Count edges

# %%
df = defaultdict(list)
for ds, div in tqdm(
    product(
        [
            "Replogle-2022-RPE1",
            "Replogle-2022-K562-ess",
            "Replogle-2022-K562-gwps",
            "Norman-2019",
        ],
        range(5),
    ),
    total=4 * 5,
):
    discover = nx.read_gml(
        f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
        f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
        f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        f"lam=0.1-alp=0.5-run_sd=0/discover.gml.gz"
    )
    corr_sub = corr.subgraph(discover.nodes)
    ppi_sub = ppi.subgraph(discover.nodes)
    kegg_sub = kegg.subgraph(discover.nodes)
    tf_target_sub = tf_target.subgraph(discover.nodes)

    df["Dataset"].append(ds)
    df["div"].append(div)
    df["Correlation:tot"].append(corr_sub.number_of_edges())
    df["PPI:tot"].append(ppi_sub.number_of_edges())
    df["KEGG:tot"].append(kegg_sub.number_of_edges())
    df["TF-target:tot"].append(tf_target_sub.number_of_edges())

    df["Correlation:ret"].append(len(discover.edges & corr_sub.edges))
    df["PPI:ret"].append(len(discover.edges & ppi_sub.edges))
    df["KEGG:ret"].append(len(discover.edges & kegg_sub.edges))
    df["TF-target:ret"].append(len(discover.edges & tf_target_sub.edges))
df = pd.DataFrame(df)

# %%
df["Correlation:frac"] = df.pop("Correlation:ret") / df["Correlation:tot"]
df["PPI:frac"] = df.pop("PPI:ret") / df["PPI:tot"]
df["KEGG:frac"] = df.pop("KEGG:ret") / df["KEGG:tot"]
df["TF-target:frac"] = df.pop("TF-target:ret") / df["TF-target:tot"]

# %% [markdown]
# ## Visualization

# %%
with open("config/display.yaml") as f:
    display = yaml.load(f, Loader=yaml.Loader)["naming"]["datasets"]
    display = {v: k.replace("<br>", "\n") for k, v in display.items()}

# %%
df_melt = df.melt(id_vars=["Dataset", "div"])
df_melt = (
    df_melt.assign(
        **df_melt.pop("variable")
        .str.split(":", expand=True)
        .rename(columns={0: "Evidence", 1: "variable"})
    )
    .pivot(index=["Dataset", "Evidence", "div"], columns="variable", values="value")
    .reset_index()
)
df_melt["Dataset"] = pd.Categorical(
    df_melt["Dataset"].map(display), categories=display.values()
).remove_unused_categories()
df_melt["Evidence"] = pd.Categorical(
    df_melt["Evidence"], categories=["Correlation", "PPI", "KEGG", "TF-target"]
)
df_melt

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax = sns.scatterplot(
    data=df_melt.sample(frac=1),
    x="tot",
    y="frac",
    hue="Evidence",
    style="Dataset",
    s=50,
    ax=ax,
)
ax.set_xlabel("Total edge number")
ax.set_ylabel("Edge retention rate")
ax.set_xscale("log")
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
fig.savefig("disc_retention.pdf")

# %% [markdown]
# # TF-biased orientation

# %%
tfs = set(np.loadtxt("../data/scaffold/TF-target/tfs.txt", dtype=str))
len(tfs)


# %%
def count_orientation(scaffold, discover):
    correct, incorrect = set(), set()
    for u, v in scaffold.edges:
        if not scaffold.has_edge(v, u):
            continue

        assumed_direction = None
        if u in tfs and v not in tfs:
            assumed_direction = "u->v"
        if v in tfs and u not in tfs:
            assumed_direction = "v->u"

        inferred_direction = None
        if discover.has_edge(u, v) and not discover.has_edge(v, u):
            inferred_direction = "u->v"
        if discover.has_edge(v, u) and not discover.has_edge(u, v):
            inferred_direction = "v->u"

        if assumed_direction is None or inferred_direction is None:
            continue
        if inferred_direction == assumed_direction:
            correct.add(tuple(sorted((u, v))))
        else:
            incorrect.add(tuple(sorted((u, v))))
    return len(correct), len(incorrect)


# %%
df = defaultdict(list)
for ds in tqdm(
    [
        "Replogle-2022-RPE1",
        "Replogle-2022-K562-ess",
        "Replogle-2022-K562-gwps",
        "Norman-2019",
    ]
):
    scaffold = nx.read_gml(
        f"dat/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/scaffold.gml.gz"
    )
    for div in range(5):
        discover = nx.read_gml(
            f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
            f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
            f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
            f"lam=0.1-alp=0.5-run_sd=0/discover.gml.gz"
        )
        n_correct, n_incorrect = count_orientation(scaffold, discover)
        print(
            f"ds = {ds}, div = {div}, "
            f"n_correct = {n_correct}, n_incorrect = {n_incorrect}"
        )
        df["Dataset"].append(ds)
        df["div"].append(div)
        df["n_correct"].append(n_correct)
        df["n_incorrect"].append(n_incorrect)
df = pd.DataFrame(df)

# %%
df["acc"] = df["n_correct"] / (df["n_correct"] + df["n_incorrect"])
df["Dataset"] = pd.Categorical(
    df["Dataset"].map(display), categories=display.values()
).remove_unused_categories()

# %%
fig, ax = plt.subplots()
ax = sns.boxplot(
    data=df,
    x="Dataset",
    y="acc",
    showmeans=True,
    width=0.75,
    flierprops={
        "marker": ".",
        "markerfacecolor": "black",
        "markeredgecolor": "none",
        "markersize": 7,
    },
    meanprops={
        "marker": "^",
        "markerfacecolor": "#EEEEEE",
        "markeredgecolor": "black",
        "markersize": 7,
    },
    ax=ax,
)
ax.set_ylabel("Edges oriented TF to gene")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xticks(rotation=60)
fig.savefig("disc_orient.pdf")
