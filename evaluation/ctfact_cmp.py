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
from functools import reduce
from operator import add

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from cascade.data import configure_dataset, encode_regime
from cascade.graph import (
    annotate_explanation,
    core_explanation_graph,
    filter_edges,
    prep_cytoscape,
)
from cascade.model import CASCADE
from cascade.plot import set_figure_params

# %%
set_figure_params()

# %% [markdown]
# # Specify configuration

# %% [markdown]
# To find good cases:
#
# - Consider single-gene perturbations with complete generalization (1/1 unseen)
# - Consider high directional accuracy first, then small normalized MSE

# %%
ds = "Replogle-2022-K562-ess"
div = 3
split = "test"
interv_col = "knockdown"

# %% [markdown]
# # Read data

# %%
with open("config/display.yaml") as f:
    display = yaml.load(f, Loader=yaml.Loader)
tfs = set(np.loadtxt("../data/scaffold/TF-target/tfs.txt", dtype=str))
len(tfs)

# %% [markdown]
# ## Data

# %%
ctrl = ad.read_h5ad(f"dat/ds={ds}/imp=20/n_vars=2000/ctrl.h5ad")

# %%
true = ad.read_h5ad(
    f"dat/ds={ds}/imp=20/n_vars=2000/kg=0.9-kc=0.75-div_sd={div}/{split}.h5ad"
)
train = ad.read_h5ad(
    f"dat/ds={ds}/imp=20/n_vars=2000/kg=0.9-kc=0.75-div_sd={div}/train.h5ad"
)
prep = ad.read_h5ad(
    f"dat/ds={ds}/imp=20/n_vars=2000/kg=0.9-kc=0.75-div_sd={div}/ctfact_{split}.h5ad"
)

# %%
rnd = np.random.RandomState(0)
idx = rnd.choice(ctrl.n_obs, prep.n_obs, replace=True)
prep.obsm["ctrl_covariate"] = ctrl.obsm["covariate"][idx]

# %% [markdown]
# ## Graph

# %%
scaffold = nx.read_gml(
    f"dat/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/scaffold.gml.gz"
)
discover = nx.read_gml(
    f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
    f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
    f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
    f"lam=0.1-alp=0.5-run_sd=0/"
    f"discover.gml.gz"
)

# %% [markdown]
# ## Model

# %%
model = CASCADE.load(
    f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
    f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
    f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
    f"lam=0.1-alp=0.5-run_sd=0-tune_ct=True/"
    f"tune.pt"
)

# %% [markdown]
# ## Read counterfactuals

# %%
linear = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/nil/"
    f"kg=0.9-kc=0.75-div_sd={div}/linear/"
    f"dim=10-lam=0.1-run_sd=0/"
    f"ctfact_{split}.h5ad"
)

# %%
additive = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/nil/"
    f"kg=0.9-kc=0.75-div_sd={div}/additive/"
    f"run/"
    f"ctfact_{split}.h5ad"
)

# %%
cpa = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/lsi/"
    f"kg=0.9-kc=0.75-div_sd={div}/cpa/"
    f"run_sd=0/"
    f"ctfact_{split}.h5ad"
)

# %%
biolord = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/lsi/"
    f"kg=0.9-kc=0.75-div_sd={div}/biolord/"
    f"run_sd=0/"
    f"ctfact_{split}.h5ad"
)

# %%
gears = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/go/"
    f"kg=0.9-kc=0.75-div_sd={div}/gears/"
    f"hidden_size=64-epochs=20-run_sd=0/"
    f"ctfact_{split}.h5ad"
)

# %%
scgpt = ad.read_h5ad(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/go/"
    f"kg=0.9-kc=0.75-div_sd={div}/scgpt/"
    f"epochs=20-run_sd=0/"
    f"ctfact_{split}.h5ad"
)

# %% [markdown]
# # Metrics

# %%
scgpt_metrics = pd.read_csv(
    f"inf/ds={ds}/imp=20/n_vars=2000/cpl/go/"
    f"kg=0.9-kc=0.75-div_sd={div}/scgpt/"
    f"epochs=20-run_sd=0/"
    f"metrics_ctfact_{split}_each.csv",
    index_col=0,
)

# %%
cascade_metrics = pd.read_csv(
    f"inf/ds={ds}/imp=20/n_vars=2000/kegg+tf+ppi+corr/lsi/"
    f"kg=0.9-kc=0.75-div_sd={div}/cascade/"
    f"nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
    f"lam=0.1-alp=0.5-run_sd=0-tune_ct=True-ablt=none/"
    f"metrics_ctfact_{split}_each.csv",
    index_col=0,
)

# %%
metrics_merged = pd.concat(
    [
        scgpt_metrics[["normalized_mse_top20"]].add_suffix("_gpt"),
        cascade_metrics[["normalized_mse_top20", "category"]],
    ],
    axis=1,
)
metrics_merged["adv"] = (
    metrics_merged["normalized_mse_top20_gpt"] - metrics_merged["normalized_mse_top20"]
)

# %%
metrics_merged.sort_values("adv", ascending=False)

# %% [markdown]
# # Pick example

# %% [markdown]
# Key points to consider when selecting examples:
#
# - Predictions should be accurate (small `normalized_mse_top20`)
# - Contribution of intervention should be large
#   (`normalized_mse_top20_ni` - `normalized_mse_top20`)

# %% [markdown]
# Examples:
#
# - Norman-2019
#   - split 0, KLF1+TGFBR2 🟡
#     - KLF1 dominates both DEG and DS
#   - split 0, FOXA1+KLF1 🟢
#     - Dominated by KLF1, but can illustrate accuracy in covariate-based
#       vs graph-based contributions
#     - DS is pretty good, with little crosstalk
#   - split 0, FOXA1+FOXL2 🟡
#     - DEG mostly covariate
#     - DS dominated by FOXA1, with weak FOXL2
#   - split 1, OSR2+UBASH3B 🟡
#     - DEG dominated by OSR2
#     - DS has weak UBASH3B
#   - split 1, OSR2+PTPN12 🟡
#     - DEG dominated by OSR2
#     - DS has weak PTPN12
#   - split 1, MAP2K6+SPI1 🟡
#     - DEG and DS are both dominated by OSR2, with weak MAP2K6
#   - split 1, CEBPB+OSR2 🟢🟡
#     - DEG and DS with strong contribution from both, and many crosstalks
#     - DEG has a separate covariate-predicted cluster, which is less accurate
#   - split 1, CEBPE+FOSB 🟢🟡
#     - DEG and DS with strong contribution from both, and many crosstalks
#     - DEG has a separate covariate-predicted cluster, which is less accurate
#   - split 2, SGK1,TBX3 🟡
#     - DEG and DS are both dominated by TBX3, with weak SGK1
#   - split 2, TBX2,TBX3 🟢
#     - Nice
#     - They are indeed inhibitory!
#   - split 2, BAK1,KLF1 🟡
#     - DEG only has KLF1
#     - DS has weak BAK1
#   - split 3, CEBPE,CNN1 🟡
#     - DEG only has CEBPE
#     - DS has weak CNN1
#   - split 3, COL2A1,KLF1 🟡
#     - DEG only weak KLF1, mostly latent
#     - DS is ok
#   - split 4, SNAI1,UBASH3B 🟡
#     - DEG only SNAI1
#     - DS is ok
#   - split 4, CEBPE,KLF1 🟡
#     - DEG only KLF1
#     - DS is ok
#   - split 4, FOXA1,FOXA3 🟡🟢
#     - DEG only FOXA1
#     - DS is ok, with many co-regulation
#
# - Replogle-2022-K562-ess
#   - split 0, CDK1 🟡
#     - DEG mostly covariate, weak graph
#     - DS is ok
#   - split 0, SKP2 🔴
#     - DEG mostly covariate, weak graph
#     - DS is not very accurate
#   - split 0, RPL31 🟡
#     - DEG has weak graph and many latent
#     - DS is ok
#   - split 0, RRP1 🟡
#     - DEG has weak graph and many latent
#     - DS is ok
#   - split 1, RPS29 🟡
#     - DEG has weak graph and many latent
#     - DS is ok
#   - split 1, RPS11 🟡
#     - DEG has weak graph and many latent
#     - DS is ok
#   - split 1, RPL23 🟢
#     - DEG has both graph and latent
#     - DDIT3 indirect effect decomposition
#     - DS is ok
#   - split 2, MRPS22 🔴
#     - DEG all latent
#     - DS is ok
#   - split 3, RPS9 🟢
#     - DEG has both graph and latent
#     - DDIT3 indirect effect decomposition
#     - DS is ok
#
# - Replogle-2022-K562-gwps
#   - split 0, MCM6 🔴
#     - DEG too much covariate
#   - split 3, MYB 🟡🟢
#     - DEG too much covariate
#     - DS is pretty good
#   - split 4, HHEX 🔴
#     - Predicted effects are too small
#   - split 4, MRPL35 🔴
#     - Both DEG and DS too much covariate
#   - split 4, SSRP1 🔴
#     - Both DEG and DS too much covariate
#
# - Replogle-2022-RPE1
#   - split 0, RARS2 🔴
#     - DEG all covariate
#     - DS only one downstream gene
#   - split 1, KRT8 🔴
#     - DEG too much covariate
#     - DS only one downstream gene

# %% [markdown]
# ## ❗Specify perturbation

# %%
pert = "RPS9"
perts = pert.split(",")
category = (
    true.obs[[interv_col, "category"]]
    .drop_duplicates()
    .set_index(interv_col)
    .loc[pert, "category"]
)
category

# %% [markdown]
# ## Counterfactual with sampling

# %%
prep_sub = prep[prep.obs[interv_col] == pert].copy()

# %%
encode_regime(prep_sub, "interv", key=interv_col)
configure_dataset(
    prep_sub,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)
prep_sub

# %%
cascade = model.counterfactual(prep_sub, sample=True)
cascade.X = np.log1p(cascade.X * (1e4 / np.asarray(cascade.obs[["ncounts"]])))

# %% [markdown]
# ## Explain

# %%
encode_regime(prep_sub, "interv")
configure_dataset(
    prep_sub,
    use_regime="interv",
    use_covariate="ctrl_covariate",
    use_size="ncounts",
    use_layer="counts",
)

# %%
configure_dataset(cascade, use_layer="X_ctfact")

# %%
explain = model.explain(prep_sub, cascade)
explain

# %%
discover_explain = annotate_explanation(
    discover, explain, model.export_causal_map(), cutoff=0.1
)
discover_explain.number_of_nodes(), discover_explain.number_of_edges()

# %% [markdown]
# ## Specify response genes

# %% [markdown]
# ### ❗Use DEGs

# %%
de_df = sc.get.rank_genes_groups_df(true, pert).merge(
    true.var[["gene_name", "means"]].rename(columns={"gene_name": "names"})
)
de_df["exists"] = de_df["names"].isin(set(true.var_names) - set(perts))
response = de_df.query("exists & means > 0.2").head(20)["names"].to_list()
xlabel = "Top DEGs"
suffix = "DEG"

# %% [markdown]
# ### ❗Use inferred downstream genes

# %%
discover_strong = filter_edges(discover_explain, edge_attr="frac", cutoff=0.1)
response = reduce(
    add,
    [list(nx.bfs_tree(discover_strong, p).nodes)[: (20 // len(perts))] for p in perts],
)
response = sorted(set(response))
xlabel = "Downstream genes"
suffix = "DS"

# %% [markdown]
# ### ❗Use non-additive genes

# %%
response = sorted(
    {
        "ALAS2",
        "GYPB",
        "HBG2",
        "HBA2",
        "GYPE",
        *perts,
    }
)
xlabel = "Non-additive genes"
suffix = "GI"

# %% [markdown]
# ## Prepare plotting

# %%
cmp = [
    ctrl[:, response].to_df().assign(data="Ctrl"),
    linear[linear.obs[interv_col] == pert, response].to_df().assign(data="Linear"),
    additive[additive.obs[interv_col] == pert, response]
    .to_df()
    .assign(data="Additive"),
    cpa[cpa.obs[interv_col] == pert, response].to_df().assign(data="CPA"),
    biolord[biolord.obs[interv_col] == pert, response].to_df().assign(data="Biolord"),
    gears[gears.obs[interv_col] == pert, response].to_df().assign(data="GEARS"),
    scgpt[scgpt.obs[interv_col] == pert, response].to_df().assign(data="scGPT"),
    cascade[cascade.obs[interv_col] == pert, response].to_df().assign(data="CASCADE"),
    true[true.obs[interv_col] == pert, response].to_df().assign(data="Truth"),
]

# %%
cmp = pd.concat(cmp, ignore_index=True)
cmp.loc[:, response] = cmp.loc[:, response] - ctrl[:, response].to_df().median()
cmp = cmp.melt(id_vars=["data"], var_name="gene_name")

# %%
cmp["gene_name"] = pd.Categorical(
    cmp["gene_name"],
    categories=cmp.query("data == 'Truth'")
    .groupby("gene_name")
    .median(numeric_only=True)
    .sort_values("value", ascending=False)
    .index,
)

# %% [markdown]
# ## Plotting

# %%
cmp_multi = cmp.copy()
cmp_multi.loc[
    cmp_multi["data"].isin(["Linear", "Additive", "GEARS", "scGPT"]), "value"
] = np.nan
cmp_single = cmp.copy()
cmp_single.loc[
    ~cmp_single["data"].isin(["Linear", "Additive", "GEARS", "scGPT"]), "value"
] = np.nan

# %%
palette = {
    "Ctrl": "#ffffff",
    "Additive": "#7f7f7f",
    "Truth": "#17becf",
    **display["palette"]["methods"],
}

# %%
fig, ax = plt.subplots(figsize=(25, 5))
ax = sns.boxplot(
    x="gene_name",
    y="value",
    hue="data",
    data=cmp_multi,
    fliersize=1,
    width=0.7,
    linewidth=0.7,
    palette=palette,
    ax=ax,
)
ax = sns.stripplot(
    x="gene_name",
    y="value",
    hue="data",
    data=cmp_single,
    dodge=True,
    jitter=False,
    palette=palette,
    legend=False,
    size=3.5,
    marker="D",
    edgecolor="#3d3d3d",
    linewidth=0.7,
    ax=ax,
)
ax.axhline(y=0, c="darkred", ls="--")
ax.legend(
    loc="upper right",
    bbox_to_anchor=(0.995, 0.99),
    ncol=4,
    frameon=True,
    framealpha=0.5,
)
ax.set_ylim(cmp["value"].quantile(0.0001) - 0.5, cmp["value"].quantile(0.9999) + 0.5)
plt.xticks(rotation=60)
ax.set_xlabel(xlabel, fontsize="large")
ax.set_ylabel("Change in log-normalized counts", fontsize="large")
ax.set_title(" + ".join(perts) + f" ({category})")
fig.savefig(
    f"inf/ds={ds}/imp=20/n_vars=2000/ctfact_cmp_{div}_{split}_{pert}_{suffix}.pdf"
)


# %% [markdown]
# ## Core explanation graph

# %%
core_subgraph = core_explanation_graph(discover_explain, response, min_frac_ptr=0.05)
core_subgraph.number_of_nodes(), core_subgraph.number_of_edges()

# %%
nx.write_gml(
    prep_cytoscape(core_subgraph, scaffold, perts, response),
    f"inf/ds={ds}/imp=20/n_vars=2000/ctfact_core_{div}_{split}_{pert}_{suffix}.gml",
)
