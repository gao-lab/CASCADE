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

# %% editable=true slideshow={"slide_type": ""}
import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from adjustText import adjust_text
from matplotlib import patheffects as pe
from matplotlib import pyplot as plt

from cascade.data import configure_dataset, encode_regime
from cascade.graph import annotate_explanation, core_explanation_graph, prep_cytoscape
from cascade.model import CASCADE, IntervDesign
from cascade.plot import set_figure_params
from cascade.utils import config

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()

# %% [markdown]
# # Read data

# %%
ctrl = ad.read_h5ad("ctrl.h5ad")
ctrl

# %%
norman = ad.read_h5ad("norman.h5ad")
norman

# %%
target = ad.read_h5ad("target.h5ad")
target

# %%
with open("markers.yaml") as f:
    markers = set(yaml.load(f, Loader=yaml.Loader)["Ery"])
    target_ctrl_diff = target.to_df().mean() - ctrl.to_df().mean()
    markers = [i for i in markers if target_ctrl_diff.get(i, 0) > 0]
len(markers)

# %%
model_path = (
    "model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
    "lam=0.1-alp=0.5-run_sd=1"
)
model = CASCADE.load(f"{model_path}/tune.pt")

# %%
discover = nx.read_gml(f"{model_path}/discover.gml.gz")
scaffold = nx.read_gml("scaffold.gml.gz")

# %% [markdown]
# # Load design

# %%
pert = "ZBTB1"
perts = pert.split(",")
size = len(perts)

# %%
design_path = f"{model_path}/design/size={size}"
design = IntervDesign.load(f"{design_path}/design.pt")

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
prep.obs["knockup"] = pert

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
ctfact_cov = model.counterfactual(
    prep,
    design=design,
    sample=True,
)

# %%
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
ctfact_design = model.counterfactual(
    prep,
    design=design,
    sample=True,
)

# %%
ctfact_design.X = np.log1p(
    ctfact_design.X * (1e4 / ctfact_design.obs[["ncounts"]].to_numpy())
)

# %% [markdown]
# ## Volcano plot

# %%
ctfact_combined = ad.concat(
    {"ctrl": ctfact_cov, "design": ctfact_design}, label="data", index_unique="-"
)

# %%
sc.tl.rank_genes_groups(
    ctfact_combined,
    "data",
    groups=["design"],
    reference="ctrl",
    rankby_abs=True,
    pts=True,
)

# %%
de_df = sc.get.rank_genes_groups_df(
    ctfact_combined, "design", log2fc_min=-10, log2fc_max=10
).query("pct_nz_group > 0.05")
de_df["signif"] = -np.log10(de_df["pvals_adj"]).clip(lower=-320)
logfc_cutoff = 0.5
signif_cutoff = 10
de_df["highlight"] = "Insignificant"
de_df.loc[
    (de_df["signif"] > signif_cutoff) & (de_df["logfoldchanges"].abs() > logfc_cutoff),
    "highlight",
] = "Significant"
de_df.loc[
    (de_df["highlight"] == "Significant") & de_df["names"].isin(markers),
    "highlight",
] = "Marker"
de_df.loc[de_df["names"].isin(perts), "highlight"] = "Intervened"
de_df["highlight"] = pd.Categorical(
    de_df["highlight"],
    categories=["Insignificant", "Significant", "Marker", "Intervened"],
)
de_df

# %%
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax = sns.scatterplot(
    data=de_df.sort_values("highlight"),
    x="logfoldchanges",
    y="signif",
    hue="highlight",
    palette={
        "Insignificant": "lightgrey",
        "Significant": "#1f77b4",
        "Marker": "#ff7f0e",
        "Intervened": "#9467bd",
    },
    edgecolor=None,
    s=15,
    rasterized=True,
)
ax.axvline(x=logfc_cutoff, c="darkred", ls="--")
ax.axvline(x=-logfc_cutoff, c="darkred", ls="--")
ax.axhline(y=signif_cutoff, c="darkred", ls="--")
y_min, y_max = ax.get_ylim()
ax.set_ylim(y_min, y_max * 1.1)
x_min, x_max = ax.get_xlim()
ax.set_xlim(x_min * 1.3, x_max * 1.3)
texts = [
    ax.text(row["logfoldchanges"], row["signif"], row["names"], fontsize="x-small")
    for _, row in de_df.query("highlight in ['Marker', 'Intervened']").iterrows()
]
adjust_text(
    texts, arrowprops={"arrowstyle": "-"}, min_arrow_len=5, force_text=(1.0, 1.0)
)
for t in texts:
    t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
ax.set_xlabel("Log FC (Prediction vs control)")
ax.set_ylabel("–Log FDR")
ax.legend(frameon=True, framealpha=0.5)
fig.savefig(f"{design_path}/volcano_{pert}.pdf")

# %%
for i in perts:
    tpm = np.expm1(ctfact_design[:, i].X).mean() * 1e2
    print(f"Designed TPM of {i} = {tpm:.3f}")

# %%
for i in perts:
    tpm = norman[norman.obs["knockup"] == i, i].X.expm1().mean() * 1e2
    print(f"Measured TPM of {i} = {tpm:.3f}")

# %% [markdown]
# # Graph explanation

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
configure_dataset(ctfact_design, use_layer="X_ctfact")

# %%
explain = model.explain(prep, ctfact_design, design=design)
explain

# %%
discover_explain = annotate_explanation(
    discover, explain, model.export_causal_map(), cutoff=0.1
)
discover_explain.number_of_nodes(), discover_explain.number_of_edges()

# %%
response = de_df.query("highlight == 'Marker'")["names"].to_list()
response

# %%
core_subgraph = core_explanation_graph(discover_explain, response, min_frac_ptr=0.05)
core_subgraph.number_of_nodes(), core_subgraph.number_of_edges()

# %%
nx.write_gml(
    prep_cytoscape(core_subgraph, scaffold, perts, response),
    f"{design_path}/explain_{pert}.gml",
)
