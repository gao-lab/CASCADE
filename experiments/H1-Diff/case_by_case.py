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
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import anndata as ad
import gseapy as gp
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
from cascade.utils import config, is_notebook

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()

# %% [markdown]
# # Parametrize

# %%
if is_notebook():
    ct = "Amacrine_cells"
    pert = "PAX6,TFAP2A"
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
perts = args.pert.split(",")
print(perts)

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
    markers = set(yaml.load(f, Loader=yaml.Loader)[args.target.stem.replace("_", " ")])
    # markers |= {
    #     "SLC6A9",
    #     "SCRT1",
    #     "PROX1",
    #     "CHAT",
    #     "SLC17A8",
    #     "GJD2",
    #     "SST",
    #     "PENK",
    #     "CCK",
    #     "NPW",
    #     "NPPB",
    #     "NEUROD6",
    # }
    target_ctrl_diff = target.to_df().mean() - ctrl.to_df().mean()
    markers = [i for i in markers if target_ctrl_diff.get(i, 0) > 0]
print(f"n_markers = {len(markers)}")

# %%
model = CASCADE.load(args.model)
design = IntervDesign.load(args.design)

# %%
discover = nx.read_gml(args.discover)
scaffold = nx.read_gml(args.scaffold)

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
de_df["signif"] = -np.log10(de_df["pvals_adj"]).clip(lower=-330)
# logfc_cutoff = 0.5
# signif_cutoff = 10
logfc_cutoff = 0.05
signif_cutoff = 2
de_df["highlight"] = "Insignificant"
de_df.loc[
    (de_df["signif"] > signif_cutoff) & (de_df["logfoldchanges"] > logfc_cutoff),
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
ax.set_xlabel("LogFC (Prediction vs control)")
ax.set_ylabel("-Log FDR")
ax.legend(frameon=True, framealpha=0.5)
fig.savefig(args.volcano)

# %%
for i in perts:
    tpm = np.expm1(ctfact_design[:, i].X).mean() * 1e2
    print(f"Designed TPM of {i} = {tpm:.3f}")

# %%
for i in perts:
    tpm = adata[adata.obs["knockup"] == i, i].X.expm1().mean() * 1e2
    print(f"Measured TPM of {i} = {tpm:.3f}")

# %% [markdown]
# # GSEA

# %%
if args.gsea_dir.exists():
    shutil.rmtree(args.gsea_dir)
args.gsea_dir.mkdir(parents=True, exist_ok=True)
prerank = gp.prerank(
    rnk=de_df.set_index("names")["logfoldchanges"].sort_values(ascending=False),
    gene_sets="../../data/function/MSigDB/c8.all.v2023.2.Hs.symbols.gmt",
    outdir=args.gsea_dir,
    threads=8,
    seed=0,
)

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
    discover, explain, model.export_causal_map(), cutoff=0.15
)
discover_explain.number_of_nodes(), discover_explain.number_of_edges()

# %%
response = de_df.query("highlight == 'Marker'")["names"].to_list()
print(response)

# %%
core_subgraph = core_explanation_graph(discover_explain, response, min_frac_ptr=0.1)
n_core_nodes = core_subgraph.number_of_nodes()
n_core_edges = core_subgraph.number_of_edges()
print(f"n_core_nodes = {n_core_nodes}, n_core_edges = {n_core_edges}")

# %%
nx.write_gml(prep_cytoscape(core_subgraph, scaffold, perts, response), args.explain)
