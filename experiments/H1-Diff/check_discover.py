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
import re
from argparse import ArgumentParser, Namespace
from itertools import chain
from pathlib import Path
from statistics import mean

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from matplotlib import rcParams
from tqdm.auto import tqdm

from cascade.data import configure_dataset, encode_regime
from cascade.graph import filter_edges, map_edges, node_stats
from cascade.model import CASCADE
from cascade.plot import set_figure_params
from cascade.utils import config, is_notebook

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()
rcParams["axes.grid"] = False

# %%
if is_notebook():
    ct = "Astrocytes"
    size = 1
    model_dir = Path(
        "model/nptc=4-dz=16-beta=0.1-sps=L1-acyc=SpecNorm-lik=NegBin-"
        "lam=0.1-alp=0.5-run_sd=1"
    )
    design_dir = model_dir / "design" / f"target={ct}-size={size}"
    args = Namespace(
        data=Path("adata.h5ad"),
        model=model_dir / "tune.pt",
        discover=model_dir / "discover.gml.gz",
        scaffold=Path("scaffold.gml.gz"),
        markers=Path("markers.yaml"),
        design=design_dir / "design.csv",
        core_subgraph=design_dir / f"{ct}.gml",
    )
else:
    parser = ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--discover", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--core-subgraph", type=Path, required=True)
    args = parser.parse_args()
print(args)

# %% [markdown]
# # Read data

# %%
ct = re.split(r"[-=]", args.design.parent.stem)[1]
ct

# %%
with args.markers.open("r") as f:
    markers = yaml.load(f, Loader=yaml.Loader)
    markers = markers[ct.replace("_", " ")]
len(markers)

# %%
adata = ad.read_h5ad(args.data)

# %%
model = CASCADE.load(args.model)

# %%
discover = nx.read_gml(args.discover)
scaffold = nx.read_gml(args.scaffold)

# %%
discover_mean = map_edges(discover, fn=mean)
discover_confident = filter_edges(discover_mean, cutoff=0.5)
nx.is_directed_acyclic_graph(discover_confident)

# %%
design = pd.read_csv(args.design, index_col=0)
design.head()

# %%
tfs = set(np.loadtxt("../../data/scaffold/TF-target/tfs.txt", dtype=str))
len(tfs)

# %% [markdown]
# # Node statistics

# %%
node_df = node_stats(discover_confident).sort_values("out_degree", ascending=False)
node_df.head()

# %% [markdown]
# # Jacobian

# %%
encode_regime(adata, "interv", key="knockup")

# %%
configure_dataset(
    adata,
    use_regime="interv",
    use_covariate="covariate",
    use_size="ncounts",
    use_layer="counts",
)

# %%
adata_jac = model.jacobian(
    adata[np.random.choice(adata.n_obs, 128, replace=False)], batch_size=32
)

# %%
jac = adata_jac.layers["X_jac"].mean(axis=(0, -1))
jac = pd.DataFrame(jac.T, index=model.vars, columns=model.vars)

# %% [markdown]
# # Core subgraph

# %%
design = {"NFIA", "SOX1", "ZIC1"}

# %%
core_genes = set(
    chain.from_iterable(
        chain.from_iterable(
            nx.all_simple_paths(discover_confident, s, t, cutoff=5)
            for s in design
            for t in tqdm(markers, desc=s)
        )
    )
)
core_genes = core_genes | design | set(markers)
len(core_genes)

# %%
core_subgraph = discover_confident.subgraph(core_genes).copy()
for v, d in tqdm(core_subgraph.nodes.items()):
    if v in design:
        d["gene_role"] = "Design"
    elif v in markers:
        d["gene_role"] = "Marker"
    else:
        d["gene_role"] = "Other"
    if v in tfs:
        d["gene_type"] = "TF"
    else:
        d["gene_type"] = "Other"
for e, d in tqdm(core_subgraph.edges.items()):
    d["domain"] = scaffold.edges[e]["domain"]
    d["evidence"] = "Multiple" if "," in d["domain"] else d["domain"]
    d["jac"] = float(jac.loc[*e])
    d["sign"] = "activates" if d["jac"] > 0 else "represses"
core_subgraph = core_subgraph.subgraph(
    v for v in core_subgraph.nodes if core_subgraph.degree(v) or v in design
)

# %%
nx.write_gml(core_subgraph, args.core_subgraph)
