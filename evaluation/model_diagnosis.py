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
from statistics import mean

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import plotly as pl
import scanpy as sc
import seaborn as sns
from diagnosis_utils import (
    args,
    densify,
    get_goea_df,
    kde_scatter,
    mde,
    run_goea,
    scatter,
    show_toggles,
    split_mask,
)
from IPython.display import display
from ipywidgets import HBox, Output, ToggleButtons, interact
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.preprocessing import OneHotEncoder

from cascade.data import aggregate_obs, configure_dataset, encode_regime
from cascade.graph import demultiplex, filter_edges, map_edges, node_stats
from cascade.metrics import annot_resp
from cascade.model import CASCADE
from cascade.plot import interactive_heatmap, pair_grid, set_figure_params
from cascade.utils import config, hclust

# %%
config.LOG_LEVEL = "DEBUG"
set_figure_params()
rcParams["axes.grid"] = False
pl.offline.init_notebook_mode(connected=True)

# %% [markdown]
# # Parametrize

# %%
show_toggles()

# %%
args

# %% [markdown]
# # Read model, data and metrics

# %% [markdown]
# ## Model

# %%
discover = CASCADE.load(args["discover"])
tune = CASCADE.load(args["tune"])

# %% [markdown]
# ## Data

# %%
scaffold = nx.read_gml(args["scaffold"])

# %%
train = ad.read_h5ad(args["train"])
test = ad.read_h5ad(args["test"])
adata = ad.concat([train, test], merge="first", uns_merge="first")

# %%
ctfact_train_input = ad.read_h5ad(args["ctfact_train_input"])
ctfact_test_input = ad.read_h5ad(args["ctfact_test_input"])
ctfact_input = ad.concat(
    [ctfact_train_input, ctfact_test_input],
    merge="first",
    uns_merge="first",
    index_unique="-",
)

# %%
ctfact_train = ad.read_h5ad(args["ctfact_train"])
ctfact_test = ad.read_h5ad(args["ctfact_test"])
ctfact = ad.concat(
    [ctfact_train, ctfact_test],
    merge="first",
    uns_merge="first",
    index_unique="-",
)

# %% [markdown]
# ## Metrics

# %%
metrics_ctfact_train = pd.read_csv(args["metrics_ctfact_train"], index_col=0)
metrics_ctfact_test = pd.read_csv(args["metrics_ctfact_test"], index_col=0)
metrics_ctfact = pd.concat([metrics_ctfact_train, metrics_ctfact_test])

# %%
metrics_ctfact_gears = pd.concat(
    [
        pd.read_csv(
            f"inf/ds={args['ds']}/imp={args['imp']}/n_vars={args['n_vars']}/cpl/go/"
            f"kg=0.9-kc=0.75-div_sd={args['div_sd']}/"
            f"gears/hidden_size=64-epochs=20-run_sd=0/metrics_ctfact_train_each.csv",
            index_col=0,
        ),
        pd.read_csv(
            f"inf/ds={args['ds']}/imp={args['imp']}/n_vars={args['n_vars']}/cpl/go/"
            f"kg=0.9-kc=0.75-div_sd={args['div_sd']}/"
            f"gears/hidden_size=64-epochs=20-run_sd=0/metrics_ctfact_test_each.csv",
            index_col=0,
        ),
    ]
)

# %% [markdown]
# # Configure data

# %%
interv_cols = (
    {"knockout", "knockdown", "knockup"}
    & set(train.obs.columns)
    & set(test.obs.columns)
)
assert len(interv_cols) == 1
interv_col = interv_cols.pop()
interv_col

# %%
encode_regime(adata, "interv", key=interv_col)

# %%
if args["lik"] == "Normal":
    configure_dataset(
        adata,
        use_regime="interv",
        use_covariate="covariate",
    )
elif args["lik"] == "NegBin":
    configure_dataset(
        adata,
        use_regime="interv",
        use_covariate="covariate",
        use_size="ncounts",
        use_layer="counts",
    )
else:
    raise ValueError("Unrecognized likelihood module")

# %% [markdown]
# # Plot preparations

# %% [markdown]
# ## Category

# %%
categories = [
    "0 seen",
    "1 seen",
    "2 seen",
    "0/2 unseen",
    "1/2 unseen",
    "2/2 unseen",
    "1/1 unseen",
]
adata.obs["category"] = adata.obs["category"].cat.set_categories(categories)
metrics_ctfact["category"] = pd.Categorical(
    metrics_ctfact["category"], categories=categories
)
metrics_ctfact_gears["category"] = pd.Categorical(
    metrics_ctfact_gears["category"], categories=categories
)

# %% [markdown]
# # Metrics comparison


# %%
@interact
def metric_cmp(
    metric=[
        col
        for col in metrics_ctfact.columns
        if col not in ("edist", "category") and not col.startswith("true_")
    ],
    split=["both", "train", "test"],
    hue=["edist", "category"],
    log=False,
):
    df = pd.DataFrame(
        {
            "CASCADE": metrics_ctfact[metric],
            "GEARS": metrics_ctfact_gears[metric],
            "category": metrics_ctfact["category"],
            "edist": metrics_ctfact["edist"],
        }
    )
    df["diff"] = df["CASCADE"] - df["GEARS"]
    split_mask(df, hue, split)

    o1, o2 = Output(), Output()
    hbox = HBox([o1, o2])
    with o1:
        fig, ax = plt.subplots()
        scatter("GEARS", "CASCADE", hue, data=df, log=log, ax=ax, edgecolor=None, s=20)
        ax.axline([0, 0], slope=1, c="darkred", ls="--")
        ax.set_title(metric)
        plt.close()
        display(fig)
    with o2:
        display(
            df.dropna().sort_values("CASCADE", ascending="mse" in metric).head(n=10)
        )
    display(hbox)


# %% [markdown]
# # Data stats

# %% [markdown]
# ## Representative genes

# %%
adata.var.query("perturbed").sort_values("ncounts", ascending=False).head(n=20)


# %% [markdown]
# ## Distribution per gene


# %%
@interact
def hist(gene=adata.var_names[0], hue=["perturbed", "category"], log=False):
    df = sc.get.obs_df(
        adata,
        [gene, interv_col, "category"],
        layer="counts" if args["lik"] == "NegBin" else None,
    )
    df["perturbed"] = df[interv_col].map(lambda x: gene in x.split(","))
    g = sns.displot(
        data=df,
        x=gene,
        hue=hue,
        kind="hist",
        stat="density",
        bins=100,
        common_norm=False,
    )
    if log:
        g.ax.set_yscale("log")


# %% [markdown]
# # Model selection

# %%
model = tune

model_toggle = ToggleButtons(
    options=["discover", "tune"], label="tune", description="model"
)


def model_click(change):
    global model
    if change["new"] == "discover":
        model = discover
    elif change["new"] == "tune":
        model = tune
    else:
        raise RuntimeError


model_toggle.observe(model_click, names="value")
display(model_toggle)


# %% [markdown]
# # Intervention parameters


# %%
def get_param(name, particle):
    param = getattr(model.net, name)
    return param[particle].detach().cpu().numpy()


def get_param_df(particle):
    return pd.DataFrame(
        {
            "interv_scale": get_param("interv_scale", particle),
            "interv_bias": get_param("interv_bias", particle),
            "interv_seen": model.vars.map(lambda x: x in model.interv_seen),
        },
        index=model.vars,
    )


# %%
@interact
def param(
    particle=list(range(int(args["nptc"]))),
    gene=adata.var_names[0],
):
    df = get_param_df(particle)
    # df.query("interv_seen", inplace=True)

    o1, o2 = Output(), Output()
    hbox = HBox([o1, o2])
    with o1:
        g = sns.displot(df["interv_scale"], bins=20)
        g.ax.axvline(x=df["interv_scale"].loc[gene], c="darkred", ls="--")
        g.fig.suptitle(f"Interventional scale (highlighting {gene})", y=1.05)
        plt.close()
        display(g.fig)
    with o2:
        g = sns.displot(df["interv_bias"], bins=20)
        g.ax.axvline(x=df["interv_bias"].loc[gene], c="darkred", ls="--")
        g.fig.suptitle(f"Interventional bias (highlighting {gene})", y=1.05)
        plt.close()
        display(g.fig)
    display(hbox)


# %% [markdown]
# # Causal structure

# %%
graph = model.export_causal_graph()
graphs = demultiplex(graph)
graph = map_edges(graph, fn=mean)

# %% [markdown]
# ## Scaffold heatmap

# %%
scf = nx.to_pandas_adjacency(scaffold, nodelist=model.vars, weight=None)
scf_row_linkage, _ = hclust(scf, cut=False)
scf_col_linkage, _ = hclust(scf.T, cut=False)
interactive_heatmap(
    scf.iloc[leaves_list(scf_row_linkage), leaves_list(scf_col_linkage)],
    row_clust=None,
    col_clust=None,
    colorscale="Viridis",
)

# %% [markdown]
# ## Adjacency heatmap


# %%
adj = nx.to_pandas_adjacency(graph, nodelist=model.vars)
adj_row_linkage, _ = hclust(adj, cut=False)
adj_col_linkage, _ = hclust(adj.T, cut=False)
interactive_heatmap(
    adj.iloc[leaves_list(adj_row_linkage), leaves_list(adj_col_linkage)],
    row_clust=None,
    col_clust=None,
    colorscale="Viridis",
)

# %% [markdown]
# ## Jacobian heatmap

# %%
# # %%debug
adata_jac = model.jacobian(
    adata[np.random.choice(adata.n_obs, 1000, replace=False)], batch_size=32
)
jac_min = adata_jac.layers["X_jac"].min(axis=0)
jac_max = adata_jac.layers["X_jac"].max(axis=0)
jac = np.where(np.fabs(jac_min) > np.fabs(jac_max), jac_min, jac_max).mean(axis=-1)
jac = pd.DataFrame(jac.T, index=model.vars, columns=model.vars)

# %%
jac_norm = jac.div(
    jac.abs().apply(
        lambda x: np.quantile(x[x > 0], 0.9) if (x > 0).any() else 1, axis=0
    ),
    axis="columns",
)
jac_norm = jac_norm.div(
    jac_norm.abs().apply(
        lambda x: np.quantile(x[x > 0], 0.9) if (x > 0).any() else 1, axis=1
    ),
    axis="index",
)
jac_norm.min(axis=None), jac_norm.max(axis=None)

# %%
jac_norm_clip = jac_norm.clip(lower=-1, upper=1)
jac_row_linkage, _ = hclust(jac_norm_clip, cut=False)
jac_col_linkage, _ = hclust(jac_norm_clip.T, cut=False)
interactive_heatmap(
    jac_norm_clip.iloc[leaves_list(jac_row_linkage), leaves_list(jac_col_linkage)],
    row_clust=None,
    col_clust=None,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
    zmin=-1,
    zmax=1,
)

# %% [markdown]
# ### GO similarity

# %%
go_essential_all = pd.read_csv("../data/function/GO/go_essential_all.csv")
go_essential_all.head()

# %%
go_cmp = (
    jac_norm.reset_index()
    .rename(columns={"index": "source"})
    .melt(id_vars=["source"], var_name="target", value_name="jacobian")
    .query("jacobian != 0")
    .merge(go_essential_all)
)
go_cmp.head()

# %%
go_cmp["direction"] = "Unclear"
go_cmp.loc[go_cmp["jacobian"] > 1, "direction"] = "Positive"
go_cmp.loc[go_cmp["jacobian"] < -1, "direction"] = "Negative"
fig, ax = plt.subplots(figsize=(3, 5))
ax = sns.stripplot(
    data=go_cmp.query("direction != 'Unclear'"),
    x="direction",
    y="importance",
    hue="direction",
    ax=ax,
)
_ = ax.set_xlabel("Jacobian")
_ = ax.set_ylabel("GO Jaccard index")


# %% [markdown]
# ### Graph embedding and clustering


# %%
jac_emb = mde(jac_norm, emb_dim=10, abs_norm=5, dissimilar_ratio=10)
jac_emb = ad.AnnData(jac_emb)
sc.pp.neighbors(jac_emb, n_neighbors=15)
sc.tl.umap(jac_emb)
sc.tl.leiden(jac_emb, resolution=2, flavor="leidenalg")

# %%
fig, ax = plt.subplots(figsize=(6, 6))
sc.pl.umap(
    jac_emb,
    color="leiden",
    na_in_legend=False,
    add_outline=True,
    outline_width=(0.2, 0.05),
    legend_loc="on data",
    legend_fontoutline=3,
    ax=ax,
)

# %%
run_goea(jac_emb.obs["leiden"])


# %%
@interact
def visualize_goea(leiden=jac_emb.obs["leiden"].cat.categories):
    try:
        df = get_goea_df(leiden)
        fig, ax = plt.subplots(figsize=(6, 8))
        ax = sns.barplot(data=df.head(30), x="-log10_fdr", y="name", ax=ax)
        ax.set_xlabel("-log10 FDR")
        ax.set_ylabel("GO term")
        ax.axvline(x=-np.log10(0.05), c="darkred", ls="--")
    except FileNotFoundError:
        print(f"No enrichment for leiden = {leiden}!")


# %%
jac_emb.obs["process"] = (
    jac_emb.obs["leiden"]
    .astype(str)
    .map(
        {
            # Replogle-2022-K562-ess
            "0": "Mitosis",
            "1": "rRNA",
            "2": "DNA replication",
            "3": "Mito",
            "4": "Translation",
            # "5": "Transcription",
            "6": "Transcription",
            "7": "Splicing",
            "9": "Nucleolus",
            "10": "TF",
            "11": "TF",
            "13": "Plasma membrane",
            "16": "Exosome",
            "17": "Respiratory chain",
            "18": "Actin cytoskeleton",
            "19": "ER",
            "20": "Calcium transport",
            "21": "Transcription",
            "22": "Respiratory chain",
            "23": "Plasma membrane",
            "24": "Microtubule",
            "25": "Proteasome",
            "27": "Amino acid metab",
            "28": "Lipoprotein",
            "29": "Oxygen transport",
            "30": "Sterol metab",
            "31": "Nucleosome",
            "36": "tRNA",
        }
    )
    .fillna("Unclear")
)

# %%
fig, ax = plt.subplots(figsize=(12, 12))
sc.pl.umap(
    jac_emb[jac_emb.obs["process"] != "Unclear"],
    color="process",
    na_in_legend=False,
    add_outline=True,
    outline_width=(0.2, 0.05),
    legend_loc="on data",
    legend_fontoutline=3,
    ax=ax,
)

# %% [markdown]
# ### Within process graph

# %%
goi = jac_emb.obs.query("process == 'Mitosis'").index
interactive_heatmap(
    jac_norm.loc[goi, goi].clip(lower=-0.5, upper=0.5),
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
    zmin=-0.5,
    zmax=0.5,
)

# %% [markdown]
# ### Across-process graph

# %%
ohe = OneHotEncoder()
process_proj = pd.DataFrame(
    ohe.fit_transform(jac_emb.obs[["process"]]).toarray(),
    index=jac_emb.obs_names,
    columns=ohe.categories_[0],
)
process_proj = process_proj / process_proj.sum()

# %%
process_jac = pd.DataFrame(
    process_proj.to_numpy().T
    @ jac_norm.clip(lower=-0.05, upper=0.05).to_numpy()
    @ process_proj.to_numpy(),
    index=process_proj.columns,
    columns=process_proj.columns,
)
process_jac = process_jac.loc[
    process_jac.index != "Unclear", process_jac.columns != "Unclear"
]
L = linkage(
    pd.concat([process_jac, process_jac.T], axis=1), metric="cosine", method="average"
)
jac_emb.obs["process"] = pd.Categorical(
    jac_emb.obs["process"], categories=process_jac.index[leaves_list(L)]
)

# %%
g = sns.clustermap(
    process_jac.clip(lower=-0.01, upper=0.01),
    row_linkage=L,
    col_linkage=L,
    cmap="bwr",
    center=0,
    figsize=(12, 12),
)

# %%
g = sns.clustermap(
    (process_jac - process_jac.T).clip(lower=-0.05, upper=0.05),
    row_linkage=L,
    col_linkage=L,
    cmap="bwr",
    center=0,
    figsize=(12, 12),
)

# %% [markdown]
# ## Response heatmap

# %%
annot_resp(scaffold, adata, interv_col)

# %%
resp = nx.to_pandas_adjacency(scaffold, model.vars, weight="fwd_diff")
interactive_heatmap(
    resp.iloc[leaves_list(jac_row_linkage), leaves_list(jac_col_linkage)].clip(
        lower=-5, upper=5
    ),
    row_clust=None,
    col_clust=None,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
)

# %% [markdown]
# ## Correlation heatmap

# %%
corr = adata.to_df().sample(5000).corr()

# %%
interactive_heatmap(
    corr,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
    highlights={"MCM3": "purple"},
    zmin=-0.5,
    zmax=0.5,
)

# %% [markdown]
# ## Grouping heatmaps by processes

# %%
process_annot = jac_emb.obs["process"].sort_values()

# %%
interactive_heatmap(
    scf.loc[process_annot.index[::-1], process_annot.index],
    row_clust=process_annot,
    col_clust=process_annot,
    colorscale="Viridis",
)

# %%
interactive_heatmap(
    adj.loc[process_annot.index[::-1], process_annot.index],
    row_clust=process_annot,
    col_clust=process_annot,
    colorscale="Viridis",
)

# %%
interactive_heatmap(
    jac_norm_clip.loc[process_annot.index[::-1], process_annot.index],
    row_clust=process_annot,
    col_clust=process_annot,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
)

# %%
interactive_heatmap(
    resp.loc[process_annot.index[::-1], process_annot.index].clip(lower=-5, upper=5),
    row_clust=process_annot,
    col_clust=process_annot,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
)

# %%
interactive_heatmap(
    corr.loc[process_annot.index[::-1], process_annot.index],
    row_clust=process_annot,
    col_clust=process_annot,
    colorscale=[[0, "blue"], [0.5, "white"], [1.0, "red"]],
    zmin=-0.5,
    zmax=0.5,
)

# %% [markdown]
# ## Specific subgraph

# %%
# Norman-2019
# goi = ["TBX2", "OSR2", "DUSP9", "COL2A1", "MAPK1", "ETS2", "SET", "KLF1"]
# goi = ["CDK1", "CCNB1", "CDC20", "SPC25"]
# Replogle-2022-K562-gwps
# goi = ["INTS14", "INTS13", "INTS10", "INTS3"]
goi = ["MCM3", "MCM7", "MCM4", "MCM5", "MCM6", "MCM2"]
sns.heatmap(jac_norm.loc[goi, goi], cmap="bwr", vmin=-0.5, vmax=0.5)

# %% [markdown]
# ## Node statistics


# %%
@interact
def stat(particle=[-1] + list(range(int(args["nptc"]))), gene=adata.var_names[0]):
    g = graph if particle < 0 else graphs[particle]
    o = Output()
    with o:
        display(node_stats(filter_edges(g, cutoff=0.5)).query(f"node == '{gene}'"))
    return o


# %% [markdown]
# # Diagnosis

# %%
# # %%debug
adata_diag = model.diagnose(adata[np.random.choice(adata.n_obs, 10000, replace=False)])
adata_diag.obs["category"] = adata_diag.obs["category"].cat.set_categories(categories)
adata_diag


# %% [markdown]
# ## Latent visualization


# %%
def extract_latent(particle):
    obsm = adata_diag.obsm
    obsm["Z"] = obsm["Z_mean_diag"][..., particle]
    obsm["Z-std"] = obsm["Z_std_diag"][..., particle]
    dim = obsm["Z"].shape[1]
    if dim == 0:
        return None
    df = sc.get.obs_df(
        adata_diag,
        keys=[interv_col, "edist", "category"],
        obsm_keys=[("Z", i) for i in range(dim)] + [("Z-std", i) for i in range(dim)],
    )
    df = df.merge(
        metrics_ctfact.drop(columns=["edist", "category"]),
        how="left",
        left_on=interv_col,
        right_index=True,
    )
    df[interv_col] = pd.Categorical(df[interv_col])
    df = df.drop_duplicates(subset=interv_col)
    return df


# %%
@interact
def pairplot(
    particle=list(range(int(args["nptc"]))),
    split=["both", "train", "test"],
    hue=[interv_col, *metrics_ctfact.columns],
    log=False,
):
    df = extract_latent(particle)
    if df is None:
        return
    dim = df.columns.str.contains("-std").sum()
    split_mask(df, hue, split)
    if hue == interv_col:
        cutoff = (
            df.drop_duplicates(subset=interv_col)["edist"]
            .sort_values(ascending=False)
            .iloc[20]
        )
        df.loc[df["edist"] < cutoff, interv_col] = np.nan
        df[interv_col] = df[interv_col].cat.remove_unused_categories()
        df.sort_values(interv_col, na_position="first", inplace=True)

    def _scatter(x, y, **kws):
        scatter(x.name, y.name, hue, data=df, log=log, ax=plt.gca())

    def _hide(*args, **kws):
        plt.gca().set_visible(False)

    o1, o2 = Output(), Output()
    hbox = HBox([o1, o2])
    with o1:
        g1 = sns.PairGrid(df, vars=[f"Z-{i}" for i in range(dim)])
        g1.map_lower(_scatter)
        g1.map_upper(_hide)
        g1.map_diag(sns.histplot, fill=False, color="black")
        g1.fig.suptitle(f"Latent mean of particle {particle}", y=1.02)
        plt.close()
        display(g1.fig)
    with o2:
        g2 = sns.PairGrid(
            df,
            x_vars=[f"Z-{i}" for i in range(dim)],
            y_vars=[f"Z-std-{i}" for i in range(dim)],
        )
        g2.map(_scatter)
        g2.fig.suptitle(f"Latent std of particle {particle}", y=1.02)
        plt.close()
        display(g2.fig)
    display(hbox)


# %% [markdown]
# ## Regression accuracy


# %% [markdown]
# ### Total library size


# %%
@interact
def library_size(
    particle=list(range(int(args["nptc"]))), split=["all", "train", "test", "ctrl"]
):
    if args["lik"] != "NegBin":
        return
    adata_diag.obs["true_ncounts"] = adata_diag.layers["counts"].sum(axis=1)
    adata_diag.obs["pred_ncounts"] = adata_diag.layers["X_mean_diag"][
        ..., particle
    ].sum(axis=1)
    df = sc.get.obs_df(
        adata_diag, [interv_col, "true_ncounts", "pred_ncounts", "category"]
    ).assign(split=True)
    df["split"] = df["split"].astype("category")
    split_mask(df, "split", split)

    fig, ax = kde_scatter("true_ncounts", "pred_ncounts", "split", df)
    ax.axline((0, 0), slope=1, c="darkred", ls="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Total count")


# %% [markdown]
# ### Per gene regression


# %%
@interact
def regression(
    particle=list(range(int(args["nptc"]))),
    gene=adata_diag.var_names[0],
    split=["all", "train", "test", "ctrl"],
    errorbars=True,
):
    adata_diag.layers["pred_mean"] = adata_diag.layers["X_mean_diag"][..., particle]
    adata_diag.layers["pred_std"] = adata_diag.layers["X_std_diag"][..., particle]
    adata_diag.layers["pred_disp"] = adata_diag.layers["X_disp_diag"][..., particle]
    df = sc.get.obs_df(
        adata_diag,
        [interv_col, gene, "ncounts", "category"],
        layer="counts" if args["lik"] == "NegBin" else None,
    ).rename(columns={gene: "true"})
    df["perturbed"] = (
        df[interv_col].map(lambda x: gene in x.split(",")).astype("category")
    )
    df["pred_mean"] = sc.get.obs_df(adata_diag, gene, layer="pred_mean")
    df["pred_std"] = sc.get.obs_df(adata_diag, gene, layer="pred_std")
    df["pred_disp"] = sc.get.obs_df(adata_diag, gene, layer="pred_disp")
    split_mask(df, "perturbed", split)

    o1, o2, o3 = Output(), Output(), Output()
    hbox = HBox([o1, o2, o3])
    with o1:
        fig1, ax = kde_scatter("pred_mean", "pred_disp", "perturbed", df)
        if args["lik"] == "NegBin":
            ax.set_xscale("log")
        ax.set_xlabel("Predicted mean")
        ax.set_ylabel("Predicted dispersion")
        ax.set_title(gene)
        plt.close()
        display(fig1)

    with o2:
        fig2, ax = kde_scatter(
            "true", "pred_mean", "perturbed", df, yerr="pred_std" if errorbars else None
        )
        ax.axline((0, 0), slope=1, c="darkred", ls="--")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(gene)
        plt.close()
        display(fig2)

    if args["lik"] != "NegBin":
        display(hbox)
        return

    with o3:
        df["true"] /= df["ncounts"]
        df["pred_mean"] /= df["ncounts"]
        df["pred_std"] /= df["ncounts"]
        fig3, ax = kde_scatter(
            "true", "pred_mean", "perturbed", df, yerr="pred_std" if errorbars else None
        )
        ax.axline((0, 0), slope=1, c="darkred", ls="--")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{gene} (normalized)")
        plt.close()
        display(fig3)
    display(hbox)


# %% [markdown]
# # Jacobian

# %%
mse = np.square(
    np.asarray(adata_diag.layers["X_mean_diag"][..., 3] - adata_diag.layers["counts"])
)

# %%
i, j = np.unravel_index(mse.ravel().argsort()[-20:][::-1], mse.shape)
adata_diag.obs_names[i], adata_diag.var_names[j]

# %%
# # %%debug
adata_jac = model.jacobian(adata[adata_diag.obs_names[i]])
adata_jac


# %%
@interact(vmin=(0, 1, 0.01), vmax=(0, 1, 0.01))
def jacobian(
    particle=list(range(int(args["nptc"]))),
    row_norm=True,
    row_cluster=False,
    col_cluster=False,
    vmin=0,
    vmax=1,
):
    jac = adata_jac.layers["X_jac"][:, :, :, particle].mean(axis=0)
    if row_norm:
        jac /= jac.std(axis=1, keepdims=True) + 1e-7
    sns.clustermap(
        pd.DataFrame(
            jac,
            index=adata.var_names,
            columns=adata.var_names,
        ),
        method="ward",
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        cmap="bwr",
        center=0,
        vmin=np.quantile(jac[jac < 0], vmin),
        vmax=np.quantile(jac[jac > 0], vmax),
        figsize=(12, 10),
    )


# %% [markdown]
# # Counterfactual

# %%
if args["lik"] == "NegBin":
    ctfact.layers["X_ctfact_norm"] = np.log1p(
        ctfact.layers["X_ctfact"]
        * (1e4 / np.asarray(ctfact.obs["ncounts"])[:, np.newaxis, np.newaxis])
    )
else:
    ctfact.layers["X_ctfact_norm"] = ctfact.layers["X_ctfact"]

# %%
ctfact_agg = aggregate_obs(
    ctfact,
    interv_col,
    X_agg=None,
    layers_agg={"X_ctfact_norm": "mean"},
    obs_agg={interv_col: "majority"},
)
adata_agg = aggregate_obs(
    adata,
    interv_col,
    X_agg="mean",
    obs_agg={interv_col: "majority", "category": "majority"},
)


# %% [markdown]
# ## Per perturbation


# %%
@interact
def pair(
    pert=ctfact_agg.obs_names[0],
    particle=list(range(int(args["nptc"]))),
    hue=[
        None,
        *adata_agg.var.columns,
        "in_degree",
        "out_degree",
        "n_ancestors",
        "n_descendants",
        "topo_gen",
    ],
    ascending=False,
):
    df = pd.DataFrame(
        {
            "ctrl": densify(adata_agg.X[adata_agg.obs_names.get_loc(""), :]).ravel(),
            "true": densify(adata_agg.X[adata_agg.obs_names.get_loc(pert), :]).ravel(),
            "pred": densify(
                ctfact_agg.layers["X_ctfact_norm"][
                    ctfact_agg.obs_names.get_loc(pert), :, particle
                ]
            ).ravel(),
        },
        index=adata_agg.var_names,
    ).join([adata_agg.var, node_stats(filter_edges(graphs[particle], cutoff=0.5))])

    o1, o2 = Output(), Output()
    hbox = HBox([o1, o2])
    with o1:
        g = pair_grid(
            df,
            vars=["ctrl", "true", "pred"],
            hue=hue,
            hist_kws={"common_norm": False},
        )
        plt.close()
        display(g.fig)
    with o2:
        df["pred_err"] = (df["pred"] - df["true"]).abs()
        df["true_diff"] = (df["true"] - df["ctrl"]).abs()
        df["rel_err"] = df["pred_err"] / df["true_diff"]
        display(
            df.sort_values("pred_err", ascending=ascending).head(n=20)[
                ["ctrl", "true", "pred", "pred_err", "true_diff", "rel_err"]
            ]
        )
    display(hbox)


# %% [markdown]
# ## Per gene per perturbation


# %%
@interact
def ctfact_pair(
    pert=ctfact_agg.obs_names[0],
    gene=ctfact_agg.var_names[0],
    particle=list(range(int(args["nptc"]))),
):
    ctfact.layers["ctfact_pred"] = ctfact.layers["X_ctfact_norm"][..., particle]
    df = sc.get.obs_df(ctfact, keys=[interv_col, gene], layer="ctfact_pred").rename(
        columns={gene: "pred"}
    )
    df["ctrl"] = sc.get.obs_df(ctfact_input, keys=gene)
    df["perturbed"] = True
    df.query(f"{interv_col} == '{pert}'", inplace=True)

    fig, ax = kde_scatter("ctrl", "pred", "perturbed", df)
    plt.close()
    display(fig)
