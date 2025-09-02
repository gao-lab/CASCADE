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
import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS

from cascade.plot import set_figure_params

# %%
set_figure_params()


# %%
def kde_scatter(data, x, y, annot=None, adjust=False, ax=None):
    ax = ax or plt.gca()
    xy = np.vstack([data[x], data[y]])
    kde = gaussian_kde(xy)(xy)
    threshold = np.percentile(kde, 5)
    outliers = kde < threshold

    ax = sns.kdeplot(data=data, x=x, y=y, fill=True, levels=20, cmap="Blues", ax=ax)
    ax.scatter(
        data[x][outliers],
        data[y][outliers],
        color="black",
        edgecolor=None,
        s=1,
        alpha=0.5,
    )
    ax.set_title(f"Pearson's r = {data.corr().loc[x, y]:.3f}")

    if annot is not None:
        texts = [
            ax.text(row[x], row[y], index, fontsize="xx-small")
            for index, row in data.reindex(annot).iterrows()
        ]
        if adjust:
            adjust_text(
                texts,
                arrowprops={"arrowstyle": "-"},
                min_arrow_len=5,
                force_text=(1.0, 1.0),
            )
    return ax


def scatter(data, x, y, annot=None, adjust=False, ax=None):
    ax = ax or plt.gca()
    ax = sns.scatterplot(data=data, x=x, y=y, edgecolor=None, s=3, ax=ax)
    if annot is not None:
        texts = [
            ax.text(row[x], row[y], index, fontsize="xx-small")
            for index, row in data.reindex(annot).iterrows()
        ]
        if adjust:
            adjust_text(
                texts,
                arrowprops={"arrowstyle": "-"},
                min_arrow_len=5,
                force_text=(1.0, 1.0),
            )
    return ax


def rscale(data):
    rmin = data.min(axis=1)
    rmax = data.max(axis=1)
    return data.sub(rmin, axis=0).div(rmax - rmin + 1e-7, axis=0)


# %% [markdown]
# # Read data

# %%
genes = pd.read_table(
    "../../data/validation/H1-diff/bulkRNA/resources/human/GENCODE_v37/genes/"
    "gencodeV37_geneid_60710_uniquename_59453.txt"
)
genes.head()

# %%
tpm = (
    pd.read_table(
        "../../data/validation/H1-diff/bulkRNA/results/03_quant/s02_tpm.txt",
        usecols=[
            "gene_id",
            "gene_name",
            "gene_unique_name",
            "20250527-GTZ-1-1",
            "20250527-GTZ-1-2",
            "20250527-GTZ-1-3",
            "20250527-GTZ-1-4",
            "20250527-GTZ-2-1",
            "20250527-GTZ-2-2",
            "20250529-GTZ-1-2",
            "20250602-GTZ-1-1",
            "20250602-GTZ-1-2",
            "20250701-GTZ-1-1",
            "20250701-GTZ-1-2",
            "20250706-GTZ-1-1",
            "20250706-GTZ-1-2",
            "20250716-GTZ-1-1",
            "20250716-GTZ-1-2",
            "20250716-GTZ-1-3",
            "20250716-GTZ-1-4",
        ],
    )
    .merge(genes)
    .query("gene_type == 'protein_coding'")
    .drop(columns=["gene_id", "gene_name", "gene_type"])
    .set_index("gene_unique_name")
    .rename(
        columns={
            "20250527-GTZ-1-1": "PAX6-1",
            "20250527-GTZ-1-2": "hESC-2",
            "20250527-GTZ-1-3": "hESC-3",
            "20250527-GTZ-1-4": "iNPC-d7-2",
            "20250527-GTZ-2-1": "PAX6+TFAP2A-1",
            "20250527-GTZ-2-2": "iNPC-d7-3",
            "20250529-GTZ-1-2": "iNPC-d7-4",
            "20250602-GTZ-1-1": "iNPC-d4-1",
            "20250602-GTZ-1-2": "iNPC-d4-2",
            "20250701-GTZ-1-1": "PAX6+TFAP2A-2",
            "20250701-GTZ-1-2": "PAX6+TFAP2A-3",
            "20250706-GTZ-1-1": "PAX6-2",
            "20250706-GTZ-1-2": "PAX6-3",
            "20250716-GTZ-1-1": "TFAP2A-1",
            "20250716-GTZ-1-2": "TFAP2A-2",
            "20250716-GTZ-1-3": "TFAP2A-3",
            "20250716-GTZ-1-4": "TFAP2A-4",
        }
    )
)
tpm

# %% [markdown]
# ## Pairwise comparisons

# %%
# fmt: off
fig, ax = plt.subplots(figsize=(20, 20))
_ = sns.heatmap(
    np.log1p(
        tpm.loc[:, [
            "hESC-2", "hESC-3",
            "iNPC-d4-1", "iNPC-d4-2",
            "iNPC-d7-2", "iNPC-d7-3", "iNPC-d7-4",
            "PAX6-1", "PAX6-2", "PAX6-3",
            "TFAP2A-1", "TFAP2A-2", "TFAP2A-3", "TFAP2A-4",
            "PAX6+TFAP2A-1", "PAX6+TFAP2A-2", "PAX6+TFAP2A-3",
        ]]
    ).corr(),
    annot=True,
    ax=ax,
)
# fmt: on

# %%
fig, ax = plt.subplots(figsize=(6, 6))
_ = kde_scatter(data=np.log1p(tpm), x="hESC-1", y="hESC-2", ax=ax)

# %% [markdown]
# # Key genes

# %%
cell_type_genes = {}
with open("cell_type_genes.gmt") as f:
    for line in f:
        k, _, *g = line.strip().split("\t")
        cell_type_genes[k] = g

# %% [markdown]
# ## Overexpression

# %%
# fmt: off
overexpress = tpm.loc[
    ["PAX6", "TFAP2A"],
    [
        "hESC-2", "hESC-3",
        "iNPC-d4-1", "iNPC-d4-2",
        "iNPC-d7-2", "iNPC-d7-3", "iNPC-d7-4",
        "PAX6-1", "PAX6-2", "PAX6-3",
        "TFAP2A-1", "TFAP2A-2", "TFAP2A-3", "TFAP2A-4",
        "PAX6+TFAP2A-1", "PAX6+TFAP2A-2", "PAX6+TFAP2A-3",
    ],
]
fig, ax = plt.subplots(figsize=(13, 3))
_ = sns.heatmap(np.log1p(overexpress), ax=ax)
ax.set_ylabel("log TPM")
overexpress
# fmt: on

# %% [markdown]
# ## iNPC

# %%
# fmt: off
inpc = tpm.loc[
    cell_type_genes["NEURAL_PROGENITOR_CELLS"],
    [
        "hESC-2", "hESC-3",
        "iNPC-d4-1", "iNPC-d4-2",
        "iNPC-d7-2", "iNPC-d7-3", "iNPC-d7-4",
    ],
]
fig, ax = plt.subplots(figsize=(3, 12))
_ = sns.heatmap(rscale(np.log1p(inpc)), ax=ax)
_ = ax.set_ylabel("Standardized log TPM")
# fmt: on

# %% [markdown]
# ## iAmacrine

# %%
# fmt: off
iama = tpm.loc[
    cell_type_genes["AMACRINE_CELLS"],
    [
        "hESC-2", "hESC-3",
        "iNPC-d7-2", "iNPC-d7-3", "iNPC-d7-4",
        "PAX6-1", "PAX6-2", "PAX6-3",
        "TFAP2A-1", "TFAP2A-2", "TFAP2A-3", "TFAP2A-4",
        "PAX6+TFAP2A-1", "PAX6+TFAP2A-2", "PAX6+TFAP2A-3",
    ],
]
fig, ax = plt.subplots(figsize=(6, 12))
_ = sns.heatmap(rscale(np.log1p(iama)), ax=ax)
_ = ax.set_ylabel("Standardized log TPM")
# fmt: on

# %%
iama = iama.T
iama["Group"] = iama.index.to_series().str.split(
    r"-(?=[d\d])", regex=True, expand=True
)[0]

# %%
fig, axes = plt.subplots(
    nrows=2, ncols=3, figsize=(5.5, 6), gridspec_kw={"wspace": 0.3, "hspace": 0.3}
)
for i, (g, ax) in enumerate(
    zip(["CHAT", "GAD2", "SLC6A9", "CALB2", "CCK", "NPY"], axes.ravel())
):
    row, col = i // 3, i % 3
    ax = sns.boxplot(
        data=iama, x="Group", hue="Group", y=g, palette="Blues", width=0.7, ax=ax
    )
    ax.set_title(g)
    ax.set_ylabel("TPM")
    if col > 0:
        ax.yaxis.label.set_visible(False)
    if row == 1:
        plt.sca(ax)
        plt.xticks(rotation=60, ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.label.set_visible(False)
fig.savefig("amacrine_markers.pdf")

# %% [markdown]
# # GSEA

# %%
lfc = pd.read_csv("Limma.iAmacrine.vs.iNPC.d7.csv", index_col=0)
lfc

# %%
prerank_custom = gp.prerank(
    rnk=lfc["logFC"].sort_values(),
    gene_sets="cell_type_genes.gmt",
    outdir="gsea-general-custom",
    no_plot=True,
    seed=0,
)

# %%
res2d = prerank_custom.res2d.sort_values("NES", ascending=False)
with pd.option_context("display.min_rows", 40):
    display(res2d)

# %%
prerank_custom.plot(
    terms=[
        "AMACRINE_CELLS",
        "RETINAL_PROGENITORS_AND_MULLER_GLIA",
        "BIPOLAR_CELLS",
        "HORIZONTAL_CELLS",
        "GANGLION_CELLS",
        "NEURAL_PROGENITOR_CELLS",
        "EMBRYONIC_STEM_CELLS",
    ],
    ofname="amacrine_PAX6+TFAP2A_gsea.pdf",
    show_ranking=True,
)

# %% [markdown]
# # Distance analysis

# %%
cao = sc.read_h5ad("../../datasets/Cao-2020.h5ad")

# %%
rnd = np.random.RandomState(42)
subsample_idx = []
for ct, idx in cao.obs.groupby("cell_type", observed=True).indices.items():
    subsample_idx.append(rnd.choice(idx, min(5000, idx.size), replace=False))
subsample_idx = np.concatenate(subsample_idx)
cao_sub = cao[subsample_idx].copy()
cao_sub.write("Cao-2020-sub.h5ad", compression="gzip")
cao_sub


# %%
def cao_cell_type_tpm(cell_type):
    return (
        np.expm1(cao_sub[cao_sub.obs["cell_type"] == cell_type].to_df()).mean(axis=0)
        * 100
    )


ama_tpm = cao_cell_type_tpm("Amacrine cells")

# %%
cmp = tpm.assign(Amacrine=ama_tpm).dropna()
cmp = cmp.div(cmp.sum() / 1e6)
cmp = np.log1p(cmp)
cmp.shape

# %%
cmp = cmp.reindex(
    sorted(
        set(
            cell_type_genes["EMBRYONIC_STEM_CELLS"]
            + cell_type_genes["NEURAL_PROGENITOR_CELLS"]
            + cell_type_genes["AMACRINE_CELLS"]
            + cell_type_genes["RETINAL_PROGENITORS_AND_MULLER_GLIA"]
            + cell_type_genes["BIPOLAR_CELLS"]
            + cell_type_genes["HORIZONTAL_CELLS"]
            + cell_type_genes["GANGLION_CELLS"]
        )
    )
).dropna()
cmp.shape

# %%
dist_mat = pd.DataFrame(
    squareform(pdist(cmp.T)),
    index=cmp.columns,
    columns=cmp.columns,
)

# %%
# fmt: off
fig, ax = plt.subplots(figsize=(15, 15))
sample_order = [
    "hESC-2", "hESC-3",
    "iNPC-d7-2", "iNPC-d7-3", "iNPC-d7-4",
    "PAX6-1", "PAX6-2", "PAX6-3",
    "TFAP2A-1", "TFAP2A-2", "TFAP2A-3", "TFAP2A-4",
    "PAX6+TFAP2A-1", "PAX6+TFAP2A-2", "PAX6+TFAP2A-3",
    "Amacrine",
]
_ = sns.heatmap(dist_mat.loc[sample_order, sample_order], annot=True, fmt=".1f", ax=ax)
_ = plt.yticks(rotation=0)
# fmt: on

# %% [markdown]
# ## MDS

# %%
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    metric=True,
    # n_init=128,
    # max_iter=10000,
    # random_state=42,
)
coords = mds.fit_transform(dist_mat.loc[sample_order, sample_order])

# %%
coords = pd.DataFrame(coords, columns=["MDS1", "MDS2"], index=sample_order)
coords["group"] = coords.index.to_series().str.split(
    r"-(?=[d\d])", regex=True, expand=True
)[0]

# %%
fig, ax = plt.subplots(figsize=(3, 3))
ax = sns.scatterplot(data=coords, x="MDS1", y="MDS2", hue="group", s=70, ax=ax)
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
fig.savefig("amacrine_mds.pdf")
