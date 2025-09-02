# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import importlib

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
from matplotlib import rcParams

# %%
import cascade

# %%
importlib.reload(cascade.plot)
cascade.plot.set_figure_params()

# %% [markdown]
# # Convert format

# %%
adata = sc.read_h5ad("rpe1_raw_singlecell_01.h5ad")

# %%
adata.X = scipy.sparse.csr_matrix(adata.X)

# %%
adata.write("rpe1.h5ad", compression="gzip")

# %%
adata = sc.read_h5ad("K562_essential_raw_singlecell_01.h5ad")

# %%
adata.X = scipy.sparse.csr_matrix(adata.X)

# %%
adata.write("K562_essential.h5ad", compression="gzip")

# %%
adata = sc.read_h5ad("K562_gwps_raw_singlecell_01.h5ad")

# %%
adata.X = scipy.sparse.csr_matrix(adata.X)

# %%
adata.write("K562_gwps.h5ad", compression="gzip")

# %% [markdown]
# # Checks

# %% [markdown]
# ## K562-essential

# %% [markdown]
# ### Number of knockout cells

# %%
adata = sc.read_h5ad("K562_essential.h5ad")
adata

# %%
kd_nums = adata.obs["gene"].value_counts().to_frame().reset_index()
kd_nums.columns = ["gene", "num"]
kd_nums["gene"] = pd.Categorical(kd_nums["gene"], categories=kd_nums["gene"].tolist())
kd_nums.head(n=20)

# %%
rcParams["figure.figsize"] = (10, 5)
kd_nums_top = kd_nums.iloc[1:301].copy()
kd_nums_top["gene"] = pd.Categorical(
    kd_nums_top["gene"], categories=kd_nums_top["gene"].tolist()
)
ax = sns.barplot(data=kd_nums_top, x="gene", y="num")
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Number of cells")
ax.get_figure().savefig("../../../media/221226/Replogle-2022-K562-essential-NCells.png")

# %% [markdown]
# ### Knockdown effect

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata.obs["gene"] = pd.Categorical(
    adata.obs["gene"], categories=kd_nums["gene"].cat.categories
)

# %%
rcParams["figure.figsize"] = (20, 20)
genes = kd_nums["gene"].iloc[1:16].tolist()
detected_genes = set(adata.var["gene_name"])
genes = [gene for gene in genes if gene in detected_genes]
fig = sc.pl.stacked_violin(
    adata[
        np.in1d(
            adata.obs["gene"],
            ["non-targeting", *genes],
        )
    ],
    genes,
    "gene",
    gene_symbols="gene_name",
    swap_axes=True,
    return_fig=True,
)
fig.savefig("../../../media/221226/Replogle-2022-K562-essential-TopKDEffect.png")

# %% [markdown]
# ### Global effect

# %%
tfdb = pd.read_csv(
    "../../download/Lambert-2018/DatabaseExtract_v_1.01.csv", index_col=0
)
tfids = set(tfdb["Ensembl ID"])
tfnames = set(tfdb["HGNC symbol"])

# %%
genes = kd_nums["gene"].iloc[1:301].tolist()

# %%
wt_mean = adata[adata.obs["gene"] == "non-targeting"].X.mean(axis=0).A1
wt_mean

# %%
avg_diff = {}
for gene in genes:
    kd_mean = adata[adata.obs["gene"] == gene].X.mean(axis=0).A1
    avg_diff[gene] = np.abs((kd_mean - wt_mean) / wt_mean).mean()
avg_diff = pd.DataFrame.from_dict(avg_diff, orient="index", columns=["avg_diff"])
avg_diff["gene_name"] = avg_diff.index
avg_diff["is_tf"] = [item in tfnames for item in avg_diff.index]
avg_diff = avg_diff.sort_values("avg_diff", ascending=False)
avg_diff

# %%
rcParams["figure.figsize"] = (10, 5)
avg_diff["gene_name"] = pd.Categorical(
    avg_diff["gene_name"], categories=avg_diff["gene_name"].tolist()
)
ax = sns.barplot(data=avg_diff, x="gene_name", y="avg_diff", hue="is_tf", width=1.0)
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Average global effect")
ax.get_figure().savefig(
    "../../../media/221226/Replogle-2022-K562-essential-TopGlobalEffect.png"
)

# %%
avg_diff["rank"] = np.arange(avg_diff.shape[0]) + 1
avg_diff["rank"].mean(), avg_diff.query("is_tf")["rank"].mean()

# %% [markdown]
# ## RPE1-essential

# %% [markdown]
# ### Number of knockout cells

# %%
adata = sc.read_h5ad("rpe1.h5ad")
adata

# %%
kd_nums = adata.obs["gene"].value_counts().to_frame().reset_index()
kd_nums.columns = ["gene", "num"]
kd_nums["gene"] = pd.Categorical(kd_nums["gene"], categories=kd_nums["gene"].tolist())
kd_nums.head(n=20)

# %%
rcParams["figure.figsize"] = (10, 5)
kd_nums_top = kd_nums.iloc[1:301].copy()
kd_nums_top["gene"] = pd.Categorical(
    kd_nums_top["gene"], categories=kd_nums_top["gene"].tolist()
)
ax = sns.barplot(data=kd_nums_top, x="gene", y="num")
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Number of cells")
ax.get_figure().savefig("../../../media/221226/Replogle-2022-RPE1-essential-NCells.png")

# %% [markdown]
# ### Knockdown effect

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata.obs["gene"] = pd.Categorical(
    adata.obs["gene"], categories=kd_nums["gene"].cat.categories
)

# %%
rcParams["figure.figsize"] = (20, 20)
genes = kd_nums["gene"].iloc[1:16].tolist()
detected_genes = set(adata.var["gene_name"])
genes = [gene for gene in genes if gene in detected_genes]
fig = sc.pl.stacked_violin(
    adata[
        np.in1d(
            adata.obs["gene"],
            ["non-targeting", *genes],
        )
    ],
    genes,
    "gene",
    gene_symbols="gene_name",
    swap_axes=True,
    return_fig=True,
)
fig.savefig("../../../media/221226/Replogle-2022-RPE1-essential-TopKDEffect.png")

# %% [markdown]
# ### Global effect

# %%
genes = kd_nums["gene"].iloc[1:301].tolist()

# %%
wt_mean = adata[adata.obs["gene"] == "non-targeting"].X.mean(axis=0).A1
wt_mean

# %%
avg_diff = {}
for gene in genes:
    kd_mean = adata[adata.obs["gene"] == gene].X.mean(axis=0).A1
    avg_diff[gene] = np.abs((kd_mean - wt_mean) / wt_mean).mean()
avg_diff = pd.DataFrame.from_dict(avg_diff, orient="index", columns=["avg_diff"])
avg_diff["gene_name"] = avg_diff.index
avg_diff["is_tf"] = [item in tfnames for item in avg_diff.index]
avg_diff = avg_diff.sort_values("avg_diff", ascending=False)
avg_diff

# %%
rcParams["figure.figsize"] = (10, 5)
avg_diff["gene_name"] = pd.Categorical(
    avg_diff["gene_name"], categories=avg_diff["gene_name"].tolist()
)
ax = sns.barplot(data=avg_diff, x="gene_name", y="avg_diff", hue="is_tf", width=1.0)
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Average global effect")
ax.get_figure().savefig(
    "../../../media/221226/Replogle-2022-RPE1-essential-TopGlobalEffect.png"
)

# %%
avg_diff["rank"] = np.arange(avg_diff.shape[0]) + 1
avg_diff["rank"].mean(), avg_diff.query("is_tf")["rank"].mean()

# %% [markdown]
# ## K562-GWPS

# %% [markdown]
# ### Number of knockout cells

# %%
adata = sc.read_h5ad("K562_gwps.h5ad")
adata

# %%
kd_nums = adata.obs["gene"].value_counts().to_frame().reset_index()
kd_nums.columns = ["gene", "num"]
kd_nums["gene"] = pd.Categorical(kd_nums["gene"], categories=kd_nums["gene"].tolist())
kd_nums.head(n=20)

# %%
rcParams["figure.figsize"] = (10, 5)
kd_nums_top = kd_nums.iloc[1:301].copy()
kd_nums_top["gene"] = pd.Categorical(
    kd_nums_top["gene"], categories=kd_nums_top["gene"].tolist()
)
ax = sns.barplot(data=kd_nums_top, x="gene", y="num")
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Number of cells")
ax.get_figure().savefig("../../../media/221226/Replogle-2022-K562-gwps-NCells.png")

# %% [markdown]
# ### Knockdown effect

# %%
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# %%
adata.obs["gene"] = pd.Categorical(
    adata.obs["gene"], categories=kd_nums["gene"].cat.categories
)

# %%
rcParams["figure.figsize"] = (20, 20)
genes = kd_nums["gene"].iloc[1:31].tolist()
detected_genes = set(adata.var["gene_name"])
genes = [gene for gene in genes if gene in detected_genes]
fig = sc.pl.stacked_violin(
    adata[
        np.in1d(
            adata.obs["gene"],
            ["non-targeting", *genes],
        )
    ],
    genes,
    "gene",
    gene_symbols="gene_name",
    swap_axes=True,
    return_fig=True,
)
fig.savefig("../../../media/221226/Replogle-2022-K562-gwps-TopKDEffect.png")

# %% [markdown]
# ### Global effect

# %%
genes = kd_nums["gene"].iloc[1:301].tolist()

# %%
wt_mean = adata[adata.obs["gene"] == "non-targeting"].X.mean(axis=0).A1
wt_mean

# %%
avg_diff = {}
for gene in genes:
    kd_mean = adata[adata.obs["gene"] == gene].X.mean(axis=0).A1
    avg_diff[gene] = np.abs((kd_mean - wt_mean) / wt_mean).mean()
avg_diff = pd.DataFrame.from_dict(avg_diff, orient="index", columns=["avg_diff"])
avg_diff["gene_name"] = avg_diff.index
avg_diff["is_tf"] = [item in tfnames for item in avg_diff.index]
avg_diff = avg_diff.sort_values("avg_diff", ascending=False)
avg_diff

# %%
rcParams["figure.figsize"] = (10, 5)
avg_diff["gene_name"] = pd.Categorical(
    avg_diff["gene_name"], categories=avg_diff["gene_name"].tolist()
)
ax = sns.barplot(data=avg_diff, x="gene_name", y="avg_diff", hue="is_tf", width=1.0)
ax.set_xticks([])
ax.set_xlabel("Target gene (top 300)")
ax.set_ylabel("Average global effect")
ax.get_figure().savefig(
    "../../../media/221226/Replogle-2022-K562-gwps-TopGlobalEffect.png"
)

# %%
avg_diff["rank"] = np.arange(avg_diff.shape[0]) + 1
avg_diff["rank"].mean(), avg_diff.query("is_tf")["rank"].mean()
