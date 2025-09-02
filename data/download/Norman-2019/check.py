# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
from scipy.io import mmread
from scipy.sparse import csr_matrix

import cascade

cascade.plot.set_figure_params()
rcParams["figure.figsize"] = (4, 4)

# %% [markdown]
# # GSE

# %%
obs_names_raw = np.loadtxt("GSE133344_raw_barcodes.tsv.gz", dtype="str")
print(obs_names_raw.size)
obs_names_raw

# %%
var_raw = pd.read_table(
    "GSE133344_raw_genes.tsv.gz", names=["gene_id", "gene_name"], index_col="gene_id"
)
print(var_raw.shape)
var_raw.head()

# %%
obs_raw = pd.read_csv("GSE133344_raw_cell_identities.csv.gz", index_col=0)
obs_raw = obs_raw.query("number_of_cells == 1")
print(obs_raw.shape)
obs_raw.head()

# %%
X_raw = mmread("GSE133344_raw_matrix.mtx.gz")
X_raw.shape

# %%
adata_raw = ad.AnnData(
    X=X_raw.T.tocsr()[pd.Index(obs_names_raw).get_indexer(obs_raw.index), :],
    obs=obs_raw,
    var=var_raw,
)
adata_raw

# %% [markdown]
# # GEARS

# %%
adata_gears = ad.read_h5ad("norman/perturb_processed.h5ad")
adata_gears

# %%
adata_gears.layers["counts"] = csr_matrix(adata_gears.layers["counts"])

# %%
adata_gears.obs.head()

# %%
sns.displot(adata_gears.X.expm1().sum(axis=1).A1)

# %% [markdown]
# EOF
