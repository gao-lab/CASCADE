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
import anndata as ad
import seaborn as sns
from matplotlib import rcParams

from cascade.data import aggregate_obs
from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %%
adata = ad.read_h5ad("dat/ds=Norman-2019/imp=20/n_vars=2000/sub.h5ad")
interv_col = "knockup"

# %%
adata = ad.read_h5ad("dat/ds=Replogle-2022-K562-gwps/imp=20/n_vars=2000/sub.h5ad")
interv_col = "knockdown"

# %%
adata_agg = aggregate_obs(adata, interv_col, X_agg="mean").to_df()

# %%
adata_agg.shape

# %%
corr = (adata_agg.drop(index="") - adata_agg.loc[""]).transpose().corr()
corr

# %%
g = sns.clustermap(corr, cmap="viridis", figsize=(100, 100))
g.savefig("interv_clustermap.pdf")
