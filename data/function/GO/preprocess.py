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
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from umap import UMAP

from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # HGNC

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["symbol", "prev_symbol", "alias_symbol", "uniprot_ids"],
    dtype=str,
)
hgnc.head()

# %%
hgnc_map = {}
for symbol in hgnc["symbol"]:
    hgnc_map[symbol] = symbol
for symbol, prev_symbol in zip(hgnc["symbol"], hgnc["prev_symbol"]):
    if isinstance(prev_symbol, str):
        for item in prev_symbol.split("|"):
            if item not in hgnc_map:
                hgnc_map[item] = symbol
for symbol, alias_symbol in zip(hgnc["symbol"], hgnc["alias_symbol"]):
    if isinstance(alias_symbol, str):
        for item in alias_symbol.split("|"):
            if item not in hgnc_map:
                hgnc_map[item] = symbol
for symbol, uniprot_ids in zip(hgnc["symbol"], hgnc["uniprot_ids"]):
    if isinstance(uniprot_ids, str):
        for item in uniprot_ids.split("|"):
            if item not in hgnc_map:
                hgnc_map[item] = symbol

# %% [markdown]
# # Read GAF

# %%
gaf = pd.read_table(
    "goa_human.gaf",
    comment="!",
    dtype=str,
    usecols=[1, 2, 4, 10],
    names=["uniprot_id", "symbol", "term", "synonyms"],
)
gaf.head()


# %%
def map_symbol(row):
    uniprot_id, symbol, term, synonyms = row
    if symbol in hgnc_map:
        return hgnc_map[symbol]
    if uniprot_id in hgnc_map:
        return hgnc_map[uniprot_id]
    if isinstance(synonyms, str):
        for item in synonyms.split("|"):
            if item in hgnc_map:
                return hgnc_map[item]
    return symbol


gaf["symbol"] = gaf.apply(map_symbol, axis=1)
gaf.head()

# %%
gene2gos = gaf[["symbol", "term"]].sort_values(["symbol", "term"]).drop_duplicates()

# %%
gene2gos.to_csv("gene2gos.csv.gz", header=False, index=False)

# %% [markdown]
# # LSI

# %%
gene2gos = gene2gos.pivot_table(
    index="symbol", columns="term", aggfunc=lambda x: 1, fill_value=0
)

# %%
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(gene2gos)

# %%
n_comps = 256
truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
svd = truncated_svd.fit_transform(tfidf)
argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
svd = svd[:, argsort]

# %%
knee = KneeLocator(
    np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
)
exp_var = exp_var_ratio[: knee.knee + 1].sum()
print(f"Identified knee = {knee.knee}")
print(f"Total explained variance = {exp_var}")

# %%
fig, ax = plt.subplots()
ax = sns.lineplot(x=np.arange(n_comps), y=exp_var_ratio, ax=ax)
_ = ax.axvline(knee.knee, color="darkred", linestyle="--")

# %%
svd = pd.DataFrame(svd[:, : knee.knee + 1], index=gene2gos.index)
svd.to_csv("gene2gos_lsi.csv.gz")

# %% [markdown]
# ## Visualization

# %%
umap = UMAP().fit_transform(svd)
umap = pd.DataFrame(umap, index=svd.index, columns=["UMAP1", "UMAP2"])

# %%
fig, ax = plt.subplots(figsize=(10, 10))
_ = sns.scatterplot(data=umap, x="UMAP1", y="UMAP2", edgecolor=None, s=1, ax=ax)
