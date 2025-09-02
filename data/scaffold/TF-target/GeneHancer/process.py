# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

# %% [markdown]
# # Connect enhancers with genes

# %%
genehancer = pd.read_table("GeneHancer_v5.17.gff")
genehancer.head()

# %%
genehancer_pairs = []
for line in tqdm(genehancer["attributes"]):
    for block in line.split(";"):
        key, val = block.split("=")
        if key == "genehancer_id":
            genehancer_id = val
        elif key == "connected_gene":
            connected_gene = val
        elif key == "score":
            score = float(val)
            genehancer_pairs.append((genehancer_id, connected_gene, score))

# %%
genehancer_pairs = pd.DataFrame(genehancer_pairs, columns=["GHid", "gene", "score"])
genehancer_pairs

# %% [markdown]
# # Connect enhancers with TFs

# %%
tfbs = pd.read_table("GeneHancer_TFBSs_v5.17.txt")
tfbs

# %%
tf_gene = pd.merge(genehancer_pairs, tfbs, on="GHid")
tf_gene

# %%
tf_gene = tf_gene.loc[:, ["TF", "gene"]].drop_duplicates()
tf_gene

# %%
scaffold = nx.from_pandas_edgelist(
    tf_gene, source="TF", target="gene", create_using=nx.DiGraph
)

# %%
nx.write_gml(scaffold, "scaffold.gml.gz")
