import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

print("Reading gene annotation...")
gene2gos = pd.read_csv("gene2gos.csv.gz")
gene2gos = gene2gos.pivot_table(
    index="gene_name", columns="go_id", aggfunc=lambda x: 1, fill_value=0
)

print("Performing TF-IDF normalization...")
tfidf_transformer = TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(gene2gos)

n_comps = 256
print(f"Running SVD up to {n_comps} components...")
truncated_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
svd = truncated_svd.fit_transform(tfidf)
argsort = np.argsort(truncated_svd.explained_variance_ratio_)[::-1]
exp_var_ratio = truncated_svd.explained_variance_ratio_[argsort]
svd = svd[:, argsort]

knee = KneeLocator(
    np.arange(n_comps), exp_var_ratio, curve="convex", direction="decreasing"
)
exp_var = exp_var_ratio[: knee.knee + 1].sum()
print(f"Identified knee = {knee.knee}")
print(f"Total explained variance = {exp_var}")

fig, ax = plt.subplots()
ax = sns.lineplot(x=np.arange(n_comps), y=exp_var_ratio, ax=ax)
ax.axvline(knee.knee, color="darkred", linestyle="--")
fig.savefig("gene2gos_lsi.png")

print("Saving result...")
svd = pd.DataFrame(svd[:, : knee.knee + 1], index=gene2gos.index)
svd.to_csv("gene2gos_lsi.csv.gz")
