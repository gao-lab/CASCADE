#!/usr/bin/env python

from collections import defaultdict

import pandas as pd
from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import GODag
from matplotlib import rcParams

from cascade.plot import set_figure_params

set_figure_params()
rcParams["axes.grid"] = False

# --------------------------------- Load data ----------------------------------

print("Loading data...")

dag = GODag("go-basic.obo")
gaf_reader = GafReader("goa_human.gaf")
associations = gaf_reader.associations

gene2gos = defaultdict(set)
for assoc in associations:
    # if assoc.NS != "BP":
    #     continue
    gene2gos[assoc.DB_Symbol].add(assoc.GO_ID)
    for synonym in assoc.DB_Synonym:
        gene2gos[synonym].add(assoc.GO_ID)


gene2gos = pd.DataFrame.from_records(
    [(gene, go) for gene, gos in gene2gos.items() for go in gos],
    columns=["gene_name", "go_id"],
)

gene2gos.to_csv("gene2gos.csv.gz", index=False)


# ---------------------------- Propagate GO terms ------------------------------

# print("Propagating GO terms...")


# def propagate(gene_gos_tuple):
#     gene, gos = gene_gos_tuple
#     sub_dag = GoSubDag(gos, dag, prt=None)
#     all_gos = (
#         reduce(or_, (sub_dag.rcntobj.go2ancestors.get(go, set()) for go in gos)) | gos
#     ) - {"GO:0008150", "GO:0003674", "GO:0005575"}
#     return gene, all_gos


# with Pool(processes=16) as pool:
#     gene2all_gos = dict(
#         tqdm(
#             pool.imap(propagate, gene2gos.items(), chunksize=10),
#             total=len(gene2gos),
#         )
#     )


# gene2all_gos = pd.DataFrame.from_records(
#     [(gene, go) for gene, all_gos in gene2all_gos.items() for go in all_gos],
#     columns=["gene_name", "go_id"],
# )

# gene2all_gos.to_csv("gene2all_gos.csv.gz", index=False)


# ----------------------------- Map to gene names ------------------------------

# print("Mapping to gene names...")

# id_mapping = (
#     pd.read_csv(
#         "../others/mart_export.txt.gz",
#         usecols=["Gene stable ID", "UniProtKB Gene Name ID"],
#     )
#     .rename(
#         columns={
#             "Gene stable ID": "gene_id",
#             "UniProtKB Gene Name ID": "uniprot_id",
#         }
#     )
#     .dropna()
#     .drop_duplicates()
# )

# gene2all_gos = (
#     pd.merge(prot2all_gos, id_mapping, on="uniprot_id")
#     .drop(columns="uniprot_id")
#     .sort_values(["gene_id", "go_id"])
#     .drop_duplicates()
# )

# gene2all_gos.to_csv("gene2all_gos.csv.gz", index=False)


# --------------------------- Compute GO similarity ----------------------------

# print("Computing GO similarity...")
#
# gene2gos = gene2gos.groupby("gene_name")["go_id"].apply(set)
# genes = gene2gos.index
# gos = gene2gos.to_numpy()
#
#
# def jaccard(pair):
#     i, j = pair
#     gos_i = gos[i]
#     gos_j = gos[j]
#     gos_both = gos_i & gos_j
#     k = len(gos_both) / (len(gos_i) + len(gos_j) - len(gos_both))
#     return i, j, k
#
#
# go_similarity = np.eye(genes.size)
# with Pool(processes=16) as pool:
#     for i, j, k in tqdm(
#         pool.imap(jaccard, combinations(range(genes.size), 2), chunksize=10000),
#         total=genes.size * (genes.size - 1) // 2,
#     ):
#         go_similarity[i, j] = k
#         go_similarity[j, i] = k
#
# go_similarity = pd.DataFrame(go_similarity, index=genes, columns=genes)
# go_similarity.to_parquet("go_similarity.parquet")


# ---------------------------- Visualize adjacency -----------------------------

# print("Visualizing adjacency...")
#
# gene_linkage = linkage(squareform(1 - go_similarity), method="average")
#
# g = sns.clustermap(
#     go_similarity,
#     row_linkage=gene_linkage,
#     col_linkage=gene_linkage,
#     xticklabels=False,
#     yticklabels=False,
#     vmin=0.0,
#     vmax=0.7,
#     figsize=(20, 20),
# )
# g.savefig("go_similarity.png")


# ------------------------------ Construct graph -------------------------------

# nn = NearestNeighbors(metric="precomputed").fit(1 - go_similarity)
# knn = nn.kneighbors_graph(n_neighbors=500)
# knn = (knn + knn.T + scipy.sparse.eye(knn.shape[0])).astype(bool)
# go_similarity_knn = knn.multiply(go_similarity)
#
# g = sns.clustermap(
#     pd.DataFrame(go_similarity_knn.toarray(), index=genes, columns=genes),
#     row_linkage=gene_linkage,
#     col_linkage=gene_linkage,
#     xticklabels=False,
#     yticklabels=False,
#     vmin=0.0,
#     vmax=0.7,
#     figsize=(20, 20),
# )
# g.savefig("go_similarity_knn.png")
#
# go_similarity_graph = nx.from_scipy_sparse_array(
#     scipy.sparse.coo_array(go_similarity_knn), create_using=nx.DiGraph
# )
# go_similarity_graph = nx.relabel_nodes(go_similarity_graph, dict(enumerate(genes)))
# nx.write_gml(go_similarity_graph, "go_similarity_graph.gml.gz")
