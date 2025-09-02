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
from collections import deque
from functools import lru_cache, reduce
from io import StringIO
from itertools import chain, product
from operator import add
from pathlib import Path

import networkx as nx
import pandas as pd
import seaborn as sns
from Bio.KEGG.Compound import parse as compound_parse
from Bio.KEGG.Gene import parse as gene_parse
from Bio.KEGG.KGML.KGML_parser import read as kgml_read
from Bio.KEGG.REST import kegg_get, kegg_list
from loguru import logger
from matplotlib import rcParams
from tqdm.auto import tqdm

from cascade.plot import set_figure_params

# %%
set_figure_params()
rcParams["axes.grid"] = False

# %% [markdown]
# # Preparations

# %%
hgnc = pd.read_table(
    "../../others/HGNC/hgnc_complete_set.txt",
    usecols=["hgnc_id", "symbol"],
    dtype=str,
)
hgnc.head()

# %%
hgnc_map = hgnc.set_index("hgnc_id")["symbol"].to_dict()
cache_dir = Path("kegg_cache")
cache_dir.mkdir(parents=True, exist_ok=True)


# %%
def cached_get(entry, option=None):
    cache_file = cache_dir / (entry if option is None else f"{entry}.{option}")
    if not cache_file.exists():
        logger.debug("Calling KEGG GET API...")
        with cache_file.open("w") as f:
            f.write(kegg_get(entry, option=option).read())
    with cache_file.open() as f:
        return StringIO(f.read())


def cached_list(database, org=None):
    cache_file = cache_dir / (database if org is None else f"{database}.{org}")
    if not cache_file.exists():
        logger.debug("Calling KEGG LIST API...")
        with cache_file.open("w") as f:
            f.write(kegg_list(database, org=org).read())
    with cache_file.open() as f:
        return StringIO(f.read())


def gene_name(gene_id):
    try:
        hgnc = dict(next(gene_parse(cached_get(gene_id))).dblinks)["HGNC"]
        assert len(hgnc) == 1
        hgnc = f"HGNC:{hgnc[0]}"
    except KeyError:
        logger.warning(f"No HGNC for {gene_id}")
        hgnc = gene_id
    return hgnc_map.get(hgnc, hgnc)


def compound_name(compound_id):
    return next(compound_parse(cached_get(compound_id))).name[0]


def glycan_name(glycan_id):
    name = next(compound_parse(cached_get(glycan_id))).name
    if name:
        return name[0]
    for line in cached_get(glycan_id):
        if line.startswith("REMARK"):
            match = re.search(r"Same as: (.+)$", line)
            compound_id = f"cpd:{match.group(1).split()[0]}"  # Might be multi
            return compound_name(compound_id)
    else:
        logger.warning(f"No name for {glycan_id}")
        return glycan_id


def drug_name(drug_id):
    return next(compound_parse(cached_get(drug_id))).name[0]


@lru_cache(maxsize=None)
def get_name_type(kegg_id):
    if kegg_id.startswith("hsa:"):
        return gene_name(kegg_id), "gene"
    if kegg_id.startswith("cpd:"):
        return compound_name(kegg_id), "compound"
    if kegg_id.startswith("gl:"):
        return glycan_name(kegg_id), "compound"
    if kegg_id.startswith("dr:"):
        return drug_name(kegg_id), "compound"
    raise ValueError(f"Unknown ID type: {kegg_id}")


def entry_names_types(entry, pathway):
    if entry.type == "group":
        kegg_ids = reduce(
            add,
            (
                pathway.entries[component.id].name.split()
                for component in entry.components
            ),
        )
    else:
        kegg_ids = entry.name.split()
    return [get_name_type(kegg_id) for kegg_id in kegg_ids]


def build_graph(pathway):
    nodes, edges = [], []
    graph = nx.MultiDiGraph()
    for relation in pathway.relations:
        if relation.type == "maplink":
            continue  # Assumes that entry1 already exists in the linked map
        nt1 = entry_names_types(relation.entry1, pathway)
        nt2 = entry_names_types(relation.entry2, pathway)
        for n, t in chain(nt1, nt2):
            attr = {"type": t}
            nodes.append((n, attr))
        for (n1, _), (n2, _) in product(nt1, nt2):
            attr = {"type": relation.type, "subtypes": str(relation.subtypes)}
            edges.append((n1, n2, attr))
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


# %% [markdown]
# # Get all pathways

# %%
pathway_list = pd.read_table(
    cached_list("pathway", org="hsa"),
    names=["pathway_id", "pathway_name"],
)
pathway_list

# %% [markdown]
# # Build pathway graphs

# %% [markdown]
# Manually remove empty files due to API rate excess.

# %%
pathway_graphs = {}
for pathway_id in tqdm(pathway_list["pathway_id"]):
    pathway = kgml_read(cached_get(pathway_id, option="kgml"))
    pathway_graphs[pathway_id] = build_graph(pathway)
    nx.set_edge_attributes(pathway_graphs[pathway_id], pathway.name, name="pathway")

# %%
merged_graph = nx.MultiDiGraph()
nodes = [
    (u, d)
    for u, d in chain(*(graph.nodes.items() for graph in pathway_graphs.values()))
]
edges = [
    (u, v, d)
    for (u, v, _), d in chain(
        *(graph.edges.items() for graph in pathway_graphs.values())
    )
]
merged_graph.add_nodes_from(nodes)
merged_graph.add_edges_from(edges)
merged_graph.number_of_nodes(), merged_graph.number_of_edges()

# %%
merged_graph.graph["data_source"] = "KEGG"
merged_graph.graph["evidence_type"] = "Pathway"
merged_graph.graph["marginalize_steps"] = 1
nx.write_gml(merged_graph, "merged_kegg.gml.gz")

# %% [markdown]
# # Post-processing

# %% [markdown]
# ## Make edges unique

# %%
unique_graph = nx.DiGraph(merged_graph)
unique_graph.number_of_nodes(), unique_graph.number_of_edges()

# %%
unique_graph.graph["data_source"] = "KEGG"
unique_graph.graph["evidence_type"] = "Pathway"
unique_graph.graph["marginalize_steps"] = 1
nx.write_gml(unique_graph, "unique_kegg.gml.gz")


# %% [markdown]
# ## Bridge over compounds


# %%
def bfs_to_gene(graph, start_node, direction):
    queue = deque([start_node])
    visited = set()
    destination = set()
    while queue:
        current_node = queue.popleft()
        if current_node in visited:
            continue
        visited.add(current_node)
        if graph.nodes[current_node]["type"] == "gene":
            if current_node != start_node:
                destination.add(current_node)
            continue
        for pred in getattr(graph, direction)(current_node):
            queue.append(pred)
    return destination


# %%
inferred_graph = unique_graph.copy()
for u, d in unique_graph.nodes.items():
    if d["type"] == "gene":
        continue
    predecessors = bfs_to_gene(unique_graph, u, direction="predecessors")
    successors = bfs_to_gene(unique_graph, u, direction="successors")
    inferred_graph.add_edges_from(
        [(i, j, {"type": "inferred"}) for i, j in product(predecessors, successors)]
    )
inferred_graph.number_of_nodes(), inferred_graph.number_of_edges()

# %%
# inferred_graph.graph["data_source"] = "KEGG"
# inferred_graph.graph["evidence_type"] = "Pathway"
# inferred_graph.graph["marginalize_steps"] = 1
# nx.write_gml(inferred_graph, "inferred_kegg.gml.gz")

# %% [markdown]
# ## Keep genes only

# %%
unique_graph_gene_only = unique_graph.subgraph(
    [n for n, d in unique_graph.nodes.items() if d["type"] == "gene"]
)
unique_graph_gene_only.number_of_nodes(), unique_graph_gene_only.number_of_edges()

# %%
# unique_graph_gene_only.graph["data_source"] = "KEGG"
# unique_graph_gene_only.graph["evidence_type"] = "Pathway"
# unique_graph_gene_only.graph["marginalize_steps"] = 1
# nx.write_gml(unique_graph_gene_only, "unique_kegg_gene_only.gml.gz")

# %%
inferred_graph_gene_only = inferred_graph.subgraph(
    [n for n, d in inferred_graph.nodes.items() if d["type"] == "gene"]
)
inferred_graph_gene_only.number_of_nodes(), inferred_graph_gene_only.number_of_edges()

# %%
inferred_graph_gene_only.graph["data_source"] = "KEGG"
inferred_graph_gene_only.graph["evidence_type"] = "Pathway"
inferred_graph_gene_only.graph["marginalize_steps"] = 1
nx.write_gml(inferred_graph_gene_only, "inferred_kegg_gene_only.gml.gz")

# %% [markdown]
# # Visualization

# %%
adj = nx.to_pandas_adjacency(inferred_graph_gene_only)
top_genes = (adj.sum() + adj.sum(axis=1)).sort_values(ascending=False).head(5000).index
adj_top = adj.loc[top_genes, top_genes]

# %%
g = sns.clustermap(adj_top, cmap="Blues", rasterized=True, figsize=(10, 10))
g.savefig("adj_top.pdf")

# %%
# interactive_heatmap(
#     adj_top.iloc[
#         g.dendrogram_row.reordered_ind[::-1],
#         g.dendrogram_col.reordered_ind],
#     ],
#     row_clust=None,
#     col_clust=None,
#     colorscale="blues",
# )
