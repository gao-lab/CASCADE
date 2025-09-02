from itertools import chain
from typing import Any, Callable

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from scipy.sparse import issparse, spmatrix

from cascade.core import CausalNetwork
from cascade.typing import SimpleGraph


def array_close(arr1: ArrayLike | spmatrix, arr2: ArrayLike | spmatrix) -> bool:
    if issparse(arr1):
        arr1 = arr1.toarray()
    else:
        arr1 = np.asarray(arr1)
    if issparse(arr2):
        arr2 = arr2.toarray()
    else:
        arr2 = np.asarray(arr2)
    return arr1.shape == arr2.shape and np.allclose(arr1, arr2)


def model_close(model1: CausalNetwork, model2: CausalNetwork) -> bool:
    for (k1, p1), (k2, p2) in zip(
        sorted(
            chain(model1.named_parameters(), model1.named_buffers()),
            key=lambda x: x[0],
        ),
        sorted(
            chain(model2.named_parameters(), model2.named_buffers()),
            key=lambda x: x[0],
        ),
    ):
        if k1 != k2 or not array_close(
            p1.detach().cpu().to_dense().numpy(), p2.detach().cpu().to_dense().numpy()
        ):
            print(k1, k2)
            return False
    return True


def graph_cmp(
    graph1: SimpleGraph, graph2: SimpleGraph, edge_attr: str, cmp: Callable[[Any], bool]
) -> bool:
    attrs1 = nx.get_edge_attributes(graph1, edge_attr)
    attrs2 = nx.get_edge_attributes(graph2, edge_attr)
    if attrs1.keys() != attrs2.keys():
        return False
    for key in attrs1.keys():
        if not cmp(attrs1[key], attrs2[key]):
            return False
    return True
