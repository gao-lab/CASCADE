r"""
Type definitions
"""

from typing import Any, Mapping, TypeVar

import networkx as nx
import numpy as np

RandomState = np.random.RandomState | int | None
SimpleGraph = TypeVar("SimpleGraph", nx.Graph, nx.DiGraph)
Kws = Mapping[str, Any] | None
