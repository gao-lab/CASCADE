r"""
Causality-Aware Single-Cell Adaptive Discover/Deduction/Design Engine
"""

try:
    from importlib.metadata import version
except ModuleNotFoundError:  # pragma: no cover
    from pkg_resources import get_distribution

    def version(name: str) -> str:
        return get_distribution(name).version


name = "cascade-reg"
__version__ = version(name)

from .utils import check_affinity_inheritance, config

check_affinity_inheritance()
