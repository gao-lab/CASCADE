r"""
Miscellaneous utilities
"""

import os
import uuid
from collections import defaultdict
from collections.abc import Hashable, Iterable
from functools import lru_cache
from heapq import nlargest
from multiprocessing import Process, Queue
from os import cpu_count, environ
from random import shuffle
from sys import stderr, stdout

import numpy as np
import pandas as pd
import pynvml
import torch
from loguru import logger
from numpy.typing import ArrayLike
from rich.console import Console
from rich.theme import Theme
from scipy import stats
from scipy.cluster.hierarchy import linkage
from scipy.linalg import pinvh
from scipy.sparse import issparse, spmatrix
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from .typing import RandomState


def internal(obj):
    r"""
    Decorator to mark a function or class as internal and exclude it from
    autosummary.
    """
    obj._internal = True
    return obj


class Config:
    r"""
    Global configurations
    """

    def __init__(self) -> None:
        self.ANNDATA_KEY = environ.get("CASCADE_ANNDATA_KEY", "__CASCADE__")
        self.NUM_WORKERS = environ.get("CASCADE_NUM_WORKERS", min(4, cpu_count()))
        self.PERSISTENT_WORKERS = environ.get("CASCADE_PERSISTENT_WORKERS", True)
        self.DETERMINISTIC = environ.get("CASCADE_DETERMINISTIC", False)
        self.CUDA_REMAP = environ.get("CASCADE_CUDA_REMAP", False)
        self.LOG_STEP_INTERVAL = environ.get("CASCADE_LOG_STEP_INTERVAL", 10)
        self.PBAR_REFRESH = environ.get("CASCADE_PBAR_REFRESH", 10)
        self.CKPT_SAVE_K = environ.get("CASCADE_CKPT_SAVE_K", 1)
        self.MIN_DELTA = environ.get("CASCADE_MIN_DELTA", 0.0)
        self.PATIENCE = environ.get("CASCADE_PATIENCE", 3)
        self.PRECISION = environ.get("CASCADE_PRECISION", "32-true")
        self.LOG_LEVEL = environ.get("CASCADE_LOG_LEVEL", "INFO")
        self.RUN_ID = environ.get("CASCADE_RUN_ID", str(uuid.uuid4()))
        self.DEBUG_FLAG = environ.get("CASCADE_DEBUG_FLAG", False)

    @property
    def ANNDATA_KEY(self) -> str:
        r"""
        Key to store data configuration in :class:`~anndata.AnnData`.
        Default value is ``__CASCADE__``.
        """
        return self._ANNDATA_KEY

    @ANNDATA_KEY.setter
    def ANNDATA_KEY(self, value: str) -> None:
        self._ANNDATA_KEY = value

    @property
    def NUM_WORKERS(self) -> int:
        r"""
        Number of worker processes to use in data loader.
        Default value is ``min(4, n_cpu)``.

        .. tip::

            If training hangs randomly, try setting this option to ``0``,
            or setting ``PERSISTENT_WORKERS = False``.
        """
        return self._NUM_WORKERS

    @NUM_WORKERS.setter
    def NUM_WORKERS(self, value: int | str) -> None:
        self._NUM_WORKERS = int(value)

    @property
    def PERSISTENT_WORKERS(self) -> bool:
        r"""
        Whether to use persistent workers in data loader.
        Default value is ``True``.


        .. tip::

            This option is only effective when ``NUM_WORKERS > 0``.
            If training hangs randomly, try setting this option to ``False``,
            or setting ``NUM_WORKERS = 0``.
        """
        return self._PERSISTENT_WORKERS and self.NUM_WORKERS > 0

    @PERSISTENT_WORKERS.setter
    def PERSISTENT_WORKERS(self, value: bool | str) -> None:
        self._PERSISTENT_WORKERS = (
            str_to_bool(value) if isinstance(value, str) else value
        )

    @property
    def DETERMINISTIC(self) -> bool:
        r"""
        Whether to use deterministic cuDNN implementations.
        Default value is ``False``.
        """
        return self._DETERMINISTIC

    @DETERMINISTIC.setter
    def DETERMINISTIC(self, value: bool | str) -> None:
        self._DETERMINISTIC = str_to_bool(value) if isinstance(value, str) else value

    @property
    def CUDA_REMAP(self) -> bool:
        r"""
        Whether to remap granted CUDA device IDs.
        Default value is ``False``.
        """
        return self._CUDA_REMAP

    @CUDA_REMAP.setter
    def CUDA_REMAP(self, value: bool | str) -> None:
        self._CUDA_REMAP = str_to_bool(value) if isinstance(value, str) else value

    @property
    def LOG_STEP_INTERVAL(self) -> int:
        r"""
        Refresh rate of the training progress bar.
        Default value is ``10``.
        """
        return self._LOG_STEP_INTERVAL

    @LOG_STEP_INTERVAL.setter
    def LOG_STEP_INTERVAL(self, value: int | str) -> None:
        self._LOG_STEP_INTERVAL = int(value)

    @property
    def PBAR_REFRESH(self) -> int:
        r"""
        Refresh rate of the training progress bar.
        Default value is ``10``.
        """
        return self._PBAR_REFRESH

    @PBAR_REFRESH.setter
    def PBAR_REFRESH(self, value: int | str) -> None:
        self._PBAR_REFRESH = int(value)

    @property
    def CKPT_SAVE_K(self) -> int:
        r"""
        Number of top models to save as checkpoints.
        Default values is ``3``.
        """
        return self._CKPT_SAVE_K

    @CKPT_SAVE_K.setter
    def CKPT_SAVE_K(self, value: int | str) -> None:
        self._CKPT_SAVE_K = int(value)

    @property
    def MIN_DELTA(self) -> float:
        r"""
        Minimal score improvement call convergence in earlystopping.
        Default value is ``0.0``.
        """
        return self._MIN_DELTA

    @MIN_DELTA.setter
    def MIN_DELTA(self, value: float | str) -> None:
        self._MIN_DELTA = float(value)

    @property
    def PATIENCE(self) -> int:
        r"""
        Patience to call convergence in earlystopping.
        Default value is ``3``.
        """
        return self._PATIENCE

    @PATIENCE.setter
    def PATIENCE(self, value: int | str) -> None:
        self._PATIENCE = int(value)

    @property
    def PRECISION(self) -> int | str:
        r"""
        Floating point precision.
        Default value is ``32-true``.
        """
        return self._PRECISION

    @PRECISION.setter
    def PRECISION(self, value: int | str = "32-true") -> None:
        precision_map = {
            "bf16": torch.bfloat16,
            "bf16-mixed": torch.bfloat16,
            16: torch.float16,
            "16": torch.float16,
            "16-mixed": torch.float16,
            32: torch.float32,
            "32": torch.float32,
            "32-true": torch.float32,
            64: torch.float64,
            "64": torch.float64,
            "64-true": torch.float64,
        }
        fallback_map = {
            "bf16": torch.float32,
            "bf16-mixed": torch.float32,
            16: torch.float32,
            "16": torch.float32,
            "16-mixed": torch.float32,
            32: torch.float32,
            "32": torch.float32,
            "32-true": torch.float32,
            64: torch.float64,
            "64": torch.float64,
            "64-true": torch.float64,
        }
        torch.set_default_dtype(precision_map[value])
        self._FALLBACK_DTYPE = fallback_map[value]
        self._PRECISION = value

    @property
    def FALLBACK_DTYPE(self) -> torch.dtype:
        return self._FALLBACK_DTYPE

    @property
    def LOG_LEVEL(self) -> str:
        r"""
        Log level.
        Default value is ``"INFO"``.
        """
        return self._LOG_LEVEL

    @LOG_LEVEL.setter
    def LOG_LEVEL(self, value: str) -> None:
        logger.remove()
        logger.add(
            stdout,
            filter=lambda record: record["level"].no < 30,
            level=value,
            format=(
                "<g>{time:HH:mm:ss.SSS}</g> | "
                "<lvl>{level: <8}</lvl> | "
                "<y>{process.id}</y>:<c>{module}</c>:<c>{function}</c> - "
                "<lvl>{message}</lvl>"
            ),
        )
        logger.add(
            stderr,
            filter=lambda record: record["level"].no >= 30,
            level=value,
            format=(
                "<g>{time:HH:mm:ss.SSS}</g> | "
                "<lvl>{level: <8}</lvl> | "
                "<y>{process.id}</y>:<c>{module}</c>:<c>{function}</c> - "
                "<lvl>{message}</lvl>"
            ),
        )
        self._LOG_LEVEL = value

    @property
    def RUN_ID(self) -> str:
        r"""
        A unique UUID for the running session
        """
        return self._RUN_ID

    @RUN_ID.setter
    def RUN_ID(self, value: str) -> None:
        self._RUN_ID = value
        environ["CASCADE_RUN_ID"] = self._RUN_ID

    @property
    def DEBUG_FLAG(self) -> bool:
        r"""
        Convenience utility for setting conditional breakpoints without
        affecting running workflows.
        Default value is ``False``.
        """
        return self._DEBUG_FLAG

    @DEBUG_FLAG.setter
    def DEBUG_FLAG(self, value: bool | str) -> None:
        self._DEBUG_FLAG = str_to_bool(value) if isinstance(value, str) else value


class MissingDependencyError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Please install {name} first.")  # pragma: no cover


def is_notebook() -> bool:  # pragma: no cover
    r"""
    Check if the code is running in a Jupyter notebook

    Returns
    -------
    Whether the code is running in a Jupyter notebook
    """
    try:
        shell = type(get_ipython()).__name__  # type: ignore
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except NameError:
        return False


def str_to_bool(x: str) -> bool:
    r"""
    Interpret string as bool

    Parameters
    ----------
    x
        String to interpret

    Returns
    -------
    Interpreted bool
    """
    if x in ("T", "True", "true", "1"):
        return True
    if x in ("F", "False", "false", "0"):
        return False
    raise ValueError(f"Cannot interpret {x} as bool")


@internal
def non_unitary_index(
    index: (
        int | slice | range | list[int] | tuple[int | slice | range | list[int], ...]
    ),
) -> slice | range | list[int] | tuple[slice | range | list[int], ...]:
    if isinstance(index, tuple):
        return tuple(non_unitary_index(i) for i in index)
    return [index] if isinstance(index, int) else index


@internal
def index_len(
    index: (
        int | slice | range | list[int] | tuple[int | slice | range | list[int], ...]
    ),
    total: int | tuple[int],
) -> int | tuple[int]:
    if not isinstance(index, tuple) and not isinstance(total, tuple):
        if isinstance(index, int):
            return 1
        if isinstance(index, slice):
            index = range(total)[index]
        return len(index)
    if isinstance(index, tuple) and not isinstance(total, tuple):
        raise ValueError("Inconsistent total")
    if not isinstance(index, tuple) and isinstance(total, tuple):
        index = (index,)
    # Now both are tuples
    if len(index) > len(total):
        raise IndexError("Too many indices")
    index = index + (slice(None),) * (len(total) - len(index))
    return tuple(index_len(i, t) for i, t in zip(index, total))


def get_random_state(random_state: RandomState = None) -> np.random.RandomState:
    r"""
    Get a random state object

    Parameters
    ----------
    random_state
        Integer seed, existing :class:`~numpy.random.RandomState` object, or
        None

    Returns
    -------
    Random state object
    """
    if isinstance(random_state, np.random.RandomState):
        return random_state
    return np.random.RandomState(random_state)


def count_occurrence(x: Iterable[Hashable]) -> list[int]:
    r"""
    Count occurrence number of list elements

    Parameters
    ----------
    x
        List of hashable elements

    Returns
    -------
    List of occurrence counts
    """
    counter = defaultdict(int)
    occurrence = []
    for element in x:
        occurrence.append(counter[element])
        counter[element] += 1
    return occurrence


def densify(x: np.ndarray | spmatrix) -> np.ndarray:
    r"""
    Convert a matrix to dense format

    Parameters
    ----------
    x
        Input matrix

    Returns
    -------
    Dense matrix
    """
    return x.toarray() if issparse(x) else x


def variance(x: ArrayLike | spmatrix, bias: bool = False) -> np.ndarray:
    r"""
    Compute variance vector where each column of the input matrix is treated as
    a variable

    Parameters
    ----------
    x
        Input matrix
    bias
        Whether to compute biased variance

    Returns
    -------
    Variance vector
    """
    if issparse(x):
        mean = x.mean(axis=0).A1
        var = x.power(2).mean(axis=0).A1 - mean**2
        if not bias:
            n = x.shape[0]
            var = var * n / (n - 1)
    else:
        var = np.var(np.asarray(x), axis=0, ddof=int(not bias))
    return var


def covariance(x: ArrayLike | spmatrix, bias: bool = False) -> np.ndarray:
    r"""
    Compute covariance matrix where each column of the input matrix is treated
    as a variable

    Parameters
    ----------
    x
        Input matrix
    bias
        Whether to compute biased covariance

    Returns
    -------
    Covariance matrix
    """
    if issparse(x):
        n = x.shape[0]
        mean = x.mean(axis=0).A1
        cov = (x.T @ x).toarray() / n - np.outer(mean, mean)
        if not bias:
            cov = cov * n / (n - 1)
    else:
        cov = np.cov(np.asarray(x), rowvar=False, bias=bias)
    return cov


def pearson_correlation(x: ArrayLike | spmatrix) -> np.ndarray:
    r"""
    Compute Pearson correlation matrix

    Parameters
    ----------
    x
        Input matrix

    Returns
    -------
    Pearson correlation matrix
    """
    cov = covariance(x)
    diag = np.sqrt(np.diag(cov))
    return cov / diag[np.newaxis, :] / diag[:, np.newaxis]


def partial_correlation(x: ArrayLike | spmatrix) -> np.ndarray:
    r"""
    Compute partial correlation matrix

    Parameters
    ----------
    x
        Input matrix

    Returns
    -------
    Partial correlation matrix
    """
    cov = covariance(x)
    prec = pinvh(cov)
    diag = np.sqrt(np.diag(prec))
    return -prec / diag[np.newaxis, :] / diag[:, np.newaxis]


def spearman_correlation(x: ArrayLike | spmatrix) -> np.ndarray:
    r"""
    Compute Spearman correlation matrix

    Parameters
    ----------
    x
        Input matrix

    Returns
    -------
    Spearman correlation matrix
    """
    if issparse(x):
        x = x.toarray()
    x = np.stack([rankdata(col) for col in x.T], axis=1)
    return pearson_correlation(x)


def hclust(
    X: pd.DataFrame,
    metric: str = "euclidean",
    method: str = "complete",
    cut: bool = True,
    **kwargs,
) -> tuple[np.ndarray, pd.Series]:
    r"""
    Hierarchical clustering followed by optional tree cutting

    Parameters
    ----------
    X
        Input data
    metric
        Distance metric
    method
        Clustering method
    cut
        Whether to cut the tree
    **kwargs
        Additional keyword arguments for tree cutting passed to
        :func:`~dynamicTreeCut.cutreeHybrid`

    Returns
    -------
    Linkage matrix
    Cluster labels
    """
    D = pdist(X, metric=metric)
    D[np.isnan(D)] = np.nanmax(D)
    L = linkage(D, method=method)
    if cut:
        try:
            from dynamicTreeCut import cutreeHybrid

            C = cutreeHybrid(L, D, **kwargs)
            C = pd.Series(C["labels"], index=X.index)
        except ImportError:  # pragma: no cover
            logger.warning("dynamicTreeCut not found, skipping tree cut...")
            C = None
    else:
        C = None
    return L, C


def _search_right2left(x: ArrayLike) -> tuple[int, int]:
    r"""
    Search a boolean 1D array from right to left and return the index of the
    first True value :math:`i` and the index of the last False value :math:`j`.

    - If the rightmost value is True, both :math`i` and :math:`j` are the
    rightmost index.
    - If all values are False, both :math`i` and :math:`j` are the
    leftmost index.
    """
    x = np.asarray(x)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Invalid array")
    for i in range(x.size - 1, -1, -1):
        if x[i]:
            return i, min(i + 1, x.size - 1)
    else:
        return i, i


def gp_regression_with_ci(
    data: pd.DataFrame, x: str, y: str, alpha: float = 0.95
) -> tuple[pd.DataFrame, float]:
    r"""
    Gaussian process regression with confidence interval

    Parameters
    ----------
    data
        Input data frame
    x
        Input variable
    y
        Output variable
    alpha
        Confidence level

    Returns
    -------
    Data frame with three additional columns of mean, lower and upper bounds
    Cutoff of input variable that covers minimal output value in CI
    """
    data_clean = data.dropna()
    gp = GaussianProcessRegressor(kernel=RBF() + WhiteKernel())
    gp.fit(np.asarray(data_clean[[x]]), np.asarray(data_clean[y]))
    y_mean, y_std = gp.predict(np.asarray(data[[x]]), return_std=True)
    lower = stats.norm.ppf((1 - alpha) / 2)
    upper = stats.norm.ppf(1 - (1 - alpha) / 2)
    y_lower = y_mean + lower * y_std
    y_upper = y_mean + upper * y_std

    x_min, x_max, y_min = data[x].min(), data[x].max(), data[y].min()
    x_ = np.linspace(x_min, x_max, 1000)
    y_mean_, y_std_ = gp.predict(x_[:, np.newaxis], return_std=True)
    y_lower_ = y_mean_ + lower * y_std_
    cut_left, cut_right = _search_right2left(y_lower_ > y_min)
    cutoff = (x_[cut_left] + x_[cut_right]) / 2

    return (
        data.assign(
            **{f"{y}_mean": y_mean, f"{y}_lower": y_lower, f"{y}_upper": y_upper}
        ),
        cutoff,
    )


@lru_cache
def autodevice(n: int = 1) -> tuple[str, list[int] | str]:
    r"""
    Get torch computation device automatically based on GPU availability and
    memory usage

    Parameters
    ----------
    n
        Number of GPUs to request

    Returns
    -------
    Accelerator type
    List of granted devices
    """
    granted = environ.get("CASCADE_GRANTED", None)
    if granted is not None:
        granted = [int(i) for i in granted.split(",")]
        if len(granted) == n:  # DDP member
            logger.info("Using GPU {} as computation device.", granted)
            return "gpu", granted
    try:
        pynvml.nvmlInit()
        devices = environ.get("CUDA_VISIBLE_DEVICES", None)
        devices = (
            list(range(pynvml.nvmlDeviceGetCount()))
            if devices is None
            else [int(d.strip()) for d in devices.split(",") if d != ""]
        )
        shuffle(devices)
        free_mems = {
            i: pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).free
            for i in devices
        }
        granted = nlargest(n, free_mems, free_mems.get)
        if len(granted) == 0:
            raise pynvml.NVMLError("GPU disabled.")
        if config.CUDA_REMAP:
            logger.info("Remapping CUDA device IDs.")
            granted = list(range(len(granted)))
        logger.info("Using GPU {} as computation device.", granted)
        environ["CASCADE_GRANTED"] = ",".join(str(i) for i in granted)
        return "gpu", granted
    except pynvml.NVMLError:  # pragma: no cover
        logger.info("Using CPU as computation device.")
        return "cpu", "auto"


def _subprocess_affinity(q: Queue) -> None:
    q.put(os.sched_getaffinity(0))


def check_affinity_inheritance() -> None:
    r"""
    Check whether affinity inheritance is broken
    """
    q = Queue()
    p = Process(target=_subprocess_affinity, args=(q,))
    p.start()
    p.join()
    if q.get() != os.sched_getaffinity(0):
        logger.warning(
            "Affinity inheritance is broken! This might be related to "
            "https://github.com/pytorch/pytorch/issues/99625. "
            "Consider setting environment variable KMP_AFFINITY=disabled. "
            "Otherwise, performance would be compromised."
        )


config = Config()
console = Console(theme=Theme({"hl": "bold magenta"}))
