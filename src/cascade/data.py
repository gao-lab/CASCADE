r"""
Dataset processing utilities and data loaders
"""

import re
from collections.abc import Iterable, Mapping
from functools import reduce
from operator import or_

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from loguru import logger
from pytorch_lightning import LightningDataModule
from scipy.sparse import csr_matrix, issparse, lil_matrix, spmatrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

from .typing import RandomState
from .utils import config, get_random_state, internal

EPS: float = 1e-7


class Targets(frozenset[str]):
    r"""
    Intervention targets

    Parameters
    ----------
    targets
        A string of comma-separated target names


    .. note::

        Use empty string for no targets
    """

    SEP = re.compile(r"\s*,\s*")

    def __new__(cls, targets: str | Iterable[str] | None = None) -> None:
        if isinstance(targets, str):
            targets = (item for item in re.split(cls.SEP, targets) if item)
        elif targets is None:
            targets = ()
        return super().__new__(cls, targets)

    def __lt__(self, other: "Targets") -> bool:
        if len(self) < len(other):
            return True
        if len(self) > len(other):
            return False
        for i, j in zip(sorted(self), sorted(other)):
            if i < j:
                return True
        return False

    def __gt__(self, other: "Targets") -> bool:
        if len(self) > len(other):
            return True
        if len(self) < len(other):
            return False
        for i, j in zip(sorted(self), sorted(other)):
            if i > j:
                return True
        return False

    def __le__(self, other: "Targets") -> bool:
        return self == other or self < other

    def __ge__(self, other: "Targets") -> bool:
        return self == other or self > other

    def __and__(self, other: Iterable[str]) -> "Targets":
        return Targets(super().__and__(set(other)))

    def __or__(self, other: Iterable[str]) -> "Targets":
        return Targets(super().__or__(set(other)))

    def __xor__(self, other: Iterable[str]) -> "Targets":
        return Targets(super().__xor__(set(other)))

    def __sub__(self, other: Iterable[str]) -> "Targets":
        return Targets(super().__sub__(set(other)))

    def __repr__(self) -> str:
        return ",".join(sorted(self))


def get_all_targets(adata: AnnData, key: str) -> Targets:
    r"""
    Get the union of all intervention targets in a give dataset

    Parameters
    ----------
    adata
        Input dataset
    key
        Key in :attr:`~anndata.AnnData.obs` containing comma-separated
        intervention targets

    Returns
    -------
    Union of all intervention targets
    """
    return reduce(or_, np.vectorize(Targets)(adata.obs[key].unique()))


def filter_unobserved_targets(adata: AnnData, key: str) -> AnnData:
    r"""
    Filter cells where the intervention targets are not observed (missing from
    :attr:`~anndata.AnnData.var_names`)

    Parameters
    ----------
    adata
        Input dataset
    key
        Key in :attr:`~anndata.AnnData.obs` containing comma-separated
        intervention targets

    Returns
    -------
    Filtered dataset
    """
    all_vars = set(adata.var_names)
    unobserved = {k: bool(Targets(k) - all_vars) for k in adata.obs[key].unique()}
    return adata[~adata.obs[key].map(unobserved)]


def encode_regime(adata: AnnData, layer: str, key: str = None) -> None:
    r"""
    Encode intervention regime

    Parameters
    ----------
    adata
        Input dataset
    layer
        Layer in :attr:`~anndata.AnnData.layers` to write the encoded regime
    key
        Column in :attr:`~anndata.AnnData.obs` containing comma-separated
        intervention targets following :class:`Targets` format
    """
    regime = lil_matrix(adata.shape, dtype=bool)
    if key is not None:
        targets = adata.obs[key].map(Targets)
        rows = np.concatenate([np.repeat(i, len(t)) for i, t in enumerate(targets)])
        cols = np.concatenate([adata.var_names.get_indexer(t) for t in targets])
        if cols.size and cols.min() < 0:
            raise ValueError("Invalid intervention target")
        regime[rows, cols] = True
    if layer in adata.layers:
        logger.warning(f'Overwriting existing regime "{layer}".')
    adata.layers[layer] = regime.tocsr().astype(bool)


def configure_dataset(
    adata: AnnData,
    use_regime: str | None = None,
    use_covariate: str | None = None,
    use_size: str | None = None,
    use_weight: str | None = None,
    use_layer: str | None = None,
) -> None:
    r"""
    Configure dataset for model training and inferences

    Parameters
    ----------
    adata
        Input dataset
    use_regime
        Key in :attr:`~anndata.AnnData.layers` containing intervention regime
        encoded with :func:`encode_regime`
    use_covariate
        Key in :attr:`~anndata.AnnData.obsm` containing covariates
    use_size
        Key in :attr:`~anndata.AnnData.obs` containing cell sizes
    use_weight
        Key in :attr:`~anndata.AnnData.obs` containing cell weights
    use_layer
        Key in :attr:`~anndata.AnnData.layers` containing data to be used for
        training and inferences, instead of :attr:`~anndata.AnnData.X`


    .. note::

        When called multiple times, individual configuration items will be
        overwritten if and only if new values are not ``None``.
    """
    configuration = adata.uns.setdefault(
        config.ANNDATA_KEY,
        {
            "regime": None,
            "covariate": None,
            "size": None,
            "weight": None,
            "layer": None,
        },
    )
    if use_regime is not None:
        if use_regime not in adata.layers:
            raise KeyError(f"{use_regime} not found in `adata.layers`")
        prev = configuration.get("regime", None)
        if prev is not None and prev != use_regime:
            logger.warning(f'Overwriting existing `regime` = "{prev}".')
        configuration["regime"] = use_regime
    if use_covariate is not None:
        if use_covariate not in adata.obsm:
            raise KeyError(f"{use_covariate} not found in `adata.obsm`")
        prev = configuration.get("covariate", None)
        if prev is not None and prev != use_covariate:
            logger.warning(f'Overwriting existing `covariate` = "{prev}".')
        configuration["covariate"] = use_covariate
    if use_size is not None:
        if use_size not in adata.obs:
            raise KeyError(f"{use_size} not found in `adata.obs`")
        prev = configuration.get("size", None)
        if prev is not None and prev != use_size:
            logger.warning(f'Overwriting existing `size` = "{prev}".')
        configuration["size"] = use_size
    if use_weight is not None:
        if use_weight not in adata.obs:
            raise KeyError(f"{use_weight} not found in `adata.obs`")
        prev = configuration.get("weight", None)
        if prev is not None and prev != use_weight:
            logger.warning(f'Overwriting existing `weight` = "{prev}".')
        configuration["weight"] = use_weight
    if use_layer is not None:
        if use_layer not in adata.layers:
            raise KeyError(f"{use_layer} not found in `adata.layers`")
        prev = configuration.get("layer", None)
        if prev is not None and prev != use_layer:
            logger.warning(f'Overwriting existing `layer` = "{prev}".')
        configuration["layer"] = use_layer


def get_configuration(adata: AnnData) -> dict[str, str]:
    r"""
    Retrieve the configuration by :func:`configure_dataset`

    Parameters
    ----------
    adata
        Input dataset

    Returns
    -------
    Configuration dictionary
    """
    if config.ANNDATA_KEY not in adata.uns:
        raise KeyError(
            "Dataset not configured yet, please call `configure_dataset` first."
        )
    return adata.uns[config.ANNDATA_KEY]


def _get_X(adata: AnnData) -> csr_matrix | np.ndarray:
    key = get_configuration(adata).get("layer", None)
    if key is None:
        logger.debug("Dataset not configured with `layer`, using `adata.X`.")
        X = adata.X
    else:
        logger.debug(f"Using configured `layer`: {key}.")
        X = adata.layers[key]
    return X.tocsr() if issparse(X) else X


def _set_X(adata: AnnData, X: spmatrix | np.ndarray) -> None:
    key = get_configuration(adata).get("layer", None)
    if key is None:
        logger.debug("Dataset not configured with `layer`, using `adata.X`.")
        adata.X = X
    else:
        logger.debug(f"Using configured `layer`: {key}.")
        adata.layers[key] = X


def _get_regime(adata: AnnData) -> csr_matrix:
    key = get_configuration(adata).get("regime", None)
    if key is None:
        logger.debug("Dataset not configured with `regime`, assuming observational.")
        return csr_matrix(adata.shape, dtype=bool)
    logger.debug(f"Using configured `regime`: {key}.")
    regime = adata.layers[key].tocsr()
    regime.eliminate_zeros()
    return regime


def _set_regime(adata: AnnData, regime: spmatrix) -> None:
    key = get_configuration(adata).get("regime", None)
    if key is None:
        raise ValueError("Dataset not configured with `regime`")
    logger.debug(f"Using configured `regime`: {key}.")
    adata.layers[key] = regime


def _get_covariate(adata: AnnData) -> np.ndarray:
    key = get_configuration(adata).get("covariate", None)
    if key is None:
        logger.debug("Dataset not configured with `covariate`, ignoring.")
        return np.empty((adata.n_obs, 0))
    logger.debug(f"Using configured `covariate`: {key}.")
    return np.asarray(adata.obsm[key])


def _set_covariate(adata: AnnData, covariate: np.ndarray) -> None:
    key = get_configuration(adata).get("covariate", None)
    if key is None:
        raise ValueError("Dataset not configured with `covariate`")
    logger.debug(f"Using configured `covariate`: {key}.")
    adata.obsm[key] = covariate


def _get_size(adata: AnnData) -> np.ndarray:
    key = get_configuration(adata).get("size", None)
    if key is None:
        logger.debug("Dataset not configured with `size`, ignoring.")
        return np.empty((adata.n_obs, 0))
    logger.debug(f"Using configured `size`: {key}.")
    return np.asarray(adata.obs[[key]])


def _set_size(adata: AnnData, covariate: np.ndarray) -> None:
    key = get_configuration(adata).get("size", None)
    if key is None:
        raise ValueError("Dataset not configured with `size`")
    logger.debug(f"Using configured `size`: {key}.")
    adata.obs[key] = covariate.ravel()


def _get_weight(adata: AnnData) -> np.ndarray:
    key = get_configuration(adata).get("weight", None)
    if key is None:
        logger.debug("Dataset not configured with `weight`, using unitary.")
        return np.ones(adata.n_obs)
    logger.debug(f"Using configured `weight`: {key}.")
    weight = np.asarray(adata.obs[key])
    weight = (weight.size / weight.sum()) * weight
    return weight


def _set_weight(adata: AnnData, weight: np.ndarray) -> None:
    key = get_configuration(adata).get("weight", None)
    if key is None:
        raise ValueError("Dataset not configured with `weight`")
    logger.debug(f"Using configured `weight`: {key}.")
    adata.obs[key] = weight.ravel()


def neighbor_impute(
    adata: AnnData,
    k: int,
    use_rep: str,
    use_batch: str | None = None,
    X_agg: str | None = "sum",
    obs_agg: Mapping[str, str] | None = None,
    obsm_agg: Mapping[str, str] | None = None,
    layers_agg: Mapping[str, str] | None = None,
) -> AnnData:
    r"""
    Impute data by aggregating nearest neighbors

    Parameters
    ----------
    adata
        Dataset to be imputed
    k
        Number of nearest neighbors
    use_rep
        Key in :attr:`~anndata.AnnData.obsm` containing the representation to be
        used for nearest neighbor search
    use_batch
        Key in :attr:`~anndata.AnnData.obs` used to group cells for nearest
        neighbor search (e.g., intervention label)
    X_agg
        Aggregation function for :attr:`~anndata.AnnData.X`, must be one of
        ``{"sum", "mean", ``None``}``. Setting to ``None`` discards the
        :attr:`~anndata.AnnData.X` matrix.
    obs_agg
        Aggregation methods for :attr:`~anndata.AnnData.obs`, indexed by obs
        columns, must be one of ``{"sum", "mean"}``. Fields not specified will
        be discarded.
    obsm_agg
        Aggregation methods for :attr:`~anndata.AnnData.obsm`, indexed by obsm
        keys, must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for :attr:`~anndata.AnnData.layers`, indexed by
        layer keys, must be one of ``{"sum", "mean"}``. Fields not specified
        will be discarded.

    Returns
    -------
    Imputed dataset
    """
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    rows, cols = [], []
    groupby = adata.obs.groupby(use_batch or np.zeros(adata.n_obs), observed=True)
    for idx in groupby.indices.values():
        rep = adata[idx].obsm[use_rep]
        knn = NearestNeighbors().fit(rep)
        knn = knn.kneighbors(rep, min(k, idx.size), return_distance=False)
        rows.append(np.repeat(idx, knn.shape[1]))
        cols.append(idx[knn].ravel())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    agg_sum = csr_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(adata.n_obs, adata.n_obs)
    )
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: (agg_sum @ x.reshape(x.shape[0], -1)).reshape(
            -1, *x.shape[1:]
        ),
        "mean": lambda x: (agg_mean @ x.reshape(x.shape[0], -1)).reshape(
            -1, *x.shape[1:]
        ),
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame(
        {
            k: agg_method[obs_agg[k]](v.to_numpy()) if k in obs_agg else v
            for k, v in adata.obs.items()
        },
        index=adata.obs_names,
    )
    obsm = {
        k: agg_method[obsm_agg[k]](v) if k in obsm_agg else v
        for k, v in adata.obsm.items()
    }
    layers = {
        k: agg_method[layers_agg[k]](v) if k in layers_agg else v
        for k, v in adata.layers.items()
    }
    return AnnData(
        X=X,
        obs=obs,
        var=adata.var,
        uns=adata.uns,
        obsm=obsm,
        varm=adata.varm,
        layers=layers,
    )


def aggregate_obs(
    adata: AnnData,
    by: str,
    X_agg: str | None = None,
    obs_agg: Mapping[str, str] | None = None,
    obsm_agg: Mapping[str, str] | None = None,
    layers_agg: Mapping[str, str] | None = None,
) -> AnnData:
    r"""
    Aggregate obs in a given dataset by certain categories

    Parameters
    ----------
    adata
        Dataset to be aggregated
    by
        Specify a column in :attr:`~anndata.AnnData.obs` used for aggregation,
        must be discrete.
    X_agg
        Aggregation function for :attr:`~anndata.AnnData.X`, must be one of
        ``{"sum", "mean", ``None``}``. Setting to ``None`` discards the
        ``adata.X`` matrix.
    obs_agg
        Aggregation methods for :attr:`~anndata.AnnData.obs`, indexed by obs
        columns, must be one of ``{"sum", "mean", "majority"}``, where ``"sum"``
        and ``"mean"`` are for continuous data, and ``"majority"`` is for
        discrete data. Fields not specified will be discarded.
    obsm_agg
        Aggregation methods for :attr:`~anndata.AnnData.obsm`, indexed by obsm
        keys, must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for :attr:`~anndata.AnnData.layers`, indexed by
        layer keys, must be one of ``{"sum", "mean"}``. Fields not specified
        will be discarded.

    Returns
    -------
    Aggregated dataset
    """
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.unique())
    agg_sum = csr_matrix(
        (np.ones(adata.n_obs), (agg_idx.get_indexer(by), np.arange(adata.n_obs)))
    )
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: (agg_sum @ x.reshape(x.shape[0], -1)).reshape(
            -1, *x.shape[1:]
        ),
        "mean": lambda x: (agg_mean @ x.reshape(x.shape[0], -1)).reshape(
            -1, *x.shape[1:]
        ),
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy(),
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame(
        {k: agg_method[v](adata.obs[k].to_numpy()) for k, v in obs_agg.items()},
        index=agg_idx.astype(str),
    )
    obsm = {k: agg_method[v](adata.obsm[k]) for k, v in obsm_agg.items()}
    layers = {k: agg_method[v](adata.layers[k]) for k, v in layers_agg.items()}
    for c in obs:
        if isinstance(adata.obs[c].dtype, pd.CategoricalDtype):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X, obs=obs, var=adata.var, obsm=obsm, varm=adata.varm, layers=layers
    )


def simple_design(
    interv: AnnData, target: AnnData, key: str, target_weight: str | None = None
) -> pd.DataFrame:
    r"""
    Perform simple intervention design by directly comparing the outcome of
    seen interventions with the target

    Parameters
    ----------
    interv
        Interventional data
    target
        Target data
    key
        Column in ``interv.obs`` containing comma-separated intervention targets
        following :class:`Targets` format
    target_weight
        Variable weights for computing "mse" deviation with target

    Returns
    -------
    Simple design with an "mse" column sorted by ascending order
    """
    if key in target.obs:
        raise ValueError(f"`target.obs` must not contain '{key}'")  # pragma: no cover
    target.obs[key] = "target"
    interv_agg = aggregate_obs(interv, by=key, X_agg="mean").to_df()
    target_agg = aggregate_obs(target, by=key, X_agg="mean").to_df().iloc[0]
    del target.obs[key]
    if target_weight is not None:
        weight = target.var[target_weight]
        weight = weight.size * weight / weight.sum()
    else:
        weight = pd.Series(1, index=target.var_names)
    mse = interv_agg.sub(target_agg).pow(2).mul(weight).mean(axis=1)
    return pd.DataFrame({"mse": mse}).sort_values("mse", kind="stable")


class SimpleDataset(Dataset):
    r"""
    A single interventional dataset

    Parameters
    ----------
    adata
        Interventional dataset
    """

    def __init__(self, adata: AnnData) -> None:
        self.n = adata.n_obs
        self.x = _get_X(adata)
        self.r = _get_regime(adata)
        self.s = _get_covariate(adata)
        self.l = _get_size(adata)
        self.w = _get_weight(adata)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        default_dtype = torch.get_default_dtype()
        x = torch.as_tensor(
            self.x[index].toarray().squeeze(0) if issparse(self.x) else self.x[index],
            dtype=default_dtype,
        )
        r = torch.as_tensor(self.r[index].toarray().squeeze(0), dtype=default_dtype)
        s = torch.as_tensor(self.s[index], dtype=default_dtype)
        l = torch.as_tensor(self.l[index], dtype=default_dtype)
        w = torch.as_tensor(self.w[index], dtype=default_dtype)
        return [x, r, s, l, w]


class PairedDataset(Dataset):
    r"""
    A pair of interventional datasets that are paired cell-by-cell

    Parameters
    ----------
    pri
        Primary interventional dataset
    sec
        Secondary interventional dataset
    """

    def __init__(self, pri: SimpleDataset, sec: SimpleDataset) -> None:
        if len(pri) != len(sec):
            raise ValueError("Datasets must have the same size")
        self.pri = pri
        self.sec = sec

    def __len__(self) -> int:
        return len(self.pri)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        return [*self.pri[index], *self.sec[index]]


class DynamicPairedDataset(Dataset):
    r"""
    A pair of interventional datasets that are not paired but fetches randomly
    paired cells on-the-fly

    Parameters
    ----------
    pri
        Primary interventional dataset
    sec
        Secondary interventional dataset
    pri_strat
        Stratification of the primary dataset
    sec_strat
        Stratification of the secondary dataset
    random_state
        Random state
    """

    def __init__(
        self,
        pri: SimpleDataset,
        sec: SimpleDataset,
        pri_strat: np.ndarray,
        sec_strat: np.ndarray,
        random_state: RandomState,
    ) -> None:
        self.pri = pri
        self.sec = sec
        pri_strat_set = set(pri_strat)
        sec_strat_set = set(sec_strat)
        if pri_strat_set != sec_strat_set:
            raise ValueError("Primary and secondary stratifications do not match")
        lut = {s: np.where(sec_strat == s)[0] for s in pri_strat_set}
        self.lut = {i: lut[s] for i, s in enumerate(pri_strat)}
        self.rnd = get_random_state(random_state)

    def __len__(self) -> int:
        return len(self.pri)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        i, j = index, self.rnd.choice(self.lut[index])
        return [*self.pri[i], *self.sec[j]]


class DataModule(LightningDataModule):
    r"""
    Abstract data module

    Parameters
    ----------
    batch_size
        Batch size
    pin_memory
        Whether to use pin memory
    val_frac
        Fraction of validation data
    random_state
        Random state
    """

    def __init__(
        self,
        batch_size: int,
        pin_memory: bool,
        val_frac: float,
        random_state: RandomState,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.val_frac = val_frac
        self.rnd = get_random_state(random_state)
        self.train = None
        self.val = None
        self.predict = None

    def __len__(self) -> int:
        raise NotImplementedError  # pragma: no cover

    @internal
    def setup(self, stage: str) -> None:
        raise NotImplementedError  # pragma: no cover

    @internal
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=config.PERSISTENT_WORKERS,
        )

    @internal
    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=config.PERSISTENT_WORKERS,
        )

    @internal
    def predict_dataloader(self):
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=config.PERSISTENT_WORKERS,
        )


class SimpleDataModule(DataModule):
    r"""
    Simple data module using :class:`SimpleDataset`

    Parameters
    ----------
    adata
        Interventional dataset
    batch_size
        Batch size
    pin_memory
        Whether to use pin memory
    val_frac
        Fraction of validation data
    random_state
        Random state
    """

    def __init__(
        self,
        adata: AnnData,
        batch_size: int,
        pin_memory: bool,
        val_frac: float,
        random_state: RandomState,
    ) -> None:
        super().__init__(batch_size, pin_memory, val_frac, random_state)
        self.adata = adata
        if 0 < self.val_frac < 1:
            if adata.n_obs == 1:
                self.train_idx = self.val_idx = np.arange(adata.n_obs)
            elif adata.n_obs > 1:
                self.train_idx, self.val_idx = train_test_split(
                    np.arange(adata.n_obs),
                    test_size=self.val_frac,
                    random_state=self.rnd,
                )
            else:
                raise ValueError("Dataset cannot be empty")
        elif self.val_frac == 0:
            self.train_idx = np.arange(self.adata.n_obs)
            self.val_idx = np.empty(0, dtype=self.train_idx.dtype)
        else:
            raise ValueError("Invalid validation fraction")  # pragma: no cover

    def __len__(self) -> int:
        return self.adata.n_obs

    @internal
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train is None:
                self.train = SimpleDataset(self.adata[self.train_idx])
            if self.val is None:
                self.val = SimpleDataset(self.adata[self.val_idx])
        if stage == "predict" and self.predict is None:
            self.predict = SimpleDataset(self.adata)


class PairedDataModule(DataModule):
    r"""
    Paired data module using :class:`PairedDataset`

    Parameters
    ----------
    pri
        Primary interventional dataset
    sec
        Secondary interventional dataset
    batch_size
        Batch size
    pin_memory
        Whether to use pin memory
    val_frac
        Fraction of validation data
    random_state
        Random state
    """

    def __init__(
        self,
        pri: AnnData,
        sec: AnnData,
        batch_size: int,
        pin_memory: bool,
        val_frac: float,
        random_state: RandomState,
    ) -> None:
        super().__init__(batch_size, pin_memory, val_frac, random_state)
        if pri.n_obs != sec.n_obs:
            raise ValueError("Datasets must have the same size")
        self.pri = pri
        self.sec = sec
        if 0 < self.val_frac < 1:
            if self.pri.n_obs == 1:
                self.train_idx = self.val_idx = np.arange(self.pri.n_obs)
            elif self.pri.n_obs > 1:
                self.train_idx, self.val_idx = train_test_split(
                    np.arange(self.pri.n_obs),
                    test_size=self.val_frac,
                    random_state=self.rnd,
                )
            else:
                raise ValueError("Datasets cannot be empty")
        elif self.val_frac == 0:
            self.train_idx = np.arange(self.pri.n_obs)
            self.val_idx = np.empty(0, dtype=self.train_idx.dtype)
        else:
            raise ValueError("Invalid validation fraction")  # pragma: no cover

    def __len__(self) -> int:
        return self.pri.n_obs

    @internal
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train is None:
                self.train = PairedDataset(
                    SimpleDataset(self.pri[self.train_idx]),
                    SimpleDataset(self.sec[self.train_idx]),
                )
            if self.val is None:
                self.val = PairedDataset(
                    SimpleDataset(self.pri[self.val_idx]),
                    SimpleDataset(self.sec[self.val_idx]),
                )
        if stage == "predict" and self.predict is None:
            self.predict = PairedDataset(
                SimpleDataset(self.pri), SimpleDataset(self.sec)
            )


class DynamicPairedDataModule(DataModule):
    r"""
    Dynamic paired data module using :class:`DynamicPairedDataset`

    Parameters
    ----------
    pri
        Primary interventional dataset
    sec
        Secondary interventional dataset
    stratify
        Column in :attr:`~anndata.AnnData.obs` used for stratification
    batch_size
        Batch size
    pin_memory
        Whether to use pin memory
    val_frac
        Fraction of validation data
    random_state
        Random state
    """

    def __init__(
        self,
        pri: AnnData,
        sec: AnnData,
        stratify: str | None,
        batch_size: int,
        pin_memory: bool,
        val_frac: float,
        random_state: RandomState,
    ) -> None:
        super().__init__(batch_size, pin_memory, val_frac, random_state)
        self.pri = pri
        self.sec = sec
        self.pri_strat = (
            pri.obs[stratify].to_numpy() if stratify else np.zeros(pri.n_obs)
        )
        self.sec_strat = (
            sec.obs[stratify].to_numpy() if stratify else np.zeros(sec.n_obs)
        )
        if 0 < self.val_frac < 1:
            if self.pri.n_obs == 1:
                self.pri_train_idx = self.pri_val_idx = np.arange(self.pri.n_obs)
            elif self.pri.n_obs > 1:
                self.pri_train_idx, self.pri_val_idx = train_test_split(
                    np.arange(self.pri.n_obs),
                    test_size=self.val_frac,
                    random_state=self.rnd,
                    stratify=self.pri_strat,
                )
            else:  # self.pri.n_obs == 0
                raise ValueError("Primary dataset cannot be empty")
            if self.sec.n_obs == 1:
                self.sec_train_idx = self.sec_val_idx = np.arange(self.sec.n_obs)
            elif self.sec.n_obs > 1:
                self.sec_train_idx, self.sec_val_idx = train_test_split(
                    np.arange(self.sec.n_obs),
                    test_size=self.val_frac,
                    random_state=self.rnd,
                    stratify=self.sec_strat,
                )
            else:  # self.sec.n_obs == 0
                raise ValueError("Secondary dataset cannot be empty")
        elif self.val_frac == 0:
            self.pri_train_idx = np.arange(self.pri.n_obs)
            self.sec_train_idx = np.arange(self.sec.n_obs)
            self.pri_val_idx = np.empty(0, dtype=self.pri_train_idx.dtype)
            self.sec_val_idx = np.empty(0, dtype=self.sec_train_idx.dtype)
        else:
            raise ValueError("Invalid validation fraction")  # pragma: no cover

    def __len__(self) -> int:
        return self.pri.n_obs

    @internal
    def setup(self, stage: str) -> None:
        if stage == "fit":
            if self.train is None:
                self.train = DynamicPairedDataset(
                    SimpleDataset(self.pri[self.pri_train_idx]),
                    SimpleDataset(self.sec[self.sec_train_idx]),
                    self.pri_strat[self.pri_train_idx],
                    self.sec_strat[self.sec_train_idx],
                    random_state=self.rnd,
                )
            if self.val is None:
                self.val = DynamicPairedDataset(
                    SimpleDataset(self.pri[self.pri_val_idx]),
                    SimpleDataset(self.sec[self.sec_val_idx]),
                    self.pri_strat[self.pri_val_idx],
                    self.sec_strat[self.sec_val_idx],
                    random_state=self.rnd,
                )
        if stage == "predict" and self.predict is None:
            self.predict = DynamicPairedDataset(
                SimpleDataset(self.pri),
                SimpleDataset(self.sec),
                self.pri_strat,
                self.sec_strat,
                random_state=self.rnd,
            )
