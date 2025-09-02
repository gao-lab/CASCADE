r"""
API entrypoint of the CASCADE model
"""

import os
import re
import shutil
import sys
from itertools import combinations
from logging import WARNING, getLogger
from pathlib import Path
from warnings import filterwarnings

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from anndata import AnnData
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from rich.panel import Panel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from . import __version__, name
from .core import (
    CausalNetwork,
    DiscoverScheduler,
    FitStage,
    LogAdj,
    ModelCheckpoint,
    PredictionWriter,
    PredictMode,
)
from .data import (
    DataModule,
    DynamicPairedDataModule,
    PairedDataModule,
    SimpleDataModule,
    _get_covariate,
    _get_regime,
    _get_size,
    _get_X,
    _set_covariate,
    _set_regime,
    configure_dataset,
    encode_regime,
)
from .nn import IntervDesign
from .typing import Kws, RandomState, SimpleGraph
from .utils import (
    autodevice,
    config,
    console,
    densify,
    get_random_state,
    gp_regression_with_ci,
    internal,
)

filterwarnings("ignore", ".*does not have many workers.*")
getLogger("pytorch_lightning.accelerators.cuda").setLevel(WARNING)
getLogger("pytorch_lightning.utilities.rank_zero").setLevel(WARNING)


class CASCADE:
    r"""
    **C**\ ausality-**A**\ ware **S**\ ingle-**C**\ ell **A**\ daptive **D**\
    iscover/**D**\ eduction/**D**\ esign Engine

    Parameters
    ----------
    vars
        List of variables to model
    n_particles
        Number of SVGD particles
    n_covariates
        Dimension of covariates
    n_layers
        Number of MLP layers in the structural equations
    hidden_dim
        MLP hidden layer dimension in the structural equations
    latent_dim
        Dimension of the latent variable (see notes below on how to specify
        ``latent_data`` depending on ``latent_mod``)
    dropout
        Dropout rate
    beta
        KL weight of the latent variable
    scaffold_mod
        Scaffold graph module, must be one of {"Edgewise", "Bilinear"}
    sparse_mod
        Sparse prior module, must be one of {"L1", "ScaleFree"}
    acyc_mod
        Acyclic prior module, must be one of {"TrExp", "SpecNorm", "LogDet"}
    latent_mod
        Latent module, must be one of {"NilLatent", "EmbLatent", "GCNLatent"}
    lik_mod
        Causal likelihood module, must be one of {"Normal", "NegBin"}
    kernel_mod
        SVGD kernel module, must be one of {"KroneckerDelta", "RBF"}
    scaffold_graph
        Optional scaffold graph
    latent_data
        Optional latent data (see notes below on how to specify ``latent_data``
        depending on ``latent_mod``)
    scaffold_kws
        Keyword arguments to the scaffold graph module, see
        :class:`~cascade.nn.Edgewise` or :class:`~cascade.nn.Bilinear` for
        details
    sparse_kws
        Keyword arguments to the sparse prior module, see
        :class:`~cascade.nn.L1` or :class:`~cascade.nn.ScaleFree` for details
    acyc_kws
        Keyword arguments to the acyclic prior module, see
        :class:`~cascade.nn.TrExp`, :class:`~cascade.nn.SpecNorm`, or
        :class:`~cascade.nn.LogDet` for details
    latent_kws
        Keyword arguments to the latent module, see
        :class:`~cascade.nn.NilLatent`, :class:`~cascade.nn.EmbLatent`, or
        :class:`~cascade.nn.GCNLatent` for details
    lik_kws
        Keyword arguments to the causal likelihood module, see
        :class:`~cascade.nn.Normal` or :class:`~cascade.nn.NegBin` for details
    kernel_kws
        Keyword arguments to the SVGD kernel module, see
        :class:`~cascade.nn.KroneckerDelta` or :class:`~cascade.nn.RBF` for
        details
    random_state
        Random state
    log_dir
        Directory to store tensorboard logs
    _net
        **Internal use ONLY**


    .. note::

        The setting for ``latent_dim`` and ``latent_data`` follows rules below:

        - When ``latent_mod="NilLatent"``, ``latent_data`` must be ``None``. The
          latent variable is always the standard normal distribution with
          dimension of ``latent_dim``.
        - When ``latent_mod="EmbLatent"``, ``latent_data`` must be a
          :class:`~pandas.DataFrame`, where the index is the variable names and
          the columns are the embedding dimensions. ``latent_dim`` but must be
          larger than 0, but does not need to equal the dimension of
          ``latent_data``, as the latent variable is encoded from the provided
          embedding with a linear transformation.
        - When ``latent_mod="GCNLatent"``, ``latent_data`` must be a
          :class:`~networkx.Graph` or :class:`~networkx.DiGraph`, where the
          nodes are the variable names and the edges are latent connections.
          ``latent_dim`` must be larger than 0. The latent variable is encoded
          from the provided graph with a graph convolutional network.
    """

    def __init__(
        self,
        vars: pd.Index | list[str],
        n_particles: int = 4,
        n_covariates: int = 0,
        n_layers: int = 1,
        hidden_dim: int = 16,
        latent_dim: int = 16,
        dropout: float = 0.2,
        beta: float = 0.1,
        scaffold_mod: str = "Edgewise",
        sparse_mod: str = "L1",
        acyc_mod: str = "SpecNorm",
        latent_mod: str = "EmbLatent",
        lik_mod: str = "NegBin",
        kernel_mod: str = "RBF",
        scaffold_graph: SimpleGraph | None = None,
        latent_data: pd.DataFrame | SimpleGraph | None = None,
        scaffold_kws: Kws = None,
        sparse_kws: Kws = None,
        acyc_kws: Kws = None,
        latent_kws: Kws = None,
        lik_kws: Kws = None,
        kernel_kws: Kws = None,
        random_state: RandomState = 0,
        log_dir: os.PathLike = ".",
        _net: CausalNetwork | None = None,
    ) -> None:
        self.vars = pd.Index(vars)
        self.rnd = get_random_state(random_state)
        self.log_dir = Path(log_dir)
        self.interv_seen = set()

        if _net is not None:
            self.net = _net
            return

        scaffold_kws = scaffold_kws or {}
        if scaffold_graph is None:
            scaffold_graph = nx.complete_graph(self.vars, create_using=nx.DiGraph)
        else:
            scaffold_graph = scaffold_graph.subgraph(self.vars)
            if not nx.is_directed(scaffold_graph):
                scaffold_graph = scaffold_graph.to_directed()
        edgelist = nx.to_pandas_edgelist(scaffold_graph)
        scaffold_kws["eidx"] = torch.as_tensor(
            np.stack(
                [
                    self.vars.get_indexer(edgelist["source"]),
                    self.vars.get_indexer(edgelist["target"]),
                ]
            )
        )

        latent_kws = latent_kws or {}
        latent_vars = pd.Index([])
        if latent_dim:
            if latent_mod == "EmbLatent":
                if not isinstance(latent_data, pd.DataFrame):
                    raise ValueError(
                        f"Latent embedding must be provided for {latent_mod}"
                    )
                latent_data = latent_data.reindex(self.vars).dropna()
                latent_vars = latent_data.index
                latent_kws["emb"] = torch.as_tensor(latent_data.to_numpy())
            elif latent_mod == "GCNLatent":
                if not isinstance(latent_data, nx.Graph):
                    raise ValueError(f"Latent graph must be provided for {latent_mod}")
                latent_data = latent_data.subgraph(
                    v for v, deg in latent_data.degree() if deg > 0
                )
                if not nx.is_directed(latent_data):
                    latent_data = latent_data.to_directed()
                latent_vars = pd.Index(latent_data.nodes)
                edgelist = nx.to_pandas_edgelist(latent_data)
                latent_kws["eidx"] = torch.as_tensor(
                    np.stack(
                        [
                            latent_vars.get_indexer(edgelist["source"]),
                            latent_vars.get_indexer(edgelist["target"]),
                        ]
                    )
                )
                latent_kws["ewt"] = (
                    torch.as_tensor(edgelist["weight"], dtype=torch.get_default_dtype())
                    if latent_kws["eidx"].size(1)
                    else torch.zeros(0)
                )
        elif latent_mod != "NilLatent":
            raise ValueError(f"Latent dimension must be non-zero for {latent_mod}")
        if latent_mod == "NilLatent" and latent_data is not None:
            raise ValueError("Latent data not accepted for NilLatent")

        common_vars = self.vars.intersection(latent_vars)
        latent_kws["vmap"] = torch.as_tensor(
            np.stack(
                [
                    self.vars.get_indexer(common_vars),
                    latent_vars.get_indexer(common_vars),
                ]
            )
        )

        self.manual_seed()
        self.net = CausalNetwork(
            n_vars=self.vars.size,
            n_particles=n_particles,
            n_covariates=n_covariates,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
            beta=beta,
            scaffold_mod=scaffold_mod,
            sparse_mod=sparse_mod,
            acyc_mod=acyc_mod,
            latent_mod=latent_mod,
            lik_mod=lik_mod,
            kernel_mod=kernel_mod,
            scaffold_kws=scaffold_kws,
            sparse_kws=sparse_kws,
            acyc_kws=acyc_kws,
            latent_kws=latent_kws,
            lik_kws=lik_kws,
            kernel_kws=kernel_kws,
        )

    @internal
    def manual_seed(self) -> None:
        torch.manual_seed(self.rnd.randint(0, 2**64 - 1, dtype=np.uint64))

    @internal
    def align_vars(self, input: AnnData) -> AnnData:
        input_vars = input.var_names
        excess_vars = set(input_vars) - set(self.vars)
        if excess_vars:
            logger.warning(
                f"{len(excess_vars)} variables are not in the "
                f"`scaffold` and will thus be ignored."
            )
        return input[:, self.vars]

    def export_causal_graph(self, edge_attr: str = "weight") -> nx.DiGraph:
        r"""
        Export learned causal graph

        Parameters
        ----------
        edge_attr
            Edge attribute name to store edge probabilities

        Returns
        -------
        Learned causal graph
        """
        digraph = self.net.scaffold.export_graph(edge_attr=edge_attr)
        return nx.relabel_nodes(digraph, dict(enumerate(self.vars)), copy=False)

    def import_causal_graph(
        self, digraph: nx.DiGraph, edge_attr: str = "weight"
    ) -> None:
        r"""
        Import pruned causal graph

        Parameters
        ----------
        digraph
            Pruned causal graph
        edge_attr
            Edge attribute name to read edge probabilities
        """
        digraph = nx.relabel_nodes(
            digraph, {v: i for i, v in enumerate(self.vars)}, copy=True
        )
        self.net.scaffold.import_graph(digraph, edge_attr=edge_attr)

    def export_causal_map(self) -> pd.DataFrame:
        r"""
        Export the reshaped causal map indicating which input gene is in each
        reshaped position for each output gene, useful for interpreting the
        result of :meth:`~CASCADE.explain`.

        Returns
        -------
        Causal map of shape (n_vars, max_indegree)


        .. note::

            Padding positions are labeled as "<pad>"
        """
        return pd.DataFrame(
            np.append(self.vars.to_numpy(), "<pad>")[
                self.net.scaffold.mask_map.numpy(force=True)
            ],
            index=self.vars,
        )

    @internal
    @rank_zero_only
    def report_banner(self, datamodule: DataModule) -> None:
        console.print(
            Panel(
                f"Training on [hl]{self.vars.size}[/hl] variables "
                f"with [hl]{self.net.scaffold.n_edges}[/hl] scaffold edges "
                f"and [hl]{len(datamodule)}[/hl] samples",
                expand=False,
                padding=(1, 2),
                title=name,
                subtitle=f"v{__version__}",
            )
        )

    def _fit(
        self,
        datamodule: DataModule,
        fit_stage: FitStage,
        accelerator: str,
        devices: list[int] | str,
        log_subdir: os.PathLike,
        opt: str,
        lr: float,
        weight_decay: float,
        accumulate_grad_batches: int,
        log_adj: LogAdj,
        **kwargs,
    ) -> None:
        self.report_banner(datamodule)
        tensorboard_logger = TensorBoardLogger(
            save_dir=self.log_dir / log_subdir, default_hp_metric=False
        )
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            precision=config.PRECISION,
            logger=tensorboard_logger,
            log_every_n_steps=config.LOG_STEP_INTERVAL,
            deterministic=config.DETERMINISTIC,
            default_root_dir=self.log_dir / log_subdir,
            **kwargs,
        )
        self.net.opt = opt
        self.net.lr = lr
        self.net.weight_decay = weight_decay
        self.net.accumulate_grad_batches = accumulate_grad_batches
        self.net.fit_stage = fit_stage
        self.net.log_adj = log_adj
        trainer.fit(self.net, datamodule=datamodule)
        if isinstance(devices, list) and len(devices) > 1:
            trainer.strategy.barrier()
            dist.destroy_process_group()
            if trainer.global_rank > 0:
                logger.debug("Exiting rank-non-zero process.")
                sys.exit()
            logger.debug("Continuing rank-zero process.")

    def _predict(
        self,
        datamodule: DataModule,
        predict_mode: PredictMode,
        accelerator: str,
        devices: list[int] | str,
        **kwargs,
    ) -> list[torch.Tensor]:
        if isinstance(devices, list) and len(devices) > 1:
            pred_dir = self.log_dir / f"pred-{config.RUN_ID}"
            callbacks = [PredictionWriter(output_dir=pred_dir)]
        else:
            callbacks = None
        trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            deterministic=config.DETERMINISTIC,
            precision=config.PRECISION,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            **kwargs,
        )
        self.net.predict_mode = predict_mode
        pred = trainer.predict(self.net, datamodule=datamodule)
        if isinstance(devices, list) and len(devices) > 1:
            trainer.strategy.barrier()
            dist.destroy_process_group()
            if trainer.global_rank > 0:
                logger.debug("Exiting rank-non-zero process.")
                sys.exit()
            logger.debug("Continuing rank-zero process.")
            pred = {
                int(re.search(r"pred(\d+)\.pt", item.name).group(1)): torch.load(
                    item, weights_only=True
                )
                for item in pred_dir.glob("pred*.pt")
            }
            ind = {
                int(re.search(r"ind(\d+)\.pt", item.name).group(1)): torch.load(
                    item, weights_only=True
                )[0]
                for item in pred_dir.glob("ind*.pt")
            }
            pred = [item for k in sorted(ind) for item in pred[k]]
            ind = [torch.as_tensor(item) for k in sorted(ind) for item in ind[k]]
            pred = tuple(torch.cat(items, dim=1) for items in zip(*pred))
            argsort = torch.cat(ind).argsort(stable=True)
            pred = tuple(item[:, argsort] for item in pred)
            shutil.rmtree(pred_dir)
        else:
            pred = tuple(torch.cat(items, dim=1) for items in zip(*pred))
        return pred

    @rank_zero_only
    def _load_checkpoint(self, path: str) -> None:
        if os.path.exists(path):
            console.print(f"Restoring best model: {path}.")
            self.net = type(self.net).load_from_checkpoint(
                path, map_location="cpu", design=self.net.design
            )
        else:
            logger.warning(
                "No best checkpoint found! Exiting as is."
            )  # pragma: no cover

    @rank_zero_only
    def _update_interv_seen(self, adata: AnnData) -> None:
        regime = _get_regime(adata)
        regime_count = np.asarray(regime.sum(axis=0)).ravel()
        for var, count in zip(self.vars, regime_count):
            if count:
                self.interv_seen.add(var)

    @rank_zero_only
    def _extrapolate_interv(self) -> None:
        unseen = set(self.vars) - self.interv_seen
        if not unseen:
            logger.info("Skipping extrapolation because all variables intervened.")
            return
        if not self.interv_seen:
            logger.warning("Skipping extrapolation because no variable intervened.")
            return
        logger.info(
            f"Extrapolating scale and bias of {len(unseen)} non-intervened variables "
            f"from {len(self.interv_seen)} intervened variables."
        )
        seen_mask = self.vars.isin(self.interv_seen)
        unseen_mask = self.vars.isin(unseen)
        for param in (self.net.interv_scale, self.net.interv_bias):
            extrapolate = param.data[:, seen_mask].quantile(0.5, dim=-1, keepdim=True)
            param.data[:, unseen_mask] = extrapolate.expand(-1, len(unseen))

    def discover(
        self,
        adata: AnnData,
        lam: float = 0.1,
        alpha: float = 0.5,
        gamma: float = 1.0,
        cyc_tol: float = 1e-4,
        prefit: bool = False,
        opt: str = "AdamW",
        lr: float = 5e-3,
        weight_decay: float = 0.01,
        accumulate_grad_batches: int = 1,
        log_adj: LogAdj = LogAdj.mean,
        batch_size: int = 128,
        val_check_interval: int = 300,
        val_frac: float = 0.1,
        max_epochs: int = 1000,
        n_devices: int = 1,
        log_subdir: os.PathLike = "discover",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        r"""
        Causal discovery

        Parameters
        ----------
        adata
            Input dataset
        lam
            Sparse penalty rate (:math:`\eta_\lambda` in paper)
        alpha
            Acyclic penalty rate (:math:`\eta_\alpha` in paper)
        gamma
            Kernel gradient rate (:math:`\eta_\gamma`)
        cyc_tol
            Acyclic violation tolerance
        prefit
            Whether to prefit the model on covariates
        opt
            Optimizer
        lr
            Learning rate
        weight_decay
            Weight decay
        accumulate_grad_batches
            Number of batches to accumulate before optimizer step
        log_adj
            Adjacency matrix logging mode (see :class:`~cascade.core.LogAdj`)
        batch_size
            Batch size
        val_check_interval
            Validation check interval
        val_frac
            Validation fraction
        max_epochs
            Maximum number of epochs
        n_devices
            Number of GPU devices to use
        log_subdir
            Tensorboard log subdirectory (under model-wise ``log_dir``)
        verbose
            Whether to print verbose logs
        **kwargs
            Additional keyword arguments are passed to
            :class:`~lightning.pytorch.trainer.trainer.Trainer`
        """
        adata = self.align_vars(adata)
        self.net.reset_parameters()
        self.net.prefit = prefit

        accelerator, granted = autodevice(n_devices)
        self.manual_seed()
        datamodule = SimpleDataModule(
            adata=adata,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=val_frac,
            random_state=self.rnd,
        )

        progress_bar = TQDMProgressBar(refresh_rate=config.PBAR_REFRESH)
        model_checkpoint = ModelCheckpoint(
            monitor="val/post_enrg",
            mode="min",
            save_top_k=config.CKPT_SAVE_K,
            verbose=verbose,
        )
        discover_scheduler = DiscoverScheduler(
            monitor="val/post_enrg",
            constraint="val/acyc_enrg",
            tolerance=cyc_tol,
            lam=lam,
            alpha=alpha,
            gamma=gamma,
            mode="min",
            min_delta=config.MIN_DELTA,
            patience=config.PATIENCE,
            verbose=verbose,
        )
        self.net.lik.set_empirical(adata)
        self._fit(
            datamodule=datamodule,
            fit_stage=FitStage.discover,
            accelerator=accelerator,
            devices=granted,
            log_subdir=log_subdir,
            opt=opt,
            lr=lr,
            weight_decay=weight_decay,
            accumulate_grad_batches=accumulate_grad_batches,
            log_adj=log_adj,
            check_val_every_n_epoch=None,
            val_check_interval=val_check_interval,
            max_epochs=max_epochs,
            callbacks=[progress_bar, discover_scheduler, model_checkpoint],
            **kwargs,
        )
        self._load_checkpoint(model_checkpoint.best_model_path)
        self._update_interv_seen(adata)
        self._extrapolate_interv()

    def tune(
        self,
        adata: AnnData,
        tune_ctfact: bool = False,
        stratify: str | None = None,
        opt: str = "AdamW",
        lr: float = 5e-3,
        weight_decay: float = 0.01,
        accumulate_grad_batches: int = 1,
        log_adj: LogAdj = LogAdj.mean,
        batch_size: int = 128,
        val_check_interval: int = 300,
        val_frac: float = 0.1,
        max_epochs: int = 1000,
        n_devices: int = 1,
        log_subdir: os.PathLike = "tune",
        verbose: bool = False,
        **kwargs,
    ) -> CausalNetwork:
        r"""
        Fine-tune structural equations with fixed causal structure

        Parameters
        ----------
        adata
            Input dataset
        tune_ctfact
            Whether to tune in counterfactual mode, i.e., to use randomly
            paired samples for counterfactual pairs for tuning.
        stratify
            Column name in :attr:`~anndata.AnnData.obs` for stratified random
            pairing (only relevant when using ``tune_ctfact=True``)
        opt
            Optimizer
        lr
            Learning rate
        weight_decay
            Weight decay
        accumulate_grad_batches
            Number of batches to accumulate before optimizer step
        log_adj
            Adjacency matrix logging mode (see :class:`~cascade.core.LogAdj`)
        batch_size
            Batch size
        val_check_interval
            Validation check interval
        val_frac
            Validation fraction
        max_epochs
            Maximum number of epochs
        n_devices
            Number of GPU devices to use
        log_subdir
            Tensorboard log subdirectory (under model-wise ``log_dir``)
        verbose
            Whether to print verbose logs
        **kwargs
            Additional keyword arguments are passed to
            :class:`~lightning.pytorch.trainer.trainer.Trainer`
        """
        adata = self.align_vars(adata)
        if not self.net.scaffold.frozen:
            raise RuntimeError(
                "Scaffold is not frozen! "
                "Did you forget to import an acyclified graph?"
            )
        logger.info("Pruning model...")
        self.net.prune()

        accelerator, granted = autodevice(n_devices)
        self.manual_seed()
        if tune_ctfact:
            datamodule = DynamicPairedDataModule(
                pri=adata,
                sec=adata,
                stratify=stratify,
                batch_size=batch_size,
                pin_memory=accelerator == "gpu",
                val_frac=val_frac,
                random_state=self.rnd,
            )
        else:
            datamodule = SimpleDataModule(
                adata=adata,
                batch_size=batch_size,
                pin_memory=accelerator == "gpu",
                val_frac=val_frac,
                random_state=self.rnd,
            )

        progress_bar = TQDMProgressBar(refresh_rate=config.PBAR_REFRESH)
        model_checkpoint = ModelCheckpoint(
            monitor="val/lik_enrg",
            mode="min",
            save_top_k=config.CKPT_SAVE_K,
            verbose=verbose,
        )
        earlystopping = EarlyStopping(
            monitor="val/lik_enrg",
            mode="min",
            min_delta=config.MIN_DELTA,
            patience=config.PATIENCE,
            verbose=verbose,
        )
        self._fit(
            datamodule=datamodule,
            fit_stage=FitStage.ctfact if tune_ctfact else FitStage.tune,
            accelerator=accelerator,
            devices=granted,
            log_subdir=log_subdir,
            opt=opt,
            lr=lr,
            weight_decay=weight_decay,
            accumulate_grad_batches=accumulate_grad_batches,
            log_adj=log_adj,
            check_val_every_n_epoch=None,
            val_check_interval=val_check_interval,
            max_epochs=max_epochs,
            callbacks=[progress_bar, earlystopping, model_checkpoint],
            **kwargs,
        )
        self._load_checkpoint(model_checkpoint.best_model_path)
        self._update_interv_seen(adata)
        self._extrapolate_interv()

    def design(
        self,
        source: AnnData,
        target: AnnData,
        pool: list[str] | None = None,
        init: list[str] | None = None,
        design_size: int = 1,
        design_scale_bias: bool = False,
        target_weight: str | None = None,
        stratify: str | None = None,
        opt: str = "AdamW",
        lr: float = 5e-2,
        weight_decay: float = 0.01,
        accumulate_grad_batches: int = 1,
        batch_size: int = 32,
        val_check_interval: int = 300,
        val_frac: float = 0.1,
        max_epochs: int = 1000,
        n_devices: int = 1,
        log_subdir: os.PathLike = "design",
        verbose: bool = False,
        **kwargs,
    ) -> tuple[pd.DataFrame, IntervDesign]:
        r"""
        Targeted intervention design with continuous optimization

        Parameters
        ----------
        source
            Source dataset
        target
            Target dataset representing desired outcome
        pool
            Optional list of variables as candidate pool
        init
            Optional list of variables to initialize the designed interventions
        design_size
            Maximal combinatorial order to consider
        design_scale_bias
            Whether to optimize the intervention scale and bias
        target_weight
            Optional column name in ``target.var`` to weight target variables
            when computing target deviation
        stratify
            Column name in :attr:`~anndata.AnnData.obs` for stratified random
            pairing
        opt
            Optimizer
        lr
            Learning rate
        weight_decay
            Weight decay
        accumulate_grad_batches
            Number of batches to accumulate before optimizer step
        batch_size
            Batch size
        val_check_interval
            Validation check interval
        val_frac
            Validation fraction
        max_epochs
            Maximum number of epochs
        n_devices
            Number of GPU devices to use
        log_subdir
            Tensorboard log subdirectory (under model-wise ``log_dir``)
        verbose
            Whether to print verbose logs
        **kwargs
            Additional keyword arguments are passed to
            :class:`~lightning.pytorch.trainer.trainer.Trainer`

        Returns
        -------
        DataFrame of design scores containing the following column:

            - "score": Design score

            Indexed by intervention and sorted by descending scores

        Intervention design module
        """
        source = self.align_vars(source)
        target = self.align_vars(target)
        mask = torch.as_tensor(self.vars.isin(pool or self.vars))
        target_weight = (
            torch.as_tensor(target.var[target_weight].to_numpy())
            if target_weight
            else torch.ones(target.n_vars)
        )
        self.net.set_design(
            mask=mask,
            k=design_size,
            design_scale_bias=design_scale_bias,
            target_weight=target_weight,
        )
        init = torch.as_tensor(self.vars.get_indexer(init or []))
        if (init < 0).any():
            raise ValueError("Invalid init variables")
        self.net.design.logits.data[
            torch.isin(self.net.design.comb, init).any(dim=1)
        ] = 10.0
        self.net.vars = self.vars

        accelerator, granted = autodevice(n_devices)
        self.manual_seed()
        datamodule = DynamicPairedDataModule(
            pri=source,
            sec=target,
            stratify=stratify,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=val_frac,
            random_state=self.rnd,
        )

        progress_bar = TQDMProgressBar(refresh_rate=config.PBAR_REFRESH)
        model_checkpoint = ModelCheckpoint(
            monitor="val/nll",
            mode="min",
            save_top_k=config.CKPT_SAVE_K,
            verbose=verbose,
        )
        early_stopping = EarlyStopping(
            monitor="val/nll",
            mode="min",
            min_delta=0.01,
            patience=config.PATIENCE,
            verbose=verbose,
        )
        self._fit(
            datamodule=datamodule,
            fit_stage=FitStage.design,
            accelerator=accelerator,
            devices=granted,
            log_subdir=log_subdir,
            opt=opt,
            lr=lr,
            weight_decay=weight_decay,
            accumulate_grad_batches=accumulate_grad_batches,
            log_adj=LogAdj.none,
            check_val_every_n_epoch=None,
            val_check_interval=val_check_interval,
            max_epochs=max_epochs,
            callbacks=[progress_bar, early_stopping, model_checkpoint],
            **kwargs,
        )
        self._load_checkpoint(model_checkpoint.best_model_path)

        scores = pd.DataFrame(
            {"score": self.net.design.logits.numpy(force=True)},
            index=[",".join(sorted(self.vars[c])) for c in self.net.design.comb_lists],
        ).sort_values("score", ascending=False, kind="stable")
        design, self.net.design = self.net.design.cpu(), None
        return scores, design

    def design_error_curve(
        self,
        source: AnnData,
        target: AnnData,
        design: IntervDesign,
        n_steps: int = 500,
        n_cells: int = 100,
        confidence_level: float = 0.95,
        stratify: str | None = None,
        batch_size: int = 128,
        n_devices: int = 1,
    ) -> tuple[pd.DataFrame, float]:
        r"""
        Fit an error curve against design scores

        Parameters
        ----------
        source
            Source dataset
        target
            Target dataset representing desired outcome
        design
            Intervention design module from :meth:`~CASCADE.design`
        n_steps
            Number of equidistant score steps
        n_cells
            Number of cells per design
        confidence_level
            Confidence level
        stratify
            Column name in :attr:`~anndata.AnnData.obs` for stratified random
            pairing
        batch_size
            Batch size
        n_devices
            Number of GPU devices to use

        Returns
        -------
        DataFrame of design error curve containing the following columns:

            - "score": Design score
            - "mse_est": Weighted MSE estimate at equidistant steps
            - "mse_est_mean": Smoothed weighted MSE estimate
            - "mse_est_lower": Lower bound of the confidence interval
            - "mse_est_upper": Upper bound of the confidence interval

            Indexed by intervention and sorted by descending scores

        Design score cutoff that covers minimal MSE in the confidence interval
        """
        source = self.align_vars(source)
        target = self.align_vars(target)
        logits, comb_lists = design.logits.detach(), design.comb_lists
        argsort = torch.argsort(logits, stable=True)  # Ascending
        min_logit, max_logit = logits.min(), logits.max()
        step_size = (max_logit - min_logit) / (n_steps - 1)
        step_logits = min_logit + step_size * torch.arange(n_steps)
        step_locs = torch.searchsorted(logits, step_logits, sorter=argsort)
        step_locs = step_locs.clamp(min=0, max=logits.size(0) - 1).unique()
        step_locs = argsort[step_locs]
        n_steps = step_locs.size(0)
        step_regime = csr_matrix(
            design.simplex2regime(
                F.one_hot(step_locs, num_classes=logits.size(0))
            ).numpy(force=True)
        )  # (n_steps, n_vars)

        source = source[
            self.rnd.choice(source.n_obs, n_steps * n_cells, replace=True)
        ].copy()
        source.obs_names_make_unique()
        repeat_idx = np.arange(n_steps).repeat(n_cells)
        _set_regime(source, step_regime[repeat_idx])

        accelerator, granted = autodevice(n_devices)
        self.manual_seed()
        datamodule = DynamicPairedDataModule(
            pri=source,
            sec=target,
            stratify=stratify,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=0.0,
            random_state=self.rnd,
        )
        self.net.design = design
        pred = self._predict(
            datamodule=datamodule,
            predict_mode=PredictMode.dsgnerr,
            accelerator=accelerator,
            devices=granted,
        )
        self.net.design = None

        error = (
            pd.DataFrame(
                {
                    "mse_est": pred[0].mean(dim=0).numpy(force=True),
                    "regime": [
                        ",".join(sorted(self.vars[comb_lists[i]]))
                        for i in step_locs[repeat_idx]
                    ],
                },
            )
            .groupby("regime")
            .mean()
            .reset_index()
        )
        score = pd.DataFrame(
            {
                "score": design.logits.numpy(force=True),
                "regime": [",".join(sorted(self.vars[c])) for c in design.comb_lists],
            },
        )
        curve = (
            pd.merge(score, error, how="outer")
            .set_index("regime")
            .sort_values("score", ascending=False, kind="stable")
        )
        curve, cutoff = gp_regression_with_ci(
            curve, x="score", y="mse_est", alpha=confidence_level
        )
        curve["qualified"] = curve["score"] > cutoff
        return curve, cutoff

    def design_brute_force(
        self,
        source: AnnData,
        target: AnnData,
        pool: list[str] | None = None,
        design_size: int = 1,
        k: int = 30,
        counterfactual_kws: Kws = None,
        neighbor_kws: Kws = None,
    ) -> tuple[pd.DataFrame, AnnData]:
        r"""
        Intervention design with brute-force exhaustion

        Parameters
        ----------
        source
            Source dataset
        target
            Target dataset representing desired outcome
        pool
            Optional list of variables as candidate pool
        design_size
            Maximal combinatorial order to consider
        k
            Number of samples to generate for each design
        counterfactual_kws
            Additional keyword arguments passed to
            :meth:`~CASCADE.counterfactual`
        neighbor_kws
            Additional keyword arguments passed to
            :class:`~sklearn.neighbors.NearestNeighbors`

        Returns
        -------
        DataFrame of intervention designs, sorted by descending vote counts
        AnnData object with counterfactual predictions for all designs
        """
        source = self.align_vars(source)
        target = self.align_vars(target)

        pool = pool or self.vars
        search_space = [
            ",".join(sorted(c))
            for s in range(design_size + 1)
            for c in combinations(pool, s)
        ]
        source_idx = self.rnd.choice(source.n_obs, len(search_space) * k)
        target_idx = self.rnd.choice(target.n_obs, len(search_space) * k)
        source = source[source_idx].copy()
        source.obs["design"] = np.repeat(search_space, k)
        encode_regime(source, "design", key="design")
        configure_dataset(source, use_regime="design")  # Others kept untouched
        try:
            _set_covariate(source, _get_covariate(target)[target_idx])
            logger.info("Using target covariates")
        except ValueError:
            logger.info("No covariates set")
        ctfact = self.counterfactual(source, **(counterfactual_kws or {}))
        self.net.eval()  # Otherwise it restores to training mode
        dtype = torch.get_default_dtype()
        device = self.net.interv_scale.device
        ref = self.net.lik.tone(
            torch.as_tensor(ctfact.X, dtype=dtype, device=device),
            torch.as_tensor(_get_size(ctfact), dtype=dtype, device=device),
        ).numpy(force=True)
        query = self.net.lik.tone(
            torch.as_tensor(densify(_get_X(target)), dtype=dtype, device=device),
            torch.as_tensor(_get_size(target), dtype=dtype, device=device),
        ).numpy(force=True)
        neighbor = NearestNeighbors(**(neighbor_kws or {})).fit(ref)
        nni = neighbor.kneighbors(query, return_distance=False)
        votes = ctfact.obs["design"].iloc[nni.ravel()].value_counts()
        outcast = [item for item in search_space if item not in votes.index]
        outcast = pd.Series(0, index=outcast, name="count")
        design = pd.concat([votes, outcast])
        design = design.to_frame().rename(columns={"count": "votes"})
        return design, ctfact

    def counterfactual(
        self,
        adata: AnnData,
        batch_size: int = 128,
        n_devices: int = 1,
        design: IntervDesign | None = None,
        fixed_genes: list[str] | None = None,
        sample: bool = False,
        ablate_latent: bool = False,
        ablate_interv: bool = False,
        ablate_graph: bool = False,
    ) -> AnnData:
        r"""
        Counterfactual deduction for the outcome of alternative interventions
        for an observed dataset

        Parameters
        ----------
        adata
            Input dataset
        batch_size
            Batch size
        n_devices
            Number of GPU devices to use
        design
            Optional intervention design module from :meth:`~CASCADE.design`
        fixed_genes
            Optional list of genes to keep their values fixed
        sample
            Whether to sample from the counterfactual distribution (True) or use
            the mean (False)
        ablate_latent
            If True, removes the effect of latent variables
        ablate_interv
            If True, removes the effect of interventions
        ablate_graph
            If True, removes the effect of the causal graph

        Returns
        -------
        Counterfactual dataset with:

            - :attr:`~anndata.AnnData.layers`\ ``["X_ctfact"]``:
              Counterfactual predictions with shape (n_obs, n_vars, n_particles)
            - :attr:`~anndata.AnnData.X`:
              Mean values across SVGD particles
        """
        adata = self.align_vars(adata).copy()
        accelerator, granted = autodevice(n_devices)

        if fixed_genes is not None:
            fixed_genes = self.vars.get_indexer(fixed_genes)
            self.net.fixed_vars = torch.as_tensor(fixed_genes)

        self.manual_seed()
        datamodule = SimpleDataModule(
            adata=adata,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=0.0,
            random_state=self.rnd,
        )
        self.net.design = design
        self.net.ablate_latent = ablate_latent
        self.net.ablate_interv = ablate_interv
        self.net.ablate_graph = ablate_graph
        pred = self._predict(
            datamodule=datamodule,
            predict_mode=PredictMode.ctsamp if sample else PredictMode.ctmean,
            accelerator=accelerator,
            devices=granted,
        )
        self.net.design = None
        adata.layers["X_ctfact"] = pred[2].movedim(0, -1).numpy(force=True)
        if fixed_genes is not None:
            fixed_X = densify(_get_X(adata)[:, fixed_genes])
            adata.layers["X_ctfact"][:, fixed_genes] = np.atleast_3d(fixed_X)
        adata.X = adata.layers["X_ctfact"].mean(axis=-1)
        return adata

    def explain(
        self,
        adata: AnnData,
        ctfact: AnnData,
        batch_size: int = 128,
        n_devices: int = 1,
        design: IntervDesign | None = None,
    ) -> AnnData:
        r"""
        Explain counterfactual outcome with individual components

        Parameters
        ----------
        adata
            Factual dataset
        ctfact
            Counterfactual prediction from :meth:`~CASCADE.counterfactual`
        batch_size
            Batch size
        n_devices
            Number of GPU devices to use
        design
            Optional intervention design module from :meth:`~CASCADE.design`

        Returns
        -------
        Dataset with the following explanation components:

            - :attr:`~anndata.AnnData.layers`\ ``["X_nil"]``:
              Baseline expression without any effect
            - :attr:`~anndata.AnnData.layers`\ ``["X_ctrb_i"]``:
              Contribution from intervention
            - :attr:`~anndata.AnnData.layers`\ ``["X_ctrb_s"]``:
              Contribution from covariates
            - :attr:`~anndata.AnnData.layers`\ ``["X_ctrb_z"]``:
              Contribution from latent
            - :attr:`~anndata.AnnData.layers`\ ``["X_ctrb_ptr"]``:
              Contribution from parents
            - :attr:`~anndata.AnnData.layers`\ ``["X_tot"]``:
              Total counterfactual prediction

            All having shape (n_obs, n_vars, n_particles)
        """
        adata = self.align_vars(adata)
        ctfact = self.align_vars(ctfact).copy()
        accelerator, granted = autodevice(n_devices)
        self.manual_seed()
        datamodule = PairedDataModule(
            pri=adata,
            sec=ctfact,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=0.0,
            random_state=self.rnd,
        )
        self.net.design = design
        nil, ctrb_i, ctrb_s, ctrb_z, ctrb_ptr, tot = self._predict(
            datamodule=datamodule,
            predict_mode=PredictMode.explain,
            accelerator=accelerator,
            devices=granted,
        )
        self.net.design = None
        ctfact.layers["X_nil"] = nil.movedim(0, -1).numpy(force=True)
        ctfact.layers["X_ctrb_i"] = ctrb_i.movedim(0, -1).numpy(force=True)
        ctfact.layers["X_ctrb_s"] = ctrb_s.movedim(0, -1).numpy(force=True)
        ctfact.layers["X_ctrb_z"] = ctrb_z.movedim(0, -1).numpy(force=True)
        ctfact.layers["X_ctrb_ptr"] = ctrb_ptr.movedim(0, -1).numpy(force=True)
        ctfact.layers["X_tot"] = tot.movedim(0, -1).numpy(force=True)
        return ctfact

    def diagnose(
        self, adata: AnnData, batch_size: int = 128, n_devices: int = 1
    ) -> AnnData:
        r"""
        Model diagnosis

        Parameters
        ----------
        adata
            Input dataset
        batch_size
            Batch size
        n_devices
            Number of GPU devices to use

        Returns
        -------
        Dataset with the following diagnostic information:

            - :attr:`~anndata.AnnData.obsm`\ ``["Z_mean_diag"]``:
              Latent mean
            - :attr:`~anndata.AnnData.obsm`\ ``["Z_std_diag"]``:
              Latent standard deviation
            - :attr:`~anndata.AnnData.layers`\ ``["X_mean_diag"]``:
              Reconstructed mean
            - :attr:`~anndata.AnnData.layers`\ ``["X_std_diag"]``:
              Reconstructed standard deviation
            - :attr:`~anndata.AnnData.layers`\ ``["X_disp_diag"]``:
              Dispersion parameter
        """
        adata = self.align_vars(adata).copy()
        accelerator, granted = autodevice(n_devices)

        self.manual_seed()
        datamodule = SimpleDataModule(
            adata=adata,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=0.0,
            random_state=self.rnd,
        )
        pred = self._predict(
            datamodule=datamodule,
            predict_mode=PredictMode.recon,
            accelerator=accelerator,
            devices=granted,
        )
        (
            adata.obsm["Z_mean_diag"],
            adata.obsm["Z_std_diag"],
            adata.layers["X_mean_diag"],
            adata.layers["X_std_diag"],
            adata.layers["X_disp_diag"],
        ) = (item.movedim(0, -1).numpy(force=True) for item in pred)
        return adata

    def jacobian(
        self, adata: AnnData, batch_size: int = 128, n_devices: int = 1
    ) -> AnnData:
        r"""
        Compute the Jacobian matrix of the model

        Parameters
        ----------
        adata
            Input dataset
        batch_size
            Batch size
        n_devices
            Number of GPU devices to use

        Returns
        -------
        Dataset with

            - :attr:`~anndata.AnnData.layers`\ ``["X_jac"]``:
              The Jacobian matrix with shape
              (n_obs, n_vars, n_parents, n_particles)
        """
        adata = self.align_vars(adata).copy()
        accelerator, granted = autodevice(n_devices)

        self.manual_seed()
        datamodule = SimpleDataModule(
            adata=adata,
            batch_size=batch_size,
            pin_memory=accelerator == "gpu",
            val_frac=0.0,
            random_state=self.rnd,
        )
        pred = self._predict(
            datamodule=datamodule,
            predict_mode=PredictMode.jac,
            accelerator=accelerator,
            devices=granted,
            inference_mode=False,
            # This is necessary because we need to make lightning use
            # torch.no_grad() rather than torch.inference_mode(), so that we can
            # re-enable gradient within predict_step() with torch.enable_grad().
        )
        adata.layers["X_jac"] = (
            pred[0].movedim(0, -1).numpy(force=True)
        )  # (bs, n_vars_out, n_vars_in, n_particles)
        return adata

    @rank_zero_only
    def save(self, fname: os.PathLike) -> None:
        r"""
        Save model to file

        Parameters
        ----------
        Path to save the model file (.pt)
        """
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        if self.net.design is not None:
            raise ValueError("Design module should not be present.")
        torch.save(
            {
                "__version__": __version__,
                "vars": self.vars.to_list(),
                "interv_seen": list(self.interv_seen),
                "rnd": tuple(
                    item.tolist() if isinstance(item, np.ndarray) else item
                    for item in self.rnd.get_state()
                ),
                "log_dir": self.log_dir.as_posix(),
                "hparams": dict(self.net.hparams),
                "state_dict": self.net.state_dict(),
            },
            fname,
        )

    @classmethod
    def load(cls, fname: os.PathLike) -> "CASCADE":
        r"""
        Load model from file

        Parameters
        ----------
        fname
            Path to the saved model file (.pt)

        Returns
        -------
        Loaded CASCADE model instance
        """
        loaded = torch.load(fname, weights_only=True)
        version = loaded.pop("__version__", "unknown")
        if version != __version__:
            logger.warning(  # pragma: no cover
                "Loaded model version {} differs from current version {}.",
                version,
                __version__,
            )
        net = CausalNetwork(**loaded.pop("hparams"))
        net.load_state_dict(loaded.pop("state_dict"))
        rnd = get_random_state()
        rnd.set_state(loaded.pop("rnd"))
        model = cls(
            loaded.pop("vars"),
            random_state=rnd,
            log_dir=loaded.pop("log_dir"),
            _net=net,
        )
        model.interv_seen = set(loaded.pop("interv_seen"))
        return model


def upgrade_saved_model(fname: os.PathLike) -> None:  # pragma: no cover
    r"""
    Update the saved model format to be compatible with the latest version of
    CASCADE

    Parameters
    ----------
    fname
        Path to the saved model file (.pt) that needs upgrading
    """
    content = torch.load(fname)
    rewrite = False

    hparams = content["hparams"]
    for k in list(hparams.keys()):
        if "skeleton" in k:
            hparams[k.replace("skeleton", "scaffold")] = hparams.pop(k)
            rewrite = True

    state_dict = content["state_dict"]
    for k in list(state_dict.keys()):
        if "skeleton" in k:
            state_dict[k.replace("skeleton", "scaffold")] = state_dict.pop(k)
            rewrite = True
    if "_fixed_vars" not in state_dict["_extra_state"]:
        state_dict["_extra_state"]["_fixed_vars"] = torch.empty(0, dtype=torch.long)
        rewrite = True

    for k in ("interv_scale", "interv_bias"):
        if f"_{k}" in state_dict:
            state_dict[k] = state_dict.pop(f"_{k}")
            rewrite = True
    for k in (
        "r_design",
        "_interv_scale_design",
        "_interv_bias_design",
        "design_pool",
        "design_pick",
        "target_weight",
    ):
        if k in state_dict:
            state_dict.pop(k)
            rewrite = True

    if rewrite:
        torch.save(content, fname)
