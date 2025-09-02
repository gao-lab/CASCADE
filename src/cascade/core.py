r"""
Core pytorch lightning module and training callbacks for the CASCADE model
"""

import os
from collections.abc import Iterable
from enum import IntEnum
from itertools import zip_longest
from pathlib import Path
from typing import Any

import torch
import torch.distributions as D
from loguru import logger
from pytorch_lightning import LightningModule, Trainer, callbacks
from torch.nn import Parameter, init

from . import nn
from .data import EPS
from .nn import (
    AcycPrior,
    Func,
    IntervDesign,
    Kernel,
    Latent,
    Likelihood,
    Module,
    MultiLinear,
    Scaffold,
    SparsePrior,
    copy_like,
)
from .typing import Kws
from .utils import console, internal


class FitStage(IntEnum):
    r"""
    Model fitting stage

    Attributes
    ----------
    discover
        Causal discover stage
    tune
        Self-reconstruction model tuning stage (after graph acyclification)
    ctfact
        Counterfactual model tuning stage (after graph acyclification)
    design
        Intervention design stage
    """

    discover = 0
    tune = 1
    ctfact = 2
    design = 3


class PredictMode(IntEnum):
    r"""
    Model prediction mode

    Attributes
    ----------
    recon
        Predict self-reconstruction
    jac
        Compute the Jacobian matrix
    explain
        Explain counterfactual prediction by components
    ctmean
        Predict counterfactual state with the mean
    ctsamp
        Predict counterfactual state with random sampling
    """

    recon = 0
    jac = 1
    explain = 2
    dsgnerr = 3
    ctmean = 4
    ctsamp = 5


class LogAdj(IntEnum):
    r"""
    Logging mode of the adjacency matrix in tensorboard

    Attributes
    ----------
    none
        Disable adjacency matrix logging
    mean
        Only log the mean adjacency matrix across SVGD particles
    particles
        Log adjacency matrices of individual SVGD particles
    both
        Log both the mean and individual adjacency matrices
    """

    none = 0
    mean = 1
    particles = 2
    both = 3


class CausalNetwork(LightningModule, Module):
    r"""
    Causal discovery neural network

    Parameters
    ----------
    n_vars
        Number of variables to model
    n_particles
        Number of SVGD particles
    n_covariates
        Dimension of covariates
    n_layers
        Number of MLP layers in the structural equations
    hidden_dim
        MLP hidden layer dimension in the structural equations
    latent_dim
        Dimension of the latent variable
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
    design
        Optional intervention design module, see
        :class:`~cascade.nn.IntervDesign` for details
    """

    EXP_AVG: float = 0.5

    def __init__(
        self,
        n_vars: int,
        n_particles: int,
        n_covariates: int,
        n_layers: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
        beta: float,
        scaffold_mod: str,
        sparse_mod: str,
        acyc_mod: str,
        latent_mod: str,
        lik_mod: str,
        kernel_mod: str,
        scaffold_kws: Kws = None,
        sparse_kws: Kws = None,
        acyc_kws: Kws = None,
        latent_kws: Kws = None,
        lik_kws: Kws = None,
        kernel_kws: Kws = None,
        design: IntervDesign | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="design")
        self.cache = {}

        self.n_vars = n_vars
        self.n_particles = n_particles
        self.n_covariates = n_covariates
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.beta = beta

        self.lam = 0.0
        self.alpha = 0.0
        self.gamma = 0.0

        # Submodules
        self.scaffold: Scaffold = self._get_mod(
            mod=scaffold_mod,
            mod_type=Scaffold,
            n_vars=n_vars,
            n_particles=n_particles,
            **(scaffold_kws or {}),
        )
        self.sparse: SparsePrior = self._get_mod(
            mod=sparse_mod,
            mod_type=SparsePrior,
            n_vars=n_vars,
            n_particles=n_particles,
            **(sparse_kws or {}),
        )
        self.acyc: AcycPrior = self._get_mod(
            mod=acyc_mod,
            mod_type=AcycPrior,
            n_vars=n_vars,
            n_particles=n_particles,
            **(acyc_kws or {}),
        )
        self.kernel: Kernel = self._get_mod(
            mod=kernel_mod,
            mod_type=Kernel,
            **(kernel_kws or {}),
        )
        self.latent: Latent = self._get_mod(
            mod=latent_mod,
            mod_type=Latent,
            n_particles=n_particles,
            latent_dim=latent_dim,
            **(latent_kws or {}),
        )
        self.lik: Likelihood = self._get_mod(
            mod=lik_mod,
            mod_type=Likelihood,
            n_vars=n_vars,
            **(lik_kws or {}),
        )
        self.func: Func = self._get_mod(
            mod="Func",
            mod_type=Func,
            in_features=self.scaffold.max_indegree,
            cov_features=self.latent_dim + self.n_covariates,
            out_features=2,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            multi_dims=(n_particles, n_vars),
            dropout=dropout,
        )
        self.func.layers[-1].weight.data[..., 1, :].fill_(0.0)  # Stabilize disp
        self.design: IntervDesign | None = design

        # Parameters
        self.interv_scale = Parameter(torch.empty(n_particles, n_vars))
        self.interv_bias = Parameter(torch.empty(n_particles, n_vars))

        # Buffers
        n_edges = self.scaffold.n_edges
        self.register_buffer("lik_grad_avg", torch.zeros(n_particles, n_edges))
        self.register_buffer("sparse_grad_avg", torch.zeros(n_particles, n_edges))
        self.register_buffer("acyc_grad_avg", torch.zeros(n_particles, n_edges))
        self.register_buffer("kernel_grad_avg", torch.zeros(n_particles, n_edges))

        self.automatic_optimization = False
        self.reset_parameters()
        self.reset_properties()

    @staticmethod
    def _get_mod(mod: str, mod_type: type, *args, **kwargs) -> Module:
        mod = getattr(nn, mod)
        if issubclass(mod, mod_type):
            return mod(*args, **kwargs)
        raise TypeError(f"Unrecognized {mod_type.__name__} module")

    def reset_parameters(self) -> None:
        init.zeros_(self.interv_scale)
        init.zeros_(self.interv_bias)

    def reset_properties(self) -> None:
        # Optimization
        self.opt = None
        self.lr = None
        self.weight_decay = None
        self.accumulate_grad_batches = None

        # Model flags
        self.fit_stage = None
        self.predict_mode = None
        self.prefit = None

        # Prediction & logging
        self.fixed_vars = None
        self.ablate_latent = None
        self.ablate_interv = None
        self.ablate_graph = None
        self.log_adj = None
        self.vars = None

    def set_design(
        self,
        mask: torch.BoolTensor,
        k: int,
        design_scale_bias: bool,
        target_weight: torch.Tensor,
    ) -> None:
        r"""
        Set the design module

        Parameters
        ----------
        mask
            Boolean mask that marks variables in the design candidate pool
        k
            Maximal combination order to consider
        design_scale_bias
            Whether to optimize interventional scale and bias as well
        target_weight
            Variable weights for computing target deviation
        """
        self.design = self._get_mod(
            mod="IntervDesign",
            mod_type=IntervDesign,
            n_vars=self.n_vars,
            k=k,
            design_scale_bias=design_scale_bias,
            mask=mask,
            interv_scale=self.interv_scale,
            interv_bias=self.interv_bias,
            target_weight=target_weight,
        )

    @property
    def fit_stage(self) -> FitStage:
        r"""
        Prediction mode, see :class:`FitStage` for details
        """
        return self._fit_stage

    @fit_stage.setter
    def fit_stage(self, fit_stage: FitStage | None) -> None:
        if fit_stage is not None:
            self._predict_mode = None
            self.unfreeze_params()
            if fit_stage >= FitStage.tune:
                self.scaffold.freeze_params()
                self.sparse.freeze_params()
                self.acyc.freeze_params()
                self.kernel.freeze_params()
                self.lik.freeze_params()
            if fit_stage >= FitStage.ctfact:
                topo_gens = self.scaffold.topo_gens()
                logger.info(
                    f"Number of topological generations: "
                    f"{[len(gens) for gens in topo_gens]}"
                )
                self._topo_gens = [gens[:-1] for gens in topo_gens]
            if fit_stage >= FitStage.design:
                self.latent.freeze_params()
                self.func.freeze_params()
                self.interv_scale.requires_grad_(False)
                self.interv_bias.requires_grad_(False)
                if self.design is None:
                    raise ValueError("Design module not initialized")
            elif self.design is not None:
                self.design = None
        else:
            self._topo_gens = None
        self._fit_stage = fit_stage

    @property
    def predict_mode(self) -> PredictMode:
        r"""
        Prediction mode, see :class:`PredictMode` for details
        """
        return self._predict_mode

    @predict_mode.setter
    def predict_mode(self, predict_mode: PredictMode | None) -> None:
        if predict_mode is not None:
            self._fit_stage = None
            if predict_mode >= PredictMode.dsgnerr:
                topo_gens = self.scaffold.topo_gens()
                logger.info(
                    f"Number of topological generations: "
                    f"{[len(gens) for gens in topo_gens]}"
                )
                self._topo_gens = [
                    [gen[~torch.isin(gen, self.fixed_vars)] for gen in gens[:-1]]
                    for gens in topo_gens
                ]
        else:
            self._topo_gens = None
        self._predict_mode = predict_mode

    @property
    def prefit(self) -> bool:
        r"""
        Whether to run prefit on the covariates only
        """
        if self.fit_stage == FitStage.discover:
            return self._prefit
        return False

    @prefit.setter
    def prefit(self, prefit: bool) -> None:
        if prefit and not self.n_covariates:
            raise ValueError("Cannot prefit without covariates")
        self._prefit = prefit

    @property
    def topo_gens(self) -> list[list[torch.LongTensor]]:
        r"""
        Topological generations of the causal graph
        """
        if self._topo_gens is None:
            raise AttributeError
        return self._topo_gens

    @property
    def fixed_vars(self) -> torch.LongTensor:
        r"""
        Fixed variables during counterfactual prediction
        """
        return self._fixed_vars

    @fixed_vars.setter
    def fixed_vars(self, fixed_vars: torch.LongTensor | None) -> None:
        if fixed_vars is None:
            self._fixed_vars = torch.empty(0, dtype=torch.long)
            return
        if fixed_vars.min() < 0 or fixed_vars.max() >= self.n_vars:
            raise ValueError("Fixed variables out of bounds")
        self._fixed_vars = fixed_vars

    def _coordinate_device(self) -> None:
        if self._topo_gens is not None:
            self._topo_gens = [
                [gen.to(self.interv_scale.device) for gen in gens]
                for gens in self._topo_gens
            ]

    @internal
    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = getattr(torch.optim, self.opt)(
            [
                {
                    "params": self.regular_params(),
                    "weight_decay": 0.0,
                },
                {
                    "params": self.decay_params(),
                    "weight_decay": self.weight_decay,
                },
            ],
            lr=self.lr,
        )
        return opt

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        l: torch.Tensor,
        l_: torch.Tensor | None = None,
        z: D.Normal | None = None,
        oidx: torch.LongTensor | None = None,
    ) -> tuple[D.Normal, D.Distribution]:
        r"""
        Forward pass of the model

        Parameters
        ----------
        x
            Sample data ([n_particles,] batch_size, n_vars)
        r
            Intervention regime ([n_particles,] batch_size, n_vars)
        s
            Covariate (batch_size, n_covariates)
        l
            Library size (batch_size, 1)
        l\_
            Counterfactual library size (batch_size, 1)
        z
            Latent variable (n_particles, batch_size, latent_dim)
        oidx
            Output variable index

        Returns
        -------
        Latent variable (n_particles, batch_size, latent_dim)
        Data reconstruction distribution
        """
        n_vars = self.n_vars if oidx is None else oidx.numel()

        if z is None:
            z = self.latent(
                torch.zeros_like(r) if self.ablate_latent else r
            )  # (n_particles, bs, latent_dim)
        if self.prefit:
            ptr = x.new_zeros(
                (self.n_particles, n_vars, x.size(-2), self.scaffold.max_indegree)
            )
            z_samp = x.new_zeros((self.n_particles, x.size(-2), self.latent_dim))
        else:
            ptr = self.lik.tone(x, l)  # ([n_particles,] bs, n_vars)
            ptr = self.scaffold.mask_data(ptr, oidx=oidx)
            z_samp = z.rsample() if self.training else z.mean

        z_samp = z_samp.unsqueeze(1).expand(-1, n_vars, -1, -1)
        s = s.expand(self.n_particles, n_vars, -1, -1)
        cov = torch.cat([z_samp, s], dim=-1)
        # (n_particles, n_vars, bs, *)

        if oidx is None:
            func = self.func(ptr, cov)
        else:
            func = self.func(ptr, cov, slice(None), oidx)
        mean, disp = func.permute(3, 0, 2, 1)  # (n_particles, bs, n_vars)

        interv_scale = (
            self.interv_scale if self.design is None else self.design.scale
        ).unsqueeze(1)
        interv_bias = (
            self.interv_bias if self.design is None else self.design.bias
        ).unsqueeze(1)
        # (n_particles, 1, n_vars)
        if oidx is not None:
            interv_scale = interv_scale[..., oidx]
            interv_bias = interv_bias[..., oidx]
            r = r[..., oidx]
        r = torch.zeros_like(r) if self.ablate_interv else r
        interv_scale = (interv_scale * r).exp()
        interv_bias = interv_bias * r

        x_est = self.lik(
            mean * interv_scale + interv_bias,
            disp,
            l if l_ is None else l_,
            oidx=oidx,
        )
        return z, x_est

    def explain(
        self,
        x: torch.Tensor,  # Factual x
        r: torch.Tensor,  # Factual r
        s: torch.Tensor,  # Factual s
        l: torch.Tensor,  # Factual l
        x_: torch.Tensor,  # Counterfactual x
        r_: torch.Tensor,  # Counterfactual r
        s_: torch.Tensor,  # Counterfactual s
        l_: torch.Tensor,  # Counterfactual l
    ) -> tuple[torch.Tensor, ...]:
        r"""
        Explanation pass of the model

        Parameters
        ----------
        x
            Factual data ([n_particles,] batch_size, n_vars)
        r
            Factual intervention regime ([n_particles,] batch_size, n_vars)
        s
            Factual covariates (batch_size, n_covariates)
        l
            Factual library size (batch_size, 1)
        x\_
            Counterfactual data ([n_particles,] batch_size, n_vars)
        r\_
            Counterfactual intervention regime ([n_particles,] batch_size, n_vars)
        s\_
            Counterfactual covariates (batch_size, n_covariates)
        l\_
            Counterfactual library size (batch_size, 1)

        Returns
        -------
        Prediction with all factual components
        Prediction with only the counterfactual intervention scaling and bias
        Prediction with only the counterfactual covariates
        Prediction with only the counterfactual latent variable
        Prediction with the counterfactual value of each parent variable
        Prediction with all counterfactual components
        """
        z = self.latent(r).mean.unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        z_ = self.latent(r_).mean.unsqueeze(1).expand(-1, self.n_vars, -1, -1)
        s = s.expand(self.n_particles, self.n_vars, -1, -1)
        s_ = s_.expand(self.n_particles, self.n_vars, -1, -1)
        cov = torch.cat([z, s], dim=-1)

        if x_.dim() == 3:  # (bs, n_vars, n_particles)
            x_ = x_.permute(2, 0, 1)  # (n_particles, bs, n_vars)
        x = self.lik.tone(x, l)
        x_ = tot = self.lik.tone(x_, l_)
        ptr = self.scaffold.mask_data(x)
        ptr_ = self.scaffold.mask_data(x_)

        interv_scale = (
            self.interv_scale if self.design is None else self.design.scale
        ).unsqueeze(1)
        interv_bias = (
            self.interv_bias if self.design is None else self.design.bias
        ).unsqueeze(1)
        scale = (interv_scale * r).exp()
        bias = interv_bias * r
        scale_ = (interv_scale * r_).exp()
        bias_ = interv_bias * r_

        mean, disp = self.func(ptr, cov).permute(3, 0, 2, 1)
        nil = self.lik.tone(self.lik(mean * scale + bias, disp, l_).mean, l_)
        ctrb_i = self.lik.tone(self.lik(mean * scale_ + bias_, disp, l_).mean, l_)

        mean, disp = self.func(ptr, torch.cat([z, s_], dim=-1)).permute(3, 0, 2, 1)
        ctrb_s = self.lik.tone(self.lik(mean * scale + bias, disp, l_).mean, l_)

        mean, disp = self.func(ptr, torch.cat([z_, s], dim=-1)).permute(3, 0, 2, 1)
        ctrb_z = self.lik.tone(self.lik(mean * scale + bias, disp, l_).mean, l_)

        ctrb_ptr = []
        for i in range(ptr.size(-1)):
            ptr_use = ptr.clone()
            ptr_use[..., i] = ptr_[..., i]  # Plug in parents one by one
            mean, disp = self.func(ptr_use, cov).permute(3, 0, 2, 1)
            ctrb_ptr.append(
                self.lik.tone(self.lik(mean * scale + bias, disp, l_).mean, l_)
            )
        ctrb_ptr = torch.stack(ctrb_ptr, dim=-1)

        return nil, ctrb_i, ctrb_s, ctrb_z, ctrb_ptr, tot

    def cascade(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        l: torch.Tensor,
        l_: torch.Tensor | None = None,
        z: D.Normal | None = None,
    ) -> tuple[D.Normal, D.Distribution]:
        r"""
        Cascade pass of the model

        Parameters
        ----------
        x
            Sample data ([n_particles,] batch_size, n_vars)
        r
            Intervention regime ([n_particles,] batch_size, n_vars)
        s
            Covariate (batch_size, n_covariates)
        l
            Library size (batch_size, 1)
        l\_
            Counterfactual library size (batch_size, 1)
        z
            Latent variable (n_particles, batch_size, latent_dim)

        Returns
        -------
        Latent variable (n_particles, batch_size, latent_dim)
        Data reconstruction distribution
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(self.n_particles, -1, -1)
        remap = torch.empty(self.n_vars, dtype=torch.long, device=x.device)
        empty_gen = torch.empty(0, dtype=torch.long, device=x.device)
        if self.ablate_graph:
            return self(x, r, s, l, l_=l_, z=z)  # (n_particles, bs, *)
        for gens in zip_longest(*self.topo_gens, fillvalue=empty_gen):
            oidx = torch.cat(gens).unique()
            remap[oidx] = torch.arange(oidx.numel(), device=x.device)
            z, x_est = self(x, r, s, l, z=z, oidx=oidx)  # (n_particles, bs, *)
            x_mean = x_est.mean
            x = x.clone()  # Expanded dim becomes independent
            for i, gen in enumerate(gens):
                x[i, :, gen] = x_mean[i, :, remap[gen]]
        return self(x, r, s, l, l_=l_, z=z)  # (n_particles, bs, *)

    def compute_lik(self, batch: Iterable[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        r"""
        Compute likelihood terms from a minibatch

        Parameters
        ----------
        batch
            Minibatch of data

        Returns
        -------
        Negative log-likelihood
        Negative log-prior
        Latent KL divergence
        """
        fit_stage = self.fit_stage
        if fit_stage < FitStage.ctfact:
            x, r, s, l, w = batch
        else:
            x, r, s, l, w, x_, r_, s_, l_, _ = batch
        if fit_stage == FitStage.design:
            r = self.design.rsample(r_.size(0))
            _, x_est = self.cascade(x, r, s_, l, l_=l_)
            x_est = self.lik.tone(x_est.mean, l_)  # (n_particles, bs, n_vars)
            x_tgt = self.lik.tone(x_, l_)  # (bs, n_vars)
            mse = self.design.loss(x_est, x_tgt)  # (n_particles, bs)
            nll = (mse * w).mean(dim=-1)  # (n_particles,)
            nlp = nll.new_zeros(())
            kl = nll.new_zeros(())
        else:
            z, x_est = self(x, r, s, l)  # (n_particles, bs, *)
            log_lik = x_est.log_prob(x)
            log_prior = self.lik.log_prior(x_est)
            nll = (log_lik.mean(dim=-1) * w).mean(dim=-1).neg()  # (n_particles,)
            nlp = (log_prior.mean(dim=-1) * w).mean(dim=-1).neg()  # (n_particles,)
            if fit_stage == FitStage.ctfact:
                _, x_ctfact = self.cascade(x, r_, s_, l, l_=l_)
                log_lik_ctfact = x_ctfact.log_prob(x_)
                log_prior_ctfact = self.lik.log_prior(x_ctfact)
                nll_ctfact = (log_lik_ctfact.mean(dim=-1) * w).mean(dim=-1).neg()
                nlp_ctfact = (log_prior_ctfact.mean(dim=-1) * w).mean(dim=-1).neg()
                nll = 0.5 * nll + 0.5 * nll_ctfact
                nlp = 0.5 * nlp + 0.5 * nlp_ctfact
            kl = (
                D.kl_divergence(z, self.latent.prior()).mean()
                if self.latent_dim
                else nll.new_zeros(())
            )  # (n_particles,)
        return nll, nlp, kl

    def compute_prior(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute the prior energy terms

        Returns
        -------
        Sparse prior energy
        Acyclic prior energy
        """
        if "compute_prior" in self.cache:
            return self.cache["compute_prior"]
        sparse_enrg = self.sparse.energy(self.scaffold)
        acyc_enrg = self.acyc.energy(self.scaffold)
        self.cache["compute_prior"] = (sparse_enrg.detach(), acyc_enrg.detach())
        return sparse_enrg, acyc_enrg

    def compute_kernel(self) -> torch.Tensor:
        r"""
        Compute the SVGD kernel

        Returns
        -------
        SVGD kernel
        """
        if "compute_kernel" in self.cache:
            return self.cache["compute_kernel"]
        kernel = self.kernel(self.scaffold.prob, self.scaffold.prob.detach())
        self.cache["compute_kernel"] = kernel.detach()
        return kernel

    def training_step(
        self, batch: Iterable[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        r"""
        Training step for a minibatch
        """
        nll, nlp, kl = self.compute_lik(batch)
        sparse_enrg, acyc_enrg = self.compute_prior()
        kernel = self.compute_kernel()

        lik_enrg = nll + nlp + self.beta * kl
        if not self.scaffold.frozen and not self.prefit:
            EXP_AVG = (
                0.0
                if torch.allclose(self.lik_grad_avg, torch.as_tensor(0.0))
                else self.EXP_AVG
            )
            self.lik_grad_avg = (
                EXP_AVG * self.lik_grad_avg
                + (1 - EXP_AVG)
                * torch.autograd.grad(
                    lik_enrg.sum(), self.scaffold.logit, retain_graph=True
                )[0]
            )
            self.sparse_grad_avg = (
                EXP_AVG * self.sparse_grad_avg
                + (1 - EXP_AVG)
                * torch.autograd.grad(
                    sparse_enrg.sum(), self.scaffold.logit, retain_graph=True
                )[0]
            )
            self.acyc_grad_avg = (
                EXP_AVG * self.acyc_grad_avg
                + (1 - EXP_AVG)
                * torch.autograd.grad(
                    acyc_enrg.sum(), self.scaffold.logit, retain_graph=True
                )[0]
            )
            self.log_dict(
                {
                    "grad/lik_norm": self.lik_grad_avg.norm(dim=1).mean(),
                    "grad/sparse_norm": self.sparse_grad_avg.norm(dim=1).mean(),
                    "grad/acyc_norm": self.acyc_grad_avg.norm(dim=1).mean(),
                    "grad/kernel_norm": self.kernel_grad_avg.norm(dim=1).mean(),
                },
                sync_dist=True,
            )

        prior_enrg = self.lam * sparse_enrg + self.alpha * acyc_enrg  # (n_particles,)
        post_enrg = lik_enrg + prior_enrg

        self.log_dict(
            {
                "train/nll": nll.mean(),
                "train/nlp": nlp.mean(),
                "train/kl": kl.mean(),
                "train/lik_enrg": lik_enrg.mean(),
                "train/sparse_enrg": sparse_enrg.mean(),
                "train/acyc_enrg": acyc_enrg.mean(),
                "train/prior_enrg": prior_enrg.mean(),
                "train/post_enrg": post_enrg.mean(),
                "train/kernel": kernel.mean(),
            },
            sync_dist=True,
        )  # OK: Consider how to properly log SVGD

        self.backward(post_enrg, kernel)
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            opt = self.optimizers()
            if self.accumulate_grad_batches > 1:
                seen = set()
                for group in opt.param_groups:
                    for p in group["params"]:
                        if p.grad is None or id(p) in seen:
                            continue
                        p.grad.div_(self.accumulate_grad_batches)
                        seen.add(id(p))
            opt.step()
            opt.zero_grad()

    def backward(self, enrg: torch.Tensor, kernel: torch.Tensor) -> None:
        r"""
        Implementation of the main SVGD logic


        .. caution::

            This implementation is only valid for symmetric and
            translation-invariant kernels.
        """
        self.scaffold.zero_grad(backup=True)
        enrg_sum = enrg.sum()
        enrg_sum.backward(retain_graph=True)
        if self.scaffold.frozen:
            return
        enrg_grad, self.scaffold.logit.grad = self.scaffold.logit.grad, None
        kernel_grad = torch.autograd.grad(
            kernel.sum(), self.scaffold.logit, allow_unused=True  # Allow KroneckerDelta
        )[0]
        if kernel_grad is None:
            kernel_grad = torch.zeros_like(self.scaffold.logit)
        EXP_AVG = (
            0.0
            if torch.allclose(self.kernel_grad_avg, torch.as_tensor(0.0))
            else self.EXP_AVG
        )
        self.kernel_grad_avg = (
            EXP_AVG * self.kernel_grad_avg + (1 - EXP_AVG) * kernel_grad
        )
        logit_grad = (
            kernel.detach().matmul(enrg_grad) + self.gamma * kernel_grad
        ) / self.n_particles
        self.scaffold.zero_grad(backup=False)
        self.scaffold.logit.backward(gradient=logit_grad, retain_graph=True)
        self.scaffold.accumulate_grad()

    def validation_step(
        self, batch: Iterable[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        r"""
        Validation step for a minibatch
        """
        nll, nlp, kl = self.compute_lik(batch)
        sparse_enrg, acyc_enrg = self.compute_prior()
        kernel = self.compute_kernel()

        lik_enrg = nll + nlp + self.beta * kl
        prior_enrg = self.lam * sparse_enrg + self.alpha * acyc_enrg  # (n_particles,)
        post_enrg = lik_enrg + prior_enrg

        self.log_dict(
            {
                "val/nll": nll.mean(),
                "val/nlp": nlp.mean(),
                "val/kl": kl.mean(),
                "val/lik_enrg": lik_enrg.mean(),
                "val/sparse_enrg": sparse_enrg.mean(),
                "val/acyc_enrg": acyc_enrg.mean(),
                "val/prior_enrg": prior_enrg.mean(),
                "val/post_enrg": post_enrg.mean(),
                "val/kernel": kernel.mean(),
                "hparam/lam": self.lam,
                "hparam/alpha": self.alpha,
                "hparam/gamma": self.gamma,
            },
            sync_dist=True,
        )  # OK: Consider how to properly log SVGD

    def predict_step(
        self, batch: Iterable[torch.Tensor], batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        r"""
        Prediction step for a minibatch
        """
        predict_mode = self.predict_mode

        if predict_mode == PredictMode.recon:
            x, r, s, l, _ = batch
            z, x_est = self(x, r, s, l)
            return (
                z.mean.cpu(),
                z.variance.sqrt().cpu(),
                x_est.mean.cpu(),
                x_est.variance.sqrt().cpu(),
                self.lik.get_disp(x_est).cpu(),
            )

        if predict_mode == PredictMode.jac:
            x, r, s, l, _ = batch
            x = x.expand(self.n_particles, -1, -1)
            with torch.enable_grad():
                x = x.requires_grad_()
                _, x_est = self(x, r, s, l)
                x_mean = x_est.mean  # (n_particles, bs, n_vars)
                grad_outputs = torch.eye(
                    x_mean.size(-1), dtype=x_mean.dtype, device=x_mean.device
                )  # (n_vars, n_vars)
                grad_outputs = grad_outputs.view(
                    grad_outputs.size(0), 1, 1, grad_outputs.size(1)
                ).expand(-1, x_mean.size(0), x_mean.size(1), -1)
                # (n_vars, n_particles, bs, n_vars)
                return (
                    torch.stack(
                        [
                            torch.autograd.grad(
                                x_mean,
                                x,
                                grad_outputs=g,  # (n_particles, bs, n_vars)
                                retain_graph=True,
                            )[0].cpu()
                            for g in grad_outputs
                        ],
                        dim=-2,
                    ),
                )
                # This is slower than the is_grads_batched approach but
                # much more memory-efficient. The latter easily runs OOM.

        if predict_mode == PredictMode.explain:
            x, r, s, l, _, x_, r_, s_, l_, _ = batch
            nil, ctrb_i, ctrb_s, ctrb_z, ctrb_ptr, tot = self.explain(
                x, r, s, l, x_, r_, s_, l_
            )
            return (
                nil.cpu(),
                ctrb_i.cpu(),
                ctrb_s.cpu(),
                ctrb_z.cpu(),
                ctrb_ptr.cpu(),
                tot.cpu(),
            )

        if predict_mode == PredictMode.dsgnerr:
            x, r, s, l, _, x_, _, s_, l_, _ = batch
            _, x_est = self.cascade(x, r, s_, l, l_=l_)
            x_est = self.lik.tone(x_est.mean, l_)  # (n_particles, bs, n_vars)
            x_tgt = self.lik.tone(x_, l_)  # (bs, n_vars)
            return (self.design.loss(x_est, x_tgt),)  # (n_particles, bs)

        # predict_mode >= PredictMode.ctmean:
        x, r, s, l, _ = batch
        z, x_est = self.cascade(x, r, s, l)
        return (
            z.mean.cpu(),
            z.variance.sqrt().cpu(),
            (
                x_est.mean.cpu()
                if predict_mode == PredictMode.ctmean
                else x_est.sample().cpu()  # PredictMode.ctsamp
            ),
            x_est.variance.sqrt().cpu(),
            self.lik.get_disp(x_est).cpu(),
        )

    @internal
    def on_train_batch_start(
        self, batch: Iterable[torch.Tensor], batch_idx: int
    ) -> None:
        if self.fit_stage == FitStage.design:
            self.eval()
            comb = self.design.comb_lists
            topk = self.design.logits.data.topk(min(10, self.design.logits.size(0)))
            top_str = "  |  ".join(
                f"{','.join(sorted(self.vars[comb[idx]]))} ({val:.2f})"
                for idx, val in zip(topk.indices.tolist(), topk.values.tolist())
            )
            self.logger.experiment.add_text(
                "design/top", top_str, global_step=self.global_step
            )
        elif self.fit_stage == FitStage.discover:
            self.scaffold.clear_cached_properties()
            self.cache.clear()

    @internal
    def on_validation_start(self) -> None:
        if self.fit_stage == FitStage.discover:
            self.scaffold.clear_cached_properties()
            self.cache.clear()
        if self.log_adj in (LogAdj.mean, LogAdj.both):
            self.logger.experiment.add_image(
                "adj/mean",
                self.scaffold.mean_adj.detach().cpu().float().to_dense(),
                global_step=self.global_step,
                dataformats="HW",
            )
        if self.log_adj in (LogAdj.particles, LogAdj.both):
            adj = self.scaffold.adj.detach().cpu().float().to_dense().permute(2, 0, 1)
            for i, particle in enumerate(adj):
                self.logger.experiment.add_image(
                    f"adj/particle_{i}",
                    particle,
                    global_step=self.global_step,
                    dataformats="HW",
                )

    @internal
    def on_fit_start(self) -> None:
        self._coordinate_device()
        self.scaffold.clear_cached_properties()
        self.cache.clear()

    @internal
    def on_fit_end(self) -> None:
        self.scaffold.clear_cached_properties()
        self.cache.clear()
        self.reset_properties()
        torch.cuda.empty_cache()

    @internal
    def on_predict_start(self) -> None:
        self._coordinate_device()
        self.scaffold.clear_cached_properties()
        self.cache.clear()

    @internal
    def on_predict_end(self) -> None:
        self.scaffold.clear_cached_properties()
        self.cache.clear()
        self.reset_properties()
        torch.cuda.empty_cache()

    def prune(self) -> None:
        r"""
        Prune the scaffold and structural equations accordingly
        """
        old_map = {(i, j): k for i, j, k in self.scaffold.idx.t().tolist()}
        mask = self.scaffold.prune()
        self.lik_grad_avg = self.lik_grad_avg[:, mask]
        self.sparse_grad_avg = self.sparse_grad_avg[:, mask]
        self.acyc_grad_avg = self.acyc_grad_avg[:, mask]
        self.kernel_grad_avg = self.kernel_grad_avg[:, mask]
        new_map = {(i, j): k for i, j, k in self.scaffold.idx.t().tolist()}
        self.hparams["scaffold_kws"]["eidx"] = self.scaffold.idx[:2].cpu()

        trailing = self.latent_dim + self.n_covariates
        old = self.func.layers[0]
        new = MultiLinear(
            in_features=self.scaffold.max_indegree + trailing,
            out_features=old.out_features,
            multi_dims=(self.n_particles, self.n_vars),
        )
        copy_like(old.bias, new.bias)
        init.zeros_(new.weight)
        new.weight.data = new.weight.data.to(
            device=old.weight.device, dtype=old.weight.dtype
        )
        for (i, j), k in new_map.items():
            new.weight.data[:, j, :, k] = old.weight.data[:, j, :, old_map[(i, j)]]
        if trailing:
            new.weight.data[:, :, :, -trailing:] = old.weight.data[:, :, :, -trailing:]
        self.func.layers[0] = new

    @internal
    def get_extra_state(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "opt": self.opt,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "_fit_stage": None if self._fit_stage is None else int(self._fit_stage),
            "_predict_mode": (
                None if self._predict_mode is None else int(self._predict_mode)
            ),
            "_prefit": self._prefit,
            "_topo_gens": self._topo_gens,
            "_fixed_vars": self._fixed_vars,
            "log_adj": None if self.log_adj is None else int(self.log_adj),
            **super().get_extra_state(),
        }

    @internal
    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.lam = state.pop("lam")
        self.alpha = state.pop("alpha")
        self.gamma = state.pop("gamma")
        self.opt = state.pop("opt")
        self.lr = state.pop("lr")
        self.weight_decay = state.pop("weight_decay")
        self.accumulate_grad_batches = state.pop("accumulate_grad_batches")
        _fit_stage = state.pop("_fit_stage")
        self._fit_stage = None if _fit_stage is None else FitStage(_fit_stage)
        _predict_mode = state.pop("_predict_mode")
        self._predict_mode = (
            None if _predict_mode is None else PredictMode(_predict_mode)
        )
        self._prefit = state.pop("_prefit")
        self._topo_gens = state.pop("_topo_gens")
        self._fixed_vars = state.pop("_fixed_vars")
        log_adj = state.pop("log_adj")
        self.log_adj = None if log_adj is None else LogAdj(log_adj)
        super().set_extra_state(state)


class DiscoverScheduler(callbacks.EarlyStopping):
    r"""
    Hyperparameter scheduler for causal discovery

    Parameters
    ----------
    monitor
        Loss to be monitored
    constraint
        Loss that specifies the constraint
    patience
        Number of checks with no improvement after which training will be stopped
    tolerance
        Maximal tolerance of constraint violation to end the scheduler
    lam
        Sparse penalty rate (:math:`\eta_\lambda` in paper)
    alpha
        Acyclic penalty rate (:math:`\eta_\alpha` in paper)
    gamma
        Kernel gradient rate (:math:`\eta_\gamma`)
    **kwargs
        Additional keyword arguments are passed to
        :class:`~lightning.pytorch.callbacks.EarlyStopping`
    """

    inf = {"min": torch.tensor(torch.inf), "max": torch.tensor(-torch.inf)}

    def __init__(
        self,
        monitor: str,
        constraint: str,
        patience: int,
        tolerance: float = None,
        lam: float = None,
        alpha: float = None,
        gamma: float = None,
        **kwargs,
    ) -> None:
        if kwargs.get("check_on_train_epoch_end", False):
            raise ValueError(
                "Only supports checking on validation epoch end"
            )  # pragma: no cover
        kwargs["check_on_train_epoch_end"] = False
        super().__init__(monitor, patience=patience, **kwargs)
        self.constraint = constraint
        self.tolerance = tolerance
        self.lam = lam
        self.alpha = alpha
        self.gamma = gamma
        self.stall_patience = patience
        self.stall_count = 0
        self.prefit = None
        self.violation = None
        self.trigger_flag = None
        self.min_violation = float("inf")

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor, constraint=self.constraint, mode=self.mode
        )

    @internal
    def state_dict(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "lam": self.lam,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "stall_patience": self.stall_patience,
            "stall_count": self.stall_count,
            "prefit": self.prefit,
            "violation": self.violation,
            "trigger_flag": self.trigger_flag,
            "min_violation": self.min_violation,
            **super().state_dict(),
        }

    @internal
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # pragma: no cover
        self.tolerance = state_dict.pop("tolerance")
        self.lam = state_dict.pop("lam")
        self.alpha = state_dict.pop("alpha")
        self.gamma = state_dict.pop("gamma")
        self.stall_patience = state_dict.pop("stall_patience")
        self.stall_count = state_dict.pop("stall_count")
        self.prefit = state_dict.pop("prefit")
        self.violation = state_dict.pop("violation")
        self.trigger_flag = state_dict.pop("trigger_flag")
        self.min_violation = state_dict.pop("min_violation")
        super().load_state_dict(state_dict)

    def on_validation_end(self, trainer: Trainer, pl_module: CausalNetwork) -> None:
        r"""
        Main logic of the scheduler


        .. note::

            The scheduler will adjust the hyperparameters of the model according
            to the gradient norms of the likelihood, sparse, acyclic, and kernel
            gradients, each time the early stopping criteria is met. The
            adjustment is based on the ratio of the likelihood gradient norm to
            the other gradient norms. The scheduler continues until the
            constraint is satisfied or constraint violation stops improving for
            a consecutive ``patience`` times.
        """
        self.prefit = pl_module.prefit
        self.violation = trainer.callback_metrics[self.constraint]
        super().on_validation_end(trainer, pl_module)
        pl_module.prefit = self.prefit

        if self.trigger_flag:
            lik_grad_norm = pl_module.lik_grad_avg.norm(dim=1).mean().item()
            sparse_grad_norm = pl_module.sparse_grad_avg.norm(dim=1).mean().item()
            acyc_grad_norm = pl_module.acyc_grad_avg.norm(dim=1).mean().item()
            kernel_grad_norm = pl_module.kernel_grad_avg.norm(dim=1).mean().item()
            pl_module.lam = self.lam * lik_grad_norm / (sparse_grad_norm + EPS)
            pl_module.alpha += self.alpha * lik_grad_norm / (acyc_grad_norm + EPS)
            if pl_module.n_particles > 1:
                pl_module.gamma = self.gamma * lik_grad_norm / (kernel_grad_norm + EPS)

            self.best_score = self.inf[self.mode]
            if trainer.checkpoint_callback is not None:
                inf = self.inf[trainer.checkpoint_callback.mode]
                for m in trainer.checkpoint_callback.best_k_models:
                    trainer.checkpoint_callback.best_k_models[m] = inf
                trainer.checkpoint_callback.kth_value = inf
                trainer.checkpoint_callback.best_model_score = inf
            trainer.checkpoint_callback.skip_once = True

            if self.verbose:
                console.print(
                    f"Discover scheduler triggered: "
                    f"lam = [hl]{pl_module.lam:.2e}[/hl], "
                    f"alpha = [hl]{pl_module.alpha:.2e}[/hl], "
                    f"gamma = [hl]{pl_module.gamma:.2e}[/hl]"
                )
            self.trigger_flag = False

    def _evaluate_stopping_criteria(
        self, current: torch.Tensor
    ) -> tuple[bool, str | None]:
        should_stop, reason = super()._evaluate_stopping_criteria(current)
        self.trigger_flag = False
        if not should_stop:
            return should_stop, reason
        if self.prefit:
            self.prefit, should_stop = False, False
            reason = "Prefit concluded"
            return should_stop, reason
        improve_ratio = (self.min_violation - self.violation) / self.min_violation
        self.min_violation = min(self.violation, self.min_violation)
        if improve_ratio < 0.01:
            self.stall_count += 1
            console.print(f"Discover scheduler stall #{self.stall_count}")
            if self.stall_count >= self.stall_patience:
                reason = "Violation stalled"
                return should_stop, reason
        else:
            self.stall_count = 0
        if self.min_violation > self.tolerance:
            self.trigger_flag, should_stop = True, False
            reason = "Constraint unsatisfied"
        else:
            reason = "Constraint satisfied"
        return should_stop, reason


class ModelCheckpoint(callbacks.ModelCheckpoint):
    r"""
    Custom model checkpoint callback that can be configured to skip saving the
    model once
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_once = False

    @internal
    def on_validation_end(self, trainer: Trainer, pl_module: CausalNetwork) -> None:
        if self.skip_once:
            logger.debug("Skipping model checkpoint.")
            self.skip_once = False
            return
        super().on_validation_end(trainer, pl_module)

    @internal
    def state_dict(self) -> dict[str, Any]:
        return {
            "skip_once": self.skip_once,
            **super().state_dict(),
        }

    @internal
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.skip_once = state_dict.pop("skip_once")
        super().load_state_dict(state_dict)


class PredictionWriter(callbacks.BasePredictionWriter):
    r"""
    Custom prediction writer to enable multi-device prediction
    """

    def __init__(self, output_dir: os.PathLike) -> None:
        super().__init__(write_interval="epoch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @internal
    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: Any,
        batch_indices: Any,
    ) -> None:
        torch.save(
            predictions,
            self.output_dir / f"pred{trainer.global_rank}.pt",
        )
        torch.save(
            batch_indices,
            self.output_dir / f"ind{trainer.global_rank}.pt",
        )
