r"""
Neural network utilities
"""

import os
from collections.abc import Generator
from functools import cached_property
from itertools import product
from math import log, log1p, sqrt
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData
from loguru import logger
from scipy.sparse import issparse
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Parameter, init

from . import __version__
from .data import EPS, _get_size, _get_X
from .utils import count_occurrence, index_len, internal, non_unitary_index


def copy_like(source: torch.Tensor, target: torch.Tensor) -> None:
    r"""
    Copy tensor device, dtype and data from a source tensor to a target tensor
    in place

    Parameters
    ----------
    source
        Source tensor
    target
        Target tensor
    """
    target.data = target.data.to(device=source.device, dtype=source.dtype)
    target.data.copy_(source.data)


def mean_squared_error(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""
    Compute the mean squared error along a specified dimension

    Parameters
    ----------
    x
        Input tensor x
    y
        Input tensor y
    dim
        Dimension along which to compute the error
    keepdim
        Whether to keep the dimension after reduction
    weight
        Optional weight tensor

    Returns
    -------
    Mean squared error tensor
    """
    if x.size() != y.size():
        raise ValueError("Incompatible input sizes")
    if weight is None:
        weight = x.new_ones(x.size(dim))
    elif weight.ndim != 1 or weight.size(0) != x.size(dim):
        raise ValueError("Incompatible weight")
    else:
        weight = weight.numel() * weight / weight.sum()
    if dim < -1:
        weight = weight.view(-1, *((1,) * (-dim - 1)))
    elif dim > -1:
        weight = weight.view(-1, *((1,) * (x.ndim - dim - 1)))
    return ((x - y).square() * weight).mean(dim, keepdim=keepdim)


def gumbel_sigmoid(x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    r"""
    Straight-through Gumbel sigmoid sampler

    Parameters
    ----------
    x
        Logit tensor
    tau
        Temperature parameter

    Returns
    -------
    Hard reparameterized samples
    """
    noise = torch.empty_like(x).uniform_(EPS, 1 - EPS).logit()
    y_soft = torch.sigmoid((x + noise) / tau)
    y_hard = (y_soft > 0.5).type_as(y_soft)
    return y_hard.detach() - y_soft.detach() + y_soft


def multi_trace(m: torch.Tensor) -> torch.Tensor:
    r"""
    Compute matrix trace with support for multiplex dims

    Parameters
    ----------
    m
        Matrix of shape (\*m, n_vars, n_vars)

    Returns
    -------
    Matrix trace of shape (\*m,)
    """
    return m.diagonal(dim1=-1, dim2=-2).sum(dim=-1)


def multi_rbf(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    r"""
    RBF kernel with support for multiplex dims

    Parameters
    ----------
    x
        Input x of shape (\*m, bs, n_vars)
    y
        Input y of shape (\*m, bs, n_vars)

    Returns
    -------
    RBF kernel of shape (\*m, bs, bs)
    """
    cdist = torch.cdist(x, y)  # (*m, bs, bs)
    med = cdist.detach().flatten(start_dim=-2).quantile(0.5, dim=-1)  # (*m,)
    scale = log(x.size(-2)) / (med.square() + EPS)
    return (cdist.square() * scale.unsqueeze(-1).unsqueeze(-2)).neg().exp()


class Module(torch.nn.Module):
    r"""
    Abstract module class supporting parameter freezing, decayed / non-decayed
    parameter iteration, and cached property clearing
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze_params(self) -> None:
        r"""
        Freeze parameters and turn on evaluation mode
        """
        for param in self.parameters(recurse=False):
            param.requires_grad_(False)
        self.eval()
        for name, module in self.named_children():
            if isinstance(module, Module):
                module.freeze_params()
            else:
                logger.debug(f"Skipping native torch submodule {name}.")
        self.clear_cached_properties()
        self._frozen = True

    def unfreeze_params(self) -> None:
        r"""
        Unfreeze parameters and turn on training mode
        """
        for param in self.parameters(recurse=False):
            param.requires_grad_(True)
        self.train()
        for name, module in self.named_children():
            if isinstance(module, Module):
                module.unfreeze_params()
            else:
                logger.debug(f"Skipping native torch submodule {name}.")
        self.clear_cached_properties()
        self._frozen = False

    def _decay_params(self) -> Generator[Parameter, None, None]:
        return
        yield

    def decay_params(self) -> Generator[Parameter, None, None]:
        r"""
        Iterate through weight decayed parameters
        """
        for param in self._decay_params():
            if param.numel() and param.requires_grad:
                yield param
        for name, module in self.named_children():
            if isinstance(module, Module):
                yield from module.decay_params()
            else:
                logger.debug(f"Skipping native torch submodule {name}.")

    def regular_params(self) -> Generator[Parameter, None, None]:
        r"""
        Iterate through non-decayed parameters
        """
        decay_params = set(self._decay_params())
        for param in self.parameters(recurse=False):
            if param in decay_params:
                continue
            if param.numel() and param.requires_grad:
                yield param
        for name, module in self.named_children():
            if isinstance(module, Module):
                yield from module.regular_params()
            else:
                logger.debug(f"Skipping native torch submodule {name}.")

    @internal
    def get_extra_state(self) -> dict[str, Any]:
        return {"_frozen": self._frozen}

    @internal
    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._frozen = state.pop("_frozen")

    def clear_cached_properties(self):
        r"""
        Clear cached properties to allow re-calculation
        """
        cls = type(self)
        for key in list(self.__dict__.keys()):
            if isinstance(getattr(cls, key, None), cached_property):
                delattr(self, key)


class ModuleList(torch.nn.ModuleList, Module):
    r"""
    A module list with the :class:`Module` capabilities
    """


class MultiLinear(Module):
    r"""
    Linear layer with support for multi-dims

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    multi_dims
        Multiplex dims at the front of input samples
    """

    def __init__(
        self, in_features: int, out_features: int, multi_dims: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.multi_dims = multi_dims
        self.weight = Parameter(torch.empty(*multi_dims, out_features, in_features))
        self.bias = Parameter(torch.empty(*multi_dims, 1, out_features))
        self.reset_parameters()

    @internal
    def reset_parameters(self) -> None:
        for s in product(*(range(dim) for dim in self.multi_dims)):
            init.kaiming_uniform_(self.weight[s], a=sqrt(5))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[s])
            bound = 1 / sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias[s], -bound, bound)

    def forward(
        self, x: torch.Tensor, *multi_idx: slice | torch.LongTensor
    ) -> torch.Tensor:
        if multi_idx:
            weight, bias = self.weight[*multi_idx], self.bias[*multi_idx]
        else:
            weight, bias = self.weight, self.bias
        return x.matmul(weight.transpose(-1, -2)) + bias

    def _decay_params(self) -> Generator[Parameter, None, None]:
        yield self.weight
        yield from super()._decay_params()


class Func(Module):
    r"""
    Structural equation with covariates

    Parameters
    ----------
    in_features
        Input dimensionality
    cov_features
        Covariate dimensionality
    out_features
        Output dimensionality
    hidden_dim
        Hidden layer dimensionality
    n_layers
        Number of hidden layers
    multi_dims
        Multiplex dims at the front of input samples
    dropout
        Dropout rate
    """

    def __init__(
        self,
        in_features: int,
        cov_features: int,
        out_features: int,
        hidden_dim: int,
        n_layers: int,
        multi_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                MultiLinear(in_features + cov_features, hidden_dim, multi_dims)
            )
            self.layers.append(LeakyReLU(negative_slope=0.2))
            if dropout:
                self.layers.append(Dropout(p=dropout))
            in_features = hidden_dim
        self.layers.append(
            MultiLinear(in_features + cov_features, out_features, multi_dims)
        )

    def forward(
        self, x: torch.Tensor, cov: torch.Tensor, *multi_idx: slice | torch.LongTensor
    ) -> torch.Tensor:
        ptr = x
        for layer in self.layers:
            if isinstance(layer, MultiLinear):
                ptr = layer(torch.cat([ptr, cov], axis=-1), *multi_idx)
            else:
                ptr = layer(ptr)
        return ptr


class AttnPool(Module):
    r"""
    Attention-based pooling layer to combine multiple intervention embeddings

    Parameters
    ----------
    emb_dim
        Embedding dimensionality
    """

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.norm = 1 / sqrt(emb_dim)
        self.query_proj = Linear(emb_dim, emb_dim)
        self.key_proj = Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(
            x.sum(dim=-2, keepdim=True)
        )  # ([n_particles,] bs, 1, emb_dim)
        key = self.key_proj(x)  # ([n_particles,] bs, n_vars, emb_dim)
        attention = (
            (query * key).sum(dim=-1, keepdim=True) * self.norm
        ).sigmoid()  # ([n_particles,] bs, n_vars, 1)
        return (attention * x).sum(dim=-2)  # ([n_particles,] bs, emb_dim)


# ------------------------------ Graph scaffold --------------------------------


class Scaffold(Module):
    r"""
    Abstract graph scaffold

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    eidx
        Scaffold edge indices of shape (2, n_edges)
    tau
        Gumbel-sigmoid temperature
    """

    def __init__(
        self,
        n_vars: int,
        n_particles: int,
        eidx: torch.LongTensor,
        tau: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_particles = n_particles
        if (eidx < 0).any() or (eidx >= n_vars).any():
            raise ValueError("Edge index out of bounds")  # pragma: no cover
        if (eidx[0] == eidx[1]).any():
            raise ValueError("Self-loops are not allowed")
        eidx = eidx[:, eidx[0].argsort(stable=True)]
        i, j = eidx
        k = self.make_k(j)
        self.max_indegree = k.max() + 1 if k.numel() else 0
        self.register_buffer("idx", torch.stack([i, j, k]))
        self.register_buffer("_logit", torch.zeros(self.n_particles, self.n_edges))
        self.tau = tau
        self.grad_backup = {}
        self.kwargs = kwargs

    @property
    def n_edges(self) -> int:
        return self.idx.size(1)

    @staticmethod
    def make_k(j: torch.LongTensor) -> torch.LongTensor:
        return torch.as_tensor(
            count_occurrence(j.tolist()), dtype=j.dtype, device=j.device
        )

    def construct_sparse_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            self.idx[:2],
            value,
            size=(self.n_vars, self.n_vars, *value.size()[1:]),
        )

    def compute_logit(self) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    @cached_property
    def logit(self) -> torch.Tensor:
        r"""
        Edge logit of shape (n_particles, n_edges)
        """
        logit = self._logit if self.frozen else self.compute_logit()
        if logit.requires_grad:
            logit.retain_grad()
        return logit

    @cached_property
    def prob(self) -> torch.Tensor:
        r"""
        Edge prob of shape (n_particles, n_edges)
        """
        return self.logit.sigmoid()

    @cached_property
    def adj(self) -> torch.Tensor:
        r"""
        Sparse adjacency matrix of all particles
        """
        return self.construct_sparse_tensor(self.prob.t())

    @cached_property
    def mean_adj(self) -> torch.Tensor:
        r"""
        Mean sparse adjacency matrix
        """
        return self.construct_sparse_tensor(self.prob.mean(dim=0))

    @cached_property
    def complete_adj(self) -> torch.Tensor:
        r"""
        Complete sparse adjacency matrix
        """
        return self.construct_sparse_tensor(self._logit.new_ones(self.n_edges, 1))

    @property
    def mask_map(self) -> torch.LongTensor:
        r"""
        A reshaped index map of shape (n_vars, max_indegree) where entry (j, k)
        has value i, indicating which input gene is in each reshaped position
        for each output gene.
        """
        mask_map = self.idx.new_zeros(self.n_vars, self.max_indegree) - 1
        i, j, k = self.idx
        mask_map[j, k] = i
        return mask_map

    def mask_data(
        self, x: torch.Tensor, oidx: torch.LongTensor | None = None
    ) -> torch.Tensor:
        if oidx is not None:
            mask = torch.isin(self.idx[1], oidx)
            i, j, k = self.idx[:, mask]
            remap = j.new_empty(self.n_vars)
            remap[oidx] = torch.arange(oidx.numel(), device=remap.device)
            j = remap[j]
            logit = self.logit[:, mask]
            n_vars = oidx.numel()
        else:
            i, j, k = self.idx
            logit = self.logit
            n_vars = self.n_vars

        if x.dim() == 2:  # (bs, n_vars)
            x = x.unsqueeze(0)
        bs = x.size(-2)
        if self.training and not self.frozen:
            samp = gumbel_sigmoid(logit.unsqueeze(-1).expand(-1, -1, bs), tau=self.tau)
        else:
            samp = (logit > 0).unsqueeze(-1).expand(-1, -1, bs)

        samp_reshape = samp.new_zeros((self.n_particles, n_vars, bs, self.max_indegree))
        samp_reshape[:, j, :, k] = samp.transpose(0, 1)

        x_reshape = x.new_zeros((x.size(0), n_vars, bs, self.max_indegree))
        x_reshape[:, j, :, k] = x[:, :, i].moveaxis(-1, 0)
        return x_reshape * samp_reshape

    def zero_grad(self, set_to_none: bool = True, backup: bool = False) -> None:
        if backup:
            for name, param in self.named_parameters():
                self.grad_backup[name] = (
                    None if param.grad is None else param.grad.detach().clone()
                )
        super().zero_grad(set_to_none=set_to_none)
        if hasattr(self, "logit") and self.logit.retains_grad:
            self.logit.grad = None

    def accumulate_grad(self) -> None:
        for name, param in self.named_parameters():
            grad = self.grad_backup.pop(name, None)
            if grad is not None:
                param.grad.add_(grad)

    def export_graph(self, edge_attr: str = "weight") -> nx.DiGraph:
        self.clear_cached_properties()
        i, j, _ = self.idx
        i = i.numpy(force=True)
        j = j.numpy(force=True)
        prob = self.prob.numpy(force=True).T.tolist()

        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n_vars))  # Ensure proper node order
        graph.add_weighted_edges_from(
            ((u, v, w) for u, v, w in zip(i, j, prob)), weight=edge_attr
        )
        return graph

    def import_graph(self, graph: nx.DiGraph, edge_attr: str = "weight") -> None:
        i, j, _ = self.idx
        i = i.numpy(force=True)
        j = j.numpy(force=True)
        attrs = nx.get_edge_attributes(graph, edge_attr)
        zeros = [0.0] * self.n_particles
        prob = torch.as_tensor(
            [attrs.get((u, v), zeros) for u, v in zip(i, j)],
            dtype=self.logit.dtype,
            device=self.logit.device,
        )
        self._logit.copy_(prob.logit().T)
        self.freeze_params()

    def __getitem__(self, index) -> "Scaffold":
        result = type(self)(
            self.n_vars,
            index_len(index, self.n_particles),
            self.idx[:2],
            self.tau,
            **self.kwargs,
        )
        index = non_unitary_index(index)
        copy_like(self._logit[index], result._logit)
        result.set_extra_state(self.get_extra_state())
        return result

    def prune(self) -> torch.BoolTensor:
        self.clear_cached_properties()
        mask = (self.logit > 0).any(dim=0)
        i, j = self.idx[:2, mask]
        k = self.make_k(j)
        self.max_indegree = k.max() + 1 if k.numel() else 0
        self.idx = torch.stack([i, j, k])
        self._logit = self._logit[:, mask]
        return mask

    def topo_gens(self) -> list[list[torch.LongTensor]]:
        gens_list = []
        for i in range(self.n_particles):
            particle = self[i]
            particle.prune()
            graph = particle.export_graph()
            gens = [torch.as_tensor(gen) for gen in nx.topological_generations(graph)]
            gens_list.append(gens)
        return gens_list


class Edgewise(Scaffold):
    r"""
    Edgewise parameterized edge logits

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    eidx
        Scaffold edge indices of shape (2, n_edges)
    tau
        Gumbel-sigmoid temperature
    """

    INIT_STD: float = 0.1

    def __init__(
        self,
        n_vars: int,
        n_particles: int,
        eidx: torch.LongTensor,
        tau: float = 10.0,
    ) -> None:
        super().__init__(n_vars, n_particles, eidx, tau=tau)
        self.edgewise = Parameter(torch.empty(self.n_particles, self.n_edges))
        self.reset_parameters()

    @internal
    def reset_parameters(self) -> None:
        init.normal_(self.edgewise, std=self.INIT_STD)

    def compute_logit(self) -> torch.Tensor:
        return self.edgewise.view_as(self.edgewise)  # Make non-leaf

    def __getitem__(self, index) -> "Edgewise":
        result = super().__getitem__(index)
        index = non_unitary_index(index)
        copy_like(self.edgewise[index], result.edgewise)
        return result

    def prune(self) -> torch.BoolTensor:
        mask = super().prune()
        edgewise = self.edgewise.data[:, mask]
        self.edgewise = Parameter(torch.empty_like(edgewise))
        self.edgewise.data.copy_(edgewise)
        return mask

    def _decay_params(self) -> Generator[Parameter, None, None]:
        yield self.edgewise
        yield from super()._decay_params()


class Bilinear(Scaffold):
    r"""
    Bilinearly parameterized edge logits

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    eidx
        Scaffold edge indices of shape (2, n_edges)
    tau
        Gumbel-sigmoid temperature
    emb_dim
        Dimension of the bilinear parameterization
    """

    INIT_STD: float = 0.1

    def __init__(
        self,
        n_vars: int,
        n_particles: int,
        eidx: torch.LongTensor,
        tau: float = 10.0,
        emb_dim: int = None,
    ) -> None:
        super().__init__(n_vars, n_particles, eidx, tau=tau, emb_dim=emb_dim)
        self.emb_dim = emb_dim or round(sqrt(self.n_vars))
        self.u = Parameter(torch.empty(self.n_particles, self.n_vars, self.emb_dim))
        self.v = Parameter(torch.empty(self.n_particles, self.n_vars, self.emb_dim))
        self.reset_parameters()

    @internal
    def reset_parameters(self) -> None:
        init.normal_(self.u, std=self.INIT_STD)
        init.normal_(self.v, std=self.INIT_STD)

    def compute_logit(self) -> torch.Tensor:
        i, j = self.idx[:2]
        if not self.frozen:
            for p in self.u.data:
                p.renorm_(2, 0, 1)
            for p in self.v.data:
                p.renorm_(2, 0, 1)
        return (
            (self.u[:, i] * self.v[:, j]).sum(dim=-1).type_as(self.u)
        )  # Revert dtype autocast

    def __getitem__(self, index) -> "Bilinear":
        result = super().__getitem__(index)
        index = non_unitary_index(index)
        copy_like(self.u[index], result.u)
        copy_like(self.v[index], result.v)
        return result

    def _decay_params(self) -> Generator[Parameter, None, None]:
        yield self.u
        yield self.v
        yield from super()._decay_params()


# -------------------------------- Graph prior ---------------------------------


class Prior(Module):
    r"""
    Compute unnormalized negative log prior probability of a scaffold graph

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """

    def __init__(self, n_vars: int, n_particles: int, **kwargs) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.n_particles = n_particles

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        Energy function (negative log probability) of a graph scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Energy of shape (n_particles,)
        """
        raise NotImplementedError  # pragma: no cover


class SparsePrior(Prior):
    r"""
    Prior that encourages sparsity

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """


class AcycPrior(Prior):
    r"""
    Prior that enforces acyclicity constraint

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """


class L1(SparsePrior):
    r"""
    L1 penalized log prior probability

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        L1 sparse energy function (negative log probability) of a graph scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Sparse energy of shape (n_particles,)
        """
        return scaffold.prob.sum(dim=1) / scaffold.n_vars**2


class ScaleFree(SparsePrior):
    r"""
    Scale-free penalized log prior probability

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        Scale-free energy function (negative log probability) of a graph
        scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Scale-free energy of shape (n_particles,)
        """
        n_particles = scaffold.n_particles
        n_vars = scaffold.n_vars
        prob = scaffold.prob
        idx = scaffold.idx
        out_degree = prob.new_zeros((n_particles, n_vars))
        out_degree.scatter_add_(1, idx[0].unsqueeze(0).expand(n_particles, -1), prob)
        return out_degree.log1p().sum(dim=1) / (n_vars * log1p(n_vars))


class TrExp(AcycPrior):
    r"""
    Tr-Exp penalized log prior probability

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        Tr-Exp acyclic energy function (negative log probability) of a graph
        scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Tr-Exp acyclic energy of shape (n_particles,)
        """
        x = scaffold.adj.to_dense().permute(2, 0, 1)
        x = x / scaffold.n_vars  # Bound Tr-exp - n to e - 1
        c = scaffold.complete_adj.to_dense().permute(2, 0, 1)
        c = c / scaffold.n_vars
        enrg = multi_trace(x.matrix_exp()) - scaffold.n_vars  # (n_particles,)
        ceil = multi_trace(c.matrix_exp()) - scaffold.n_vars
        return enrg / ceil


class SpecNorm(AcycPrior):
    r"""
    Spectral norm penalized log prior probability

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    n_iter
        Number of power iterations
    """

    def __init__(self, n_vars: int, n_particles: int, n_iter: int = 5) -> None:
        super().__init__(n_vars, n_particles, n_iter=n_iter)
        self.n_iter = n_iter
        self.register_buffer("u1", torch.empty(self.n_vars, self.n_particles))
        self.register_buffer("v1", torch.empty(self.n_vars, self.n_particles))
        self.register_buffer("u2", torch.empty(self.n_vars, 1))
        self.register_buffer("v2", torch.empty(self.n_vars, 1))
        self.reset_parameters()

    @internal
    def reset_parameters(self) -> None:
        init.normal_(self.u1)
        init.normal_(self.v1)
        init.normal_(self.u2)
        init.normal_(self.v2)
        F.normalize(self.u1, dim=0, out=self.u1)
        F.normalize(self.v1, dim=0, out=self.v1)
        F.normalize(self.u2, dim=0, out=self.u2)
        F.normalize(self.v2, dim=0, out=self.v2)
        self.fresh = True
        self.limit = None

    @staticmethod
    def mv(
        idx: torch.Tensor,
        val: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Sparse matrix-vector product with particles in the last dimension

        Parameters
        ----------
        idx
            Index of the sparse matrix (2, n_edges)
        val
            Values of the sparse matrix (n_edges, n_particles)
        vec
            Vector of shape (n_vars, n_particles)

        Returns
        -------
        Matrix-vector product of shape (n_vars, n_particles)
        """
        i, j = idx
        res = torch.zeros_like(vec)
        return res.scatter_add_(0, i.unsqueeze(1).expand_as(val), val * vec[j])

    @staticmethod
    def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Vector dot product with particles in the last dimension

        Parameters
        ----------
        x
            Vector of shape (n_vars, n_particles)
        y
            Vector of shape (n_vars, n_particles)

        Returns
        -------
        Dot product of shape (n_particles,)
        """
        return (x * y).sum(dim=0)

    @torch.no_grad()
    def power_iteration(
        self,
        idx: torch.Tensor,
        val: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        n_iter: int,
    ) -> torch.Tensor:
        val = val.detach()
        for _ in range(n_iter):
            F.normalize(self.mv(idx.flip(0), val, u) + EPS * u.sum(dim=0), dim=0, out=u)
            F.normalize(self.mv(idx, val, v) + EPS * v.sum(dim=0), dim=0, out=v)
        self.fresh = False

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        Spectral norm acyclic energy function (negative log probability) of a
        graph scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Spectral norm acyclic energy of shape (n_particles,)
        """
        idx = scaffold.idx[:2]
        val = scaffold.prob.t()  # (n_edges, n_particles)
        one = val.new_ones(val.size(0), 1)
        if self.training:
            n_iter = self.n_iter * 5 if self.fresh else self.n_iter
            self.power_iteration(idx, val, self.u1, self.v1, n_iter)
            self.power_iteration(idx, one, self.u2, self.v2, n_iter)
        u1, v1 = self.u1.clone(), self.v1.clone()
        u2, v2 = self.u2.clone(), self.v2.clone()
        self.limit = (
            self.compute_limit(scaffold) if self.limit is None else self.limit
        )  # NOTE: Cached limit would be incorrect if scaffold changes
        enrg = (
            self.dot(u1, self.mv(idx, val, v1)) + EPS * u1.sum(dim=0) * v1.sum(dim=0)
        ) / (self.dot(u1, v1) * self.n_vars) - self.limit
        ceil = (
            self.dot(u2, self.mv(idx, one, v2)) + EPS * u2.sum(dim=0) * v2.sum(dim=0)
        ) / (self.dot(u2, v2) * self.n_vars) - self.limit
        return enrg.clamp(min=0.0) / ceil.clamp(min=EPS)  # (n_particles,)

    def compute_limit(self, scaffold: Scaffold) -> float:
        r"""
        Detection limit on the complete DAG
        """
        complete = scaffold.complete_adj.to_dense().squeeze(-1).numpy(force=True)
        complete_dag = np.triu(complete + complete.T, k=1).clip(max=1.0)
        if complete_dag.sum() == 0:
            return 0.0
        max_eig = eigs(
            complete_dag,
            k=1,
            which="LR",
            v0=np.ones(self.n_vars),
            return_eigenvectors=False,
        )[0]
        return max_eig.real.item() / self.n_vars

    @internal
    def get_extra_state(self) -> dict[str, Any]:
        return {"fresh": self.fresh, **super().get_extra_state()}

    @internal
    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.fresh = state.pop("fresh")
        super().set_extra_state(state)


class LogDet(AcycPrior):
    r"""
    Log-determinant penalized log prior probability

    Parameters
    ----------
    n_vars
        Number of variables in the graph
    n_particles
        Number of SVGD particles
    """

    def energy(self, scaffold: Scaffold) -> torch.Tensor:
        r"""
        Log-determinant acyclic energy function (negative log probability) of a
        graph scaffold

        Parameters
        ----------
        scaffold
            Graph scaffold

        Returns
        -------
        Log-determinant acyclic energy of shape (n_particles,)
        """
        x = scaffold.adj.to_dense().permute(2, 0, 1)
        x = x / scaffold.n_vars
        c = scaffold.complete_adj.to_dense().permute(2, 0, 1)
        c = c / scaffold.n_vars
        eye = torch.eye(scaffold.n_vars, dtype=x.dtype, device=x.device)
        enrg = -(eye - x).slogdet()[1]  # (n_particles,)
        ceil = -(eye - c).slogdet()[1]
        return enrg / ceil


# ----------------------------- Latent inference -------------------------------


class Latent(Module):
    r"""
    Interventional latent module

    Parameters
    ----------
    n_particles
        Number of SVGD particles
    latent_dim
        Dimensionality of the latent variable
    vmap
        Variable index mapping with the parent module
        :class:`~cascade.core.CausalNetwork`
    """

    def __init__(
        self,
        n_particles: int,
        latent_dim: int,
        vmap: torch.LongTensor,
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_particles = n_particles
        self.latent_dim = latent_dim
        self.register_buffer("vmap", vmap)
        self.register_buffer("prior_loc", torch.as_tensor(0.0))
        self.register_buffer("prior_scale", torch.as_tensor(1.0))

    def prior(self) -> D.Normal:
        return D.Normal(self.prior_loc, self.prior_scale)

    def forward(self, r: torch.Tensor) -> D.Normal:
        raise NotImplementedError  # pragma: no cover


class NilLatent(Latent):
    r"""
    Nil interventional latent module that always outputs the standard normal

    Parameters
    ----------
    n_particles
        Number of SVGD particles
    latent_dim
        Dimensionality of the latent variable
    vmap
        Variable index mapping with the parent module
        :class:`~cascade.core.CausalNetwork`
    """

    def forward(self, r: torch.Tensor) -> D.Normal:
        mu = r.new_zeros(self.n_particles, r.size(0), self.latent_dim)
        sigma = r.new_ones(self.n_particles, r.size(0), self.latent_dim)
        return D.Normal(mu, sigma)


class EmbLatent(Latent):
    r"""
    Intervention latent module encoding from fixed embeddings

    Parameters
    ----------
    n_particles
        Number of SVGD particles
    latent_dim
        Dimensionality of the latent variable
    vmap
        Variable index mapping with the parent module
        :class:`~cascade.core.CausalNetwork`
    emb
        Fixed embedding tensor
    """

    def __init__(
        self,
        n_particles: int,
        latent_dim: int,
        vmap: torch.LongTensor,
        emb: torch.Tensor = None,
    ) -> None:
        if emb is None:
            raise ValueError("Embedding tensor must be specified")
        super().__init__(n_particles, latent_dim, vmap, emb=emb)
        self.register_buffer("emb", emb.to(torch.get_default_dtype()))
        self.emb_dim = self.emb.size(1)
        self.pool = AttnPool(self.emb_dim)
        self.linear = MultiLinear(
            in_features=self.emb_dim,
            out_features=self.latent_dim * 2,
            multi_dims=(self.n_particles,),
        )

    def forward(self, r: torch.Tensor) -> D.Normal:
        vi, vj = self.vmap
        ptr = (
            r[..., vi].unsqueeze(-1) * self.emb[vj]
        )  # ([n_particles,] bs, n_vars, emb_dim)
        ptr = self.pool(ptr)  # ([n_particles,] bs, emb_dim)
        ptr = self.linear(ptr)  # (n_particles, bs, latent_dim * 2)
        mu = ptr[..., : self.latent_dim]
        sigma = F.softplus(ptr[..., -self.latent_dim :]) + EPS
        return D.Normal(mu, sigma)


class GCNLatent(Latent):
    r"""
    Intervention latent module encoding from a graph

    Parameters
    ----------
    n_particles
        Number of SVGD particles
    latent_dim
        Dimensionality of the latent variable
    vmap
        Variable index mapping with the parent module
        :class:`~cascade.core.CausalNetwork`
    eidx
        Graph edge index of shape (2, n_edges)
    ewt
        Graph edge weight of shape (n_edges,)
    emb_dim
        Dimensionality of the learnable node embedding
    n_layers
        Number of graph convolution layers
    """

    INIT_STD: float = 0.01

    def __init__(
        self,
        n_particles: int,
        latent_dim: int,
        vmap: torch.LongTensor,
        eidx: torch.LongTensor = None,
        ewt: torch.FloatTensor = None,
        emb_dim: int = None,
        n_layers: int = 1,
    ) -> None:
        if eidx is None:
            raise ValueError("Edge index tensor must be specified")
        if ewt is None:
            raise ValueError("Edge weight tensor must be specified")
        super().__init__(
            n_particles,
            latent_dim,
            vmap,
            eidx=eidx,
            ewt=ewt,
            emb_dim=emb_dim,
            n_layers=n_layers,
        )
        if (eidx < 0).any():
            raise ValueError("Edge index out of bounds")  # pragma: no cover
        self.register_buffer("eidx", eidx)
        self.register_buffer("ewt", ewt.clone())  # Will normalize in-place
        self.n_vars = (
            max(
                self.eidx.max() if self.eidx.numel() else -1,
                self.vmap[1].max() if self.vmap.numel() else -1,
            )
            + 1
        )
        self.emb_dim = emb_dim or round(sqrt(self.n_vars))
        self.emb = Parameter(torch.empty(self.n_vars, self.emb_dim))
        self.n_layers = n_layers

        self.pool = AttnPool(self.emb_dim)
        self.linear = MultiLinear(
            in_features=self.emb_dim,
            out_features=self.latent_dim * 2,
            multi_dims=(self.n_particles,),
        )
        self.normalize_edges()
        self.reset_parameters()

    def vertex_degrees(self, direction: str) -> torch.Tensor:
        if direction not in ("in", "out", "both"):
            raise ValueError("Unrecognized direction")
        degree = self.ewt.new_zeros(self.n_vars)
        if direction in ("in", "both"):
            degree.scatter_add_(0, self.eidx[1], self.ewt)
        if direction in ("out", "both"):
            degree.scatter_add_(0, self.eidx[0], self.ewt)
        if direction == "both":
            loop_mask = self.eidx[0] == self.eidx[1]
            degree.scatter_add_(0, self.eidx[0, loop_mask], -self.ewt[loop_mask])
        return degree

    def normalize_edges(self, method: str = "keepvar") -> None:
        if method not in ("in", "out", "sym", "keepvar"):
            raise ValueError("Unrecognized method")
        enorm = self.ewt
        if method in ("in", "keepvar", "sym"):
            in_degrees = self.vertex_degrees("in")
            in_norm = in_degrees.pow(-1 if method == "in" else -0.5)
            in_norm[in_norm.isinf()] = 0
            enorm = enorm * in_norm[self.eidx[1]]
        if method in ("out", "sym"):
            out_degrees = self.vertex_degrees("out")
            out_norm = out_degrees.pow(-1 if method == "out" else -0.5)
            out_norm[out_norm.isinf()] = 0
            enorm = enorm * out_norm[self.eidx[0]]
        self.ewt.copy_(enorm)

    @internal
    def reset_parameters(self) -> None:
        init.normal_(self.emb, std=self.INIT_STD)

    def forward(self, r: torch.Tensor) -> D.Normal:
        sidx, tidx = self.eidx
        emb = self.emb
        if not self.frozen:
            emb.data.renorm_(2, 0, 1)
        for _ in range(self.n_layers):
            message = emb[sidx] * self.ewt.unsqueeze(1)
            emb = torch.zeros_like(emb)
            tidx = tidx.unsqueeze(1).expand_as(message)
            emb.scatter_add_(0, tidx, message)
            emb = emb.renorm(2, 0, 1)

        vi, vj = self.vmap
        ptr = r[:, vi].unsqueeze(-1) * emb[vj]  # (bs, n_vars, emb_dim)
        ptr = self.pool(ptr)  # (bs, emb_dim)

        ptr = self.linear(ptr)
        mu = ptr[..., : self.latent_dim]
        sigma = F.softplus(ptr[..., -self.latent_dim :]) + EPS
        return D.Normal(mu, sigma)

    def _decay_params(self) -> Generator[Parameter, None, None]:
        yield self.emb
        yield from super()._decay_params()


# ----------------------------- Causal likelihood ------------------------------


class Likelihood(Module):
    r"""
    Abstract class for causal distributions

    Parameters
    ----------
    n_vars
        Number of variables
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.n_vars = n_vars

    def set_empirical(self, AnnData: AnnData) -> None:
        raise NotImplementedError  # pragma: no cover

    def tone(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    def forward(
        self,
        mean: torch.Tensor,
        disp: torch.Tensor,
        l: torch.Tensor,
        oidx: torch.LongTensor | None = None,
    ) -> D.Distribution:
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def get_mean(est: D.Distribution) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def get_disp(est: D.Distribution) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover

    def log_prior(self, est: D.Distribution) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover


class Normal(Likelihood):
    r"""
    Normal causal distribution

    Parameters
    ----------
    n_vars
        Number of variables
    """

    PRIOR_RATE: float = 0.01

    def __init__(self, n_vars: int) -> None:
        super().__init__(n_vars)
        self.register_buffer(
            "prior_shape", torch.ones(self.n_vars) * self.PRIOR_RATE + 1
        )
        self.register_buffer("prior_rate", torch.ones(self.n_vars) * self.PRIOR_RATE)
        self.batch_norm = BatchNorm1d(self.n_vars, eps=1, affine=False)

    def set_empirical(self, adata: AnnData) -> None:
        X = _get_X(adata)
        std = torch.as_tensor(
            np.sqrt(X.power(2).mean(axis=0).A1 - np.square(X.mean(axis=0).A1))
            if issparse(X)
            else X.std(axis=0)
        )
        self.prior_shape.copy_(std * self.prior_rate + 1)  # Mode at std

    def tone(self, x: torch.Tensor, l: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:  # (bs, n_vars)
            return self.batch_norm(x)
        # (n_particles, bs, n_vars)
        return self.batch_norm(x.permute(1, 2, 0)).permute(2, 0, 1)

    def forward(
        self,
        mean: torch.Tensor,
        disp: torch.Tensor,
        l: torch.Tensor,
        oidx: torch.LongTensor | None = None,
    ) -> D.Normal:
        return D.Normal(mean, F.softplus(disp) + EPS)

    @staticmethod
    def get_mean(est: D.Normal) -> torch.Tensor:
        return est.loc

    @staticmethod
    def get_disp(est: D.Normal) -> torch.Tensor:
        return est.scale

    def log_prior(self, est: D.Normal) -> torch.Tensor:
        return D.Gamma(self.prior_shape, self.prior_rate).log_prob(self.get_disp(est))


class NegBin(Likelihood):
    r"""
    Negative binomial causal distribution

    Parameters
    ----------
    n_vars
        Number of variables
    """

    NORM_TARGET: float = 1e4
    CAP_RATE: float = 0.75
    PRIOR_RATE: float = 0.05

    def __init__(self, n_vars: int) -> None:
        super().__init__(n_vars)
        self.theta_coef: float = None
        self.theta_intercept: float = None
        self.register_buffer("log_cap", torch.zeros(self.n_vars))
        self.register_buffer("prior_rate", torch.as_tensor(self.PRIOR_RATE))

    def set_empirical(self, adata: AnnData) -> None:
        X = _get_X(adata)
        size = _get_size(adata)
        if size.size == 0:
            raise ValueError("Size not configured")
        cap = (X / size).max(axis=0)
        cap = torch.as_tensor(cap.toarray().ravel() if issparse(cap) else cap)
        self.log_cap.copy_(self.CAP_RATE * cap.clamp(min=EPS, max=1.0).log())
        bins = pd.qcut(size.ravel(), 5)  # Group cells by size
        mean_list, theta_list = [], []
        for i in bins.categories:
            X_ = X[bins == i]
            if issparse(X_):
                mean = X_.mean(axis=0).A1
                var = X_.power(2).mean(axis=0).A1 - np.square(mean)
            else:
                mean = X_.mean(axis=0)
                var = X_.var(axis=0)
            mean_list.append(mean)
            theta_list.append(
                np.clip(np.square(mean) / (var - mean + EPS), a_min=0, a_max=200)
            )
        mean = np.concatenate(mean_list)
        theta = np.concatenate(theta_list)
        df = pd.DataFrame({"mean": mean, "mar_theta": theta})
        df["log1p_mean"] = np.log1p(df["mean"])
        df["bin"] = pd.qcut(df["log1p_mean"], 5)  # Group genes by expression
        df["res_theta"] = df.groupby("bin", observed=True)[["mar_theta"]].transform(
            lambda x: x.quantile(0.9)
        )  # Use smaller marginal variances as residual variance estimates
        lm = LinearRegression().fit(df[["log1p_mean"]], df["res_theta"])
        self.theta_coef = lm.coef_[0].item()
        self.theta_intercept = lm.intercept_.item()
        logger.info(f"Using theta coefficient = {self.theta_coef:.3f}")
        logger.info(f"Using theta intercept = {self.theta_intercept:.3f}")

    def tone(self, x: torch.Tensor, l: torch.Tensor | None = None) -> torch.Tensor:
        if l is None:
            l = x.sum(dim=-1, keepdim=True)
            norm_target = round(
                self.NORM_TARGET * x.size(-1) / 2e4
            )  # Roughly 20k genes -> NORM_TARGET
        else:
            norm_target = self.NORM_TARGET
        if self.training:
            sample_target = round(l.quantile(0.5).item())
            x_pad = torch.cat([x, l - x.sum(dim=-1, keepdim=True)], dim=-1)
            logits = x_pad.log() - l.log()
            x_samp = D.Multinomial(
                sample_target, logits=logits, validate_args=False
            ).sample()[
                ..., :-1
            ]  # Suppress occasional rounding errors
            x_norm = x_samp * (norm_target / sample_target)
        else:
            x_norm = x * (norm_target / l)
        return x_norm.log1p()

    def forward(
        self,
        mean: torch.Tensor,
        disp: torch.Tensor,
        l: torch.Tensor,
        oidx: torch.LongTensor | None = None,
    ) -> D.NegativeBinomial:
        log_cap = self.log_cap if oidx is None else self.log_cap[oidx]
        log_mu = l.log() - F.softplus(-mean) + log_cap
        theta = disp.exp() + 1
        return D.NegativeBinomial(theta, logits=log_mu - theta.log())

    @staticmethod
    def get_mean(est: D.NegativeBinomial) -> torch.Tensor:
        return est.logits.exp() * est.total_count

    @staticmethod
    def get_disp(est: D.NegativeBinomial) -> torch.Tensor:
        return est.total_count

    def log_prior(self, est: D.NegativeBinomial) -> torch.Tensor:
        mean = self.get_mean(est).detach()
        prior_theta = (self.theta_coef * mean.log1p() + self.theta_intercept).clamp(
            min=1.0
        )
        return D.Gamma(
            self.prior_rate * prior_theta + 1,
            self.prior_rate,
        ).log_prob(self.get_disp(est))

    @internal
    def get_extra_state(self) -> dict[str, Any]:
        return {
            "theta_coef": self.theta_coef,
            "theta_intercept": self.theta_intercept,
            **super().get_extra_state(),
        }

    @internal
    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.theta_coef = state.pop("theta_coef")
        self.theta_intercept = state.pop("theta_intercept")
        super().set_extra_state(state)


# -------------------------------- SVGD kernel ---------------------------------


class Kernel(Module):
    r"""
    Abstract class for kernels
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError  # pragma: no cover


class KroneckerDelta(Kernel):
    r"""
    Kronecker delta kernel
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.eye(
            x.size(0), y.size(0), dtype=x.dtype, device=x.device, requires_grad=True
        )


class RBF(Kernel):
    r"""
    Radial basis function kernel
    """

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:  # OK: Verify with the original SVGD paper
        x = x.flatten(start_dim=1)
        y = y.flatten(start_dim=1)
        return multi_rbf(x, y)


# ---------------------------- Intervention design -----------------------------


class IntervDesign(Module):
    r"""
    Intervention design module

    Parameters
    ----------
    n_vars
        Number of variables
    k
        Maximal combinatorial order to consider
    design_scale_bias
        Whether to optimize the intervention scale and bias
    mask
        Boolean mask that marks variables in the design candidate pool
    interv_scale
        Intervention scale tensor trained in the discover phase
    interv_bias
        Intervention bias tensor trained in the discover phase
    target_weight
        Variable weight when computing target deviation
    """

    def __init__(
        self,
        n_vars: int,
        k: int,
        design_scale_bias: bool,
        mask: torch.BoolTensor,
        interv_scale: torch.Tensor,
        interv_bias: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.k = k
        self.design_scale_bias = design_scale_bias
        elem = torch.cat(
            [
                torch.arange(n_vars)[mask],
                torch.as_tensor([n_vars]),
            ]
        )  # The last element indicates no intervention
        comb = torch.combinations(elem, k, with_replacement=True)
        comb = torch.stack(
            [
                row
                for row in comb.unbind()
                if (row[row < n_vars].unique(return_counts=True)[1] == 1).all()
            ]
        )  # All elements except for the last one must be unique

        self.register_buffer("mask", mask)
        self.register_buffer("comb", comb)
        self.register_buffer("interv_scale", interv_scale.detach())
        self.register_buffer("interv_bias", interv_bias.detach())
        self.register_buffer("target_weight", target_weight)
        self.logits = Parameter(torch.empty(self.comb.size(0)))
        self.design_scale = Parameter(torch.empty_like(self.interv_scale))
        self.design_bias = Parameter(torch.empty_like(self.interv_bias))
        self.reset_parameters()

    @internal
    def reset_parameters(self) -> None:
        init.zeros_(self.logits)
        self.design_scale.data.copy_(self.interv_scale)
        self.design_bias.data.copy_(self.interv_bias)

    def simplex2regime(self, simplex: torch.Tensor) -> torch.Tensor:
        bs = simplex.size(0)
        comb = self.comb.expand(bs, -1, -1)  # (bs, n_comb, k)
        simplex = simplex.unsqueeze(-1).expand_as(comb)  # (bs, n_comb, k)
        return (
            simplex.new_zeros(bs, self.n_vars + 1, self.k)
            .scatter_add_(1, comb, simplex)
            .sum(dim=-1)
        )[:, :-1]

    def rsample(self, bs: int) -> torch.Tensor:
        simplex = F.gumbel_softmax(self.logits.expand(bs, -1), hard=True)
        return self.simplex2regime(simplex)

    def loss(self, x_est: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
        return mean_squared_error(
            x_est, x_tgt.expand_as(x_est), dim=-1, weight=self.target_weight
        )  # (n_particles, bs)

    @property
    def scale(self) -> torch.Tensor:
        return self.design_scale if self.design_scale_bias else self.interv_scale

    @property
    def bias(self) -> torch.Tensor:
        return self.design_bias if self.design_scale_bias else self.interv_bias

    @cached_property
    def comb_lists(self) -> list[list[int]]:
        return [row[row < self.n_vars].tolist() for row in self.comb.unbind()]

    @internal
    def get_extra_state(self) -> dict[str, Any]:
        return {
            "n_vars": self.n_vars,
            "k": self.k,
            "design_scale_bias": self.design_scale_bias,
            **super().get_extra_state(),
        }

    @internal
    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.n_vars = state.pop("n_vars")
        self.k = state.pop("k")
        self.design_scale_bias = state.pop("design_scale_bias")
        super().set_extra_state(state)

    def save(self, fname: os.PathLike) -> None:
        r"""
        Save the design module to file

        Parameters
        ----------
        fname
            Path to save the design module (.pt)
        """
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "__version__": __version__,
                "state_dict": self.state_dict(),
            },
            fname,
        )

    @classmethod
    def load(cls, fname: os.PathLike) -> "IntervDesign":
        r"""
        Load design module from file

        Parameters
        ----------
        fname
            Path to load the design module (.pt)

        Returns
        -------
        Loaded design module
        """
        loaded = torch.load(fname, weights_only=True)
        version = loaded.pop("__version__", "unknown")
        if version != __version__:
            logger.warning(  # pragma: no cover
                "Loaded module version {} differs from current version {}.",
                version,
                __version__,
            )
        state_dict = loaded.pop("state_dict")
        extra_state = state_dict["_extra_state"]
        mod = cls(
            extra_state["n_vars"],
            extra_state["k"],
            extra_state["design_scale_bias"],
            state_dict["mask"],
            state_dict["interv_scale"],
            state_dict["interv_bias"],
            state_dict["target_weight"],
        )
        mod.load_state_dict(state_dict)
        return mod
