#!/usr/bin/env python

r"""
Adapted from https://github.com/Genentech/dcdfg/blob/main/run_perturbseq_linear.py
"""

import argparse
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from dcdfg.callback import (  # type: ignore
    AugLagrangianCallback,
    ConditionalEarlyStopping,
    CustomProgressBar,
)
from dcdfg.linear_baseline.model import LinearGaussianModel  # type: ignore
from dcdfg.lowrank_linear_baseline.model import (  # type: ignore
    LinearModuleGaussianModel,
)
from dcdfg.lowrank_mlp.model import MLPModuleGaussianModel  # type: ignore
from dcdfg.perturbseq_data import PerturbSeqDataset  # type: ignore
from pytorch_lightning.callbacks import EarlyStopping
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interv-key", type=str, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--info", type=Path, required=True)

    parser.add_argument(
        "--train-samples",
        type=int,
        default=0.8,
        help="Number of samples used for training (default is 80% of the total size)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        help="number of samples in a minibatch",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=600,
        help="number of meta gradient steps",
    )
    parser.add_argument(
        "--num-fine-epochs", type=int, default=50, help="number of meta gradient steps"
    )
    parser.add_argument("--num-modules", type=int, default=20, help="number of modules")
    # optimization
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate for optim"
    )
    parser.add_argument(
        "--reg-coeff",
        type=float,
        default=0.1,
        help="regularization coefficient (lambda)",
    )
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default="exp",
        help="technique for acyclicity constraint",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="linear|linearlr|mlplr",
    )
    parser.add_argument(
        "--poly", action="store_true", help="Polynomial on linear model"
    )
    parser.add_argument("--num-gpus", type=int, default=1)

    arg = parser.parse_args()

    # load data and make dataset
    adata = ad.read_h5ad(arg.input)
    adata.X = csr_matrix(adata.X)
    adata.obs["targets"] = adata.obs[arg.interv_key]
    adata.obs["regimes"] = np.unique(adata.obs["targets"], return_inverse=True)[1]
    arg.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = arg.output.parent / "adata.h5ad"
    adata.write(tmp_file, compression="gzip")

    train_dataset = PerturbSeqDataset(tmp_file)
    tmp_file.unlink()
    start_time = time()

    nb_nodes = train_dataset.dim
    np.random.seed(arg.random_seed)
    torch.manual_seed(arg.random_seed)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    if arg.model == "linear":
        # create model
        model = LinearGaussianModel(
            nb_nodes,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
            poly=arg.poly,
        )
    elif arg.model == "linearlr":
        model = LinearModuleGaussianModel(
            nb_nodes,
            arg.num_modules,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
        )
    elif arg.model == "mlplr":
        model = MLPModuleGaussianModel(
            nb_nodes,
            2,
            arg.num_modules,
            16,
            lr_init=arg.lr,
            reg_coeff=arg.reg_coeff,
            constraint_mode=arg.constraint_mode,
        )
    else:
        raise ValueError("couldn't find model")

    # Step 1: augmented lagrangian
    early_stop_1_callback = ConditionalEarlyStopping(
        monitor="Val/aug_lagrangian",
        min_delta=1e-4,
        patience=5,
        verbose=True,
        mode="min",
    )
    trainer = pl.Trainer(
        gpus=arg.num_gpus,
        max_epochs=arg.num_train_epochs,
        val_check_interval=1.0,
        callbacks=[AugLagrangianCallback(), early_stop_1_callback, CustomProgressBar()],
        default_root_dir=arg.output.parent,
    )
    trainer.fit(
        model,
        DataLoader(train_dataset, batch_size=arg.train_batch_size, num_workers=4),
        DataLoader(val_dataset, num_workers=8, batch_size=256),
    )

    # freeze and prune adjacency
    model.module.threshold()
    # WE NEED THIS BECAUSE IF it's exactly a DAG THE POWER ITERATIONS DOESN'T CONVERGE
    # TODO Just refactor and remove constraint at validation time
    model.module.constraint_mode = "exp"
    # remove dag constraints: we have a prediction problem now!
    model.gamma = 0.0
    model.mu = 0.0

    # Step 2:fine tune weights with frozen model
    early_stop_2_callback = EarlyStopping(
        monitor="Val/nll", min_delta=1e-6, patience=5, verbose=True, mode="min"
    )
    trainer_fine = pl.Trainer(
        gpus=arg.num_gpus,
        max_epochs=arg.num_fine_epochs,
        val_check_interval=1.0,
        callbacks=[early_stop_2_callback, CustomProgressBar()],
        default_root_dir=arg.output.parent,
    )
    trainer_fine.fit(
        model,
        DataLoader(train_dataset, batch_size=arg.train_batch_size),
        DataLoader(val_dataset, num_workers=2, batch_size=256),
    )

    # Step 3: save graph
    pred_adj = model.module.weight_mask.detach().cpu().numpy()
    # check integers
    assert np.equal(np.mod(pred_adj, 1), 0).all()
    digraph = nx.from_numpy_array(pred_adj, create_using=nx.DiGraph)
    nx.relabel_nodes(digraph, {i: v for i, v in enumerate(adata.var_names)}, copy=False)
    elapsed_time = time() - start_time

    nx.write_gml(digraph, arg.output)
    arg.info.parent.mkdir(parents=True, exist_ok=True)
    with arg.info.open("w") as f:
        yaml.dump(
            {
                "cmd": " ".join(argv),
                "args": vars(arg),
                "time": elapsed_time,
                "n_obs": adata.n_obs,
                "n_vars": adata.n_vars,
            },
            f,
        )
