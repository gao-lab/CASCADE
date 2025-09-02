import argparse
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import biolord  # type: ignore
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--input-train", type=Path, required=True)
    train.add_argument("--input-test", type=Path, required=True)
    train.add_argument("--pert-emb", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--interv-key", type=str, required=True)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--info", type=Path, default=None)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-data", type=Path, required=True)
    predict.add_argument("--input-model", type=Path, required=True)
    predict.add_argument("--pert-emb", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--info", type=Path, default=None)
    design = subparsers.add_parser("design")
    design.add_argument("--input-data", type=Path, required=True)
    design.add_argument("--input-model", type=Path, required=True)
    design.add_argument("--input-target", type=Path, required=True)
    design.add_argument("--pert-emb", type=Path, required=True)
    design.add_argument("--pool", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--design-size", type=int, required=True)
    design.add_argument("-k", type=int, default=30)
    design.add_argument("--n-neighbors", type=int, default=30)
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--info", type=Path, required=True)
    return parser.parse_args()


def interv2condition(x):
    if x == "":
        return "ctrl"
    x = x.split(",")
    if len(x) == 1:
        x.append("ctrl")
    return "+".join(sorted(x))


def get_emb(pert, emb):
    pert = sorted(set(pert.split("+")) - {"ctrl"})
    if len(pert):
        return emb.reindex(pert).fillna(0).mean().to_numpy()
    return np.zeros(emb.shape[1])


def train(args):
    adata_train = ad.read_h5ad(args.input_train)
    if issparse(adata_train.X):
        adata_train.X = csr_matrix(adata_train.X)
    adata_train.obs["condition"] = adata_train.obs[args.interv_key].map(
        interv2condition
    )
    pert_emb = pd.read_csv(args.pert_emb, index_col=0)

    adata_test = ad.read_h5ad(args.input_test)
    adata_test.obs["condition"] = adata_test.obs[args.interv_key].map(interv2condition)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time()

    adata_train.obsm["condition_emb"] = np.stack(
        adata_train.obs["condition"].astype(str).map(lambda x: get_emb(x, pert_emb))
    )
    biolord.Biolord.setup_anndata(
        adata_train,
        ordered_attributes_keys="condition_emb",
    )
    module_params = {
        "decoder_width": 1024,
        "decoder_depth": 4,
        "attribute_nn_width": 512,
        "attribute_nn_depth": 2,
        "n_latent_attribute_categorical": 4,
        "gene_likelihood": "normal",
        "reconstruction_penalty": 1e2,
        "unknown_attribute_penalty": 1e1,
        "unknown_attribute_noise_param": 1e-1,
        "attribute_dropout_rate": 0.1,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": 42,
    }
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": 1e-4,
        "latent_wd": 1e-4,
        "decoder_lr": 1e-4,
        "decoder_wd": 1e-4,
        "attribute_nn_lr": 1e-2,
        "attribute_nn_wd": 4e-8,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }
    model = biolord.Biolord(
        adata=adata_train,
        n_latent=32,
        model_name="benchmark",
        module_params=module_params,
        train_classifiers=False,
    )
    model.train(
        max_epochs=500,
        batch_size=512,
        plan_kwargs=trainer_params,
        early_stopping=True,
        early_stopping_patience=20,
        check_val_every_n_epoch=10,
        num_workers=1,
        enable_checkpointing=False,
    )
    elapsed_time = time() - start_time

    model.save(args.output, save_anndata=True)
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


def predict(args):
    adata_test = ad.read_h5ad(args.input_data)
    if issparse(adata_test.X):
        adata_test.X = csr_matrix(adata_test.X)
    adata_test.obs["condition"] = adata_test.obs[args.interv_key].map(interv2condition)
    pert_emb = pd.read_csv(args.pert_emb, index_col=0)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time()

    adata_test.obsm["condition_emb"] = np.stack(
        adata_test.obs["condition"].astype(str).map(lambda x: get_emb(x, pert_emb))
    )
    biolord.Biolord.setup_anndata(
        adata_test,
        ordered_attributes_keys="condition_emb",
    )
    model = biolord.Biolord.load(args.input_model)

    adata_pred = []
    for batch_idx in np.array_split(
        np.arange(adata_test.n_obs), np.ceil(adata_test.n_obs / 2048)
    ):  # Avoid VRAM overflow
        adata_pred.append(
            model.compute_prediction_adata(
                adata_test[batch_idx], adata_test[batch_idx], []
            )
        )
    adata_pred = ad.concat(adata_pred)
    adata_test.X = adata_pred.X.copy()
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata_test.write(args.output, compression="gzip")
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


def design(args):
    source = ad.read_h5ad(args.input_data)
    if issparse(source.X):
        source.X = csr_matrix(source.X)
    target = ad.read_h5ad(args.input_target)
    if issparse(target.X):
        target.X = target.X.toarray()
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()
    pert_emb = pd.read_csv(args.pert_emb, index_col=0)
    model = biolord.Biolord.load(args.input_model)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time()

    search_space = [
        ",".join(sorted(c))
        for s in range(args.design_size + 1)
        for c in combinations(pool, s)
    ]
    source_idx = np.random.choice(source.n_obs, len(search_space) * args.k)
    source = source[source_idx].copy()
    source.obs["design"] = np.repeat(search_space, args.k)
    source.obs["condition"] = source.obs["design"].map(interv2condition)
    source.obsm["condition_emb"] = np.stack(
        source.obs["condition"].astype(str).map(lambda x: get_emb(x, pert_emb))
    )
    biolord.Biolord.setup_anndata(
        source,
        ordered_attributes_keys="condition_emb",
    )

    adata_pred = []
    for batch_idx in np.array_split(
        np.arange(source.n_obs), np.ceil(source.n_obs / 2048)
    ):  # Avoid VRAM overflow
        adata_pred.append(
            model.compute_prediction_adata(source[batch_idx], source[batch_idx], [])
        )
    adata_pred = ad.concat(adata_pred)
    neighbor = NearestNeighbors(n_neighbors=args.n_neighbors).fit(adata_pred.X)
    nni = neighbor.kneighbors(target.X, return_distance=False)
    votes = source.obs["design"].iloc[nni.ravel()].value_counts()
    outcast = [item for item in search_space if item not in votes.index]
    outcast = pd.Series(0, index=outcast, name="count")
    design = pd.concat([votes, outcast])
    design = design.to_frame().rename(columns={"count": "votes"})
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output)
    if args.info:
        args.info.parent.mkdir(parents=True, exist_ok=True)
        with args.info.open("w") as f:
            yaml.dump(
                {
                    "cmd": " ".join(argv),
                    "args": vars(args),
                    "time": elapsed_time,
                },
                f,
            )


if __name__ == "__main__":
    args = parse_args()
    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "predict":
        predict(args)
    else:  # args.subcommand == "design"
        design(args)
