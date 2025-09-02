import argparse
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import cpa  # type: ignore
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
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--info", type=Path, default=None)
    design = subparsers.add_parser("design")
    design.add_argument("--input-data", type=Path, required=True)
    design.add_argument("--input-model", type=Path, required=True)
    design.add_argument("--input-target", type=Path, required=True)
    design.add_argument("--pool", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--design-size", type=int, required=True)
    design.add_argument("-k", type=int, default=30)
    design.add_argument("--n-neighbors", type=int, default=30)
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--info", type=Path, required=True)
    return parser.parse_args()


def interv2targets(x):
    return set(x.split(",")) - {""}


def interv2condition(x):
    if x == "":
        return "ctrl"
    x = x.split(",")
    if len(x) == 1:
        x.append("ctrl")
    return "+".join(sorted(x))


def condition2dosage(x):
    return "+".join("1.0" for _ in x.split("+"))


def train(args):
    adata_train = ad.read_h5ad(args.input_train)
    if issparse(adata_train.X):
        adata_train.X = csr_matrix(adata_train.X)
    adata_test = ad.read_h5ad(args.input_test)
    if issparse(adata_test.X):
        adata_test.X = csr_matrix(adata_test.X)
    pert_emb_df = pd.read_csv(args.pert_emb, index_col=0)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time()

    adata_train.obs["split"] = np.random.choice(
        ["train", "valid"], size=adata_train.n_obs, p=[0.85, 0.15]
    )
    adata_test.obs["split"] = "ood"
    adata_combined = ad.concat([adata_train, adata_test])
    adata_combined.obs["condition"] = adata_combined.obs[args.interv_key].map(
        interv2condition
    )
    adata_combined.obs["dosage"] = adata_combined.obs["condition"].map(condition2dosage)

    cpa.CPA.setup_anndata(
        adata_combined,
        perturbation_key="condition",
        control_group="ctrl",
        dosage_key="dosage",
        is_count_data=False,
        max_comb_len=2,
    )

    all_perts = sorted(
        cpa.CPA.pert_encoder.keys(), key=lambda k: cpa.CPA.pert_encoder[k]
    )
    pert_emb = torch.nn.Embedding(len(all_perts), pert_emb_df.shape[1], _freeze=True)
    pert_emb.weight.data.copy_(
        torch.as_tensor(pert_emb_df.reindex(all_perts).fillna(0).to_numpy())
    )

    model_params = {
        "drug_embeddings": pert_emb,
        "n_latent": 32,
        "recon_loss": "nb",
        "doser_type": "linear",
        "n_hidden_encoder": 256,
        "n_layers_encoder": 4,
        "n_hidden_decoder": 256,
        "n_layers_decoder": 2,
        "use_batch_norm_encoder": True,
        "use_layer_norm_encoder": False,
        "use_batch_norm_decoder": False,
        "use_layer_norm_decoder": False,
        "dropout_rate_encoder": 0.2,
        "dropout_rate_decoder": 0.0,
        "variational": False,
        "seed": args.seed,
    }
    trainer_params = {
        "n_epochs_kl_warmup": None,
        "n_epochs_adv_warmup": 50,
        "n_epochs_mixup_warmup": 10,
        "n_epochs_pretrain_ae": 10,
        "mixup_alpha": 0.1,
        "lr": 0.0001,
        "wd": 3.2170178270865573e-06,
        "adv_steps": 3,
        "reg_adv": 10.0,
        "pen_adv": 20.0,
        "adv_lr": 0.0001,
        "adv_wd": 7.051355554517135e-06,
        "n_layers_adv": 2,
        "n_hidden_adv": 128,
        "use_batch_norm_adv": True,
        "use_layer_norm_adv": False,
        "dropout_rate_adv": 0.3,
        "step_size_lr": 25,
        "do_clip_grad": False,
        "adv_loss": "cce",
        "gradient_clip_value": 5.0,
    }
    model = cpa.CPA(
        adata=adata_combined,
        split_key="split",
        train_split="train",
        valid_split="valid",
        test_split="ood",
        **model_params,
    )
    model.train(
        max_epochs=2000,
        use_gpu=True,
        batch_size=2048,
        plan_kwargs=trainer_params,
        early_stopping_patience=5,
        check_val_every_n_epoch=5,
        save_path=args.output,
    )
    elapsed_time = time() - start_time

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
    adata_test.obs["dosage"] = adata_test.obs["condition"].map(condition2dosage)
    adata_test.obs["split"] = "ood"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start_time = time()

    cpa.CPA.setup_anndata(
        adata_test,
        perturbation_key="condition",
        control_group="ctrl",
        dosage_key="dosage",
        is_count_data=False,
        max_comb_len=2,
    )
    model = cpa.CPA.load(dir_path=args.input_model, adata=adata_test, use_gpu=True)
    model.predict(adata_test, batch_size=2048)
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    adata_test.X = adata_test.obsm.pop("CPA_pred")
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


def design(args):  # CPA does not work with conditions missing from `adata_combined`
    source = ad.read_h5ad(args.input_data)
    if issparse(source.X):
        source.X = csr_matrix(source.X)
    target = ad.read_h5ad(args.input_target)
    if issparse(target.X):
        target.X = target.X.toarray()
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()

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
    source.obs["dosage"] = source.obs["condition"].map(condition2dosage)
    source.obs["split"] = "ood"

    cpa.CPA.setup_anndata(
        source,
        perturbation_key="condition",
        control_group="ctrl",
        dosage_key="dosage",
        is_count_data=False,
        max_comb_len=2,
    )
    model = cpa.CPA.load(dir_path=args.input_model, adata=source, use_gpu=True)
    model.predict(source, batch_size=2048)
    neighbor = NearestNeighbors(n_neighbors=args.n_neighbors).fit(
        source.obsm.pop("CPA_pred")
    )
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
