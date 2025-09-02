import argparse
import shutil
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from gears import GEARS, PertData  # type: ignore
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

pretrained_model = "/opt/scFoundation/model/models/models.ckpt"


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--go-path", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--data-path", type=Path, required=True)
    train.add_argument("--interv-key", type=str, required=True)
    train.add_argument("--hidden-size", type=int, default=512)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--info", type=Path, default=None)
    train.add_argument("--model-type", type=str, default="maeautobin")
    train.add_argument("--bin-set", type=str, default="autobin_resolution_append")
    train.add_argument("--finetune-method", type=str, default="frozen")
    train.add_argument("--mode", type=str, default="v1")
    train.add_argument("--accumulation-steps", type=int, default=5)
    train.add_argument("--highres", type=int, default=0)
    train.add_argument("--lr", type=float, default=1e-3)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-data", type=Path, required=True)
    predict.add_argument("--input-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--data-path", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--info", type=Path, default=None)
    predict.add_argument("--hidden-size", type=int, default=512)
    predict.add_argument("--model-type", type=str, default=None)
    predict.add_argument("--bin-set", type=str, default=None)
    predict.add_argument("--finetune-method", type=str, default=None)
    predict.add_argument("--mode", type=str, default="v1")
    predict.add_argument("--accumulation-steps", type=int, default=1)
    predict.add_argument("--highres", type=int, default=0)

    design = subparsers.add_parser("design")
    design.add_argument("--input-data", type=Path, required=True)
    design.add_argument("--input-model", type=Path, required=True)
    design.add_argument("--input-target", type=Path, required=True)
    design.add_argument("--pool", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--data-path", type=Path, required=True)
    design.add_argument("--design-size", type=int, required=True)
    design.add_argument("-k", type=int, default=30)
    design.add_argument("--n-neighbors", type=int, default=1)
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--info", type=Path, required=True)
    design.add_argument("--hidden-size", type=int, default=512)
    design.add_argument("--model-type", type=str, default="maeautobin")
    design.add_argument("--bin-set", type=str, default="autobin_resolution_append")
    design.add_argument("--finetune-method", type=str, default="frozen")
    design.add_argument("--mode", type=str, default="v1")
    design.add_argument("--accumulation-steps", type=int, default=5)
    design.add_argument("--highres", type=int, default=0)
    return parser.parse_args()


def interv2condition(x):
    if x == "":
        return "ctrl"
    x = x.split(",")
    if len(x) == 1:
        x.append("ctrl")
    return "+".join(sorted(x))


def aggregate_obs(adata, by):
    by = adata.obs[by]
    agg_idx = pd.Index(by.unique())
    agg_sum = csr_matrix(
        (np.ones(adata.n_obs), (agg_idx.get_indexer(by), np.arange(adata.n_obs)))
    )
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))
    return ad.AnnData(
        X=agg_mean @ adata.X,
        obs=pd.DataFrame(index=agg_idx.astype(str)),
        var=adata.var,
    )


def train(args):
    adata = ad.read_h5ad(args.input)
    if issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    if isinstance(adata.X, np.ndarray):
        adata.X = sp.csr_matrix(adata.X)
    adata.obs["condition"] = adata.obs[args.interv_key].map(interv2condition)
    adata.var["gene_name"] = adata.var_names
    adata.obs["cell_type"] = "placeholder"
    adata = adata[
        adata.obs["condition"].value_counts().loc[adata.obs["condition"]] > 1
    ].copy()  # Conditions with single observations would cause DE error

    # adata.obs['total_count'] = np.sum(adata.X.toarray(), axis=1)
    if hasattr(adata.X, "toarray"):  # 确保它是 sparse matrix
        adata.obs["total_count"] = np.sum(adata.X.toarray(), axis=1)
    else:  # 如果是 numpy.ndarray，直接 sum
        adata.obs["total_count"] = np.sum(adata.X, axis=1)

    # adata.write_h5ad(args.input)

    if args.data_path.exists():
        shutil.rmtree(args.data_path)
    args.data_path.mkdir(parents=True)
    for file in (
        "gene2go_all.pkl",
        "essential_all_data_pert_genes.pkl",
        "go_essential_all",
    ):
        (args.data_path / file).symlink_to(
            (args.go_path / file).resolve(),
            target_is_directory=file == "go_essential_all",
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    start_time = time()
    pert_data = PertData(args.data_path)
    pert_data.new_data_process("custom", adata=adata)
    pert_data.prepare_split(split="no_test", seed=args.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    gears_model = GEARS(pert_data, device="cuda")
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        model_type=args.model_type,
        bin_set=args.bin_set,
        load_path=pretrained_model,
        finetune_method=args.finetune_method,
        accumulation_steps=args.accumulation_steps,
        mode=args.mode,
        highres=args.highres,
    )
    gears_model.train(epochs=args.epochs, lr=args.lr)
    elapsed_time = time() - start_time

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gears_model.save_model(args.output)
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
    ctfact = ad.read_h5ad(args.input_data)
    if issparse(ctfact.X):
        ctfact.X = csr_matrix(ctfact.X)
    if isinstance(ctfact.X, np.ndarray):
        ctfact.X = sp.csr_matrix(ctfact.X)
    ctfact = aggregate_obs(ctfact, args.interv_key).to_df()

    pert_data = PertData(args.data_path)
    pert_data.load(data_path=(args.data_path / "custom").as_posix())
    pert_data.prepare_split(split="no_test", seed=args.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gears_model = GEARS(pert_data, device="cuda")
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        model_type=args.model_type,
        bin_set=args.bin_set,
        load_path=pretrained_model,
        finetune_method=args.finetune_method,
        accumulation_steps=args.accumulation_steps,
        mode=args.mode,
        highres=args.highres,
    )
    gears_model.load_pretrained(args.input_model)
    pert_set = set(gears_model.pert_list)

    start_time = time()
    tasks = [task for task in ctfact.index.str.split(",") if not (set(task) - pert_set)]
    result = gears_model.predict(tasks)
    elapsed_time = time() - start_time

    result = pd.DataFrame.from_dict(
        result, orient="index", columns=pert_data.adata.var_names
    )
    result.index = result.index.map(lambda x: ",".join(sorted(set(x.split("_")))))
    assert (result.columns == ctfact.columns).all()
    assert result.index.isin(ctfact.index).all()
    ctfact.loc[result.index] = result  # Unpredictable conditions are left unchanged
    ctfact = ad.AnnData(ctfact)
    ctfact.obs[args.interv_key] = ctfact.obs_names

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ctfact.write(args.output, compression="gzip")
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
    if isinstance(source.X, np.ndarray):
        source.X = sp.csr_matrix(source.X)
    target = ad.read_h5ad(args.input_target)
    if issparse(target.X):
        target.X = target.X.toarray()
    if isinstance(target.X, np.ndarray):
        target.X = sp.csr_matrix(target.X)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()

    pert_data = PertData(args.data_path)
    pert_data.load(data_path=(args.data_path / "custom").as_posix())
    pert_data.prepare_split(split="no_test", seed=args.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gears_model = GEARS(pert_data, device="cuda")
    gears_model.model_initialize(
        hidden_size=args.hidden_size,
        model_type=args.model_type,
        bin_set=args.bin_set,
        load_path=pretrained_model,
        finetune_method=args.finetune_method,
        accumulation_steps=args.accumulation_steps,
        mode=args.mode,
        highres=args.highres,
    )
    gears_model.load_pretrained(args.input_model)
    pert_set = set(gears_model.pert_list)

    start_time = time()
    search_space = [
        ",".join(sorted(c))
        for s in range(args.design_size + 1)
        for c in combinations(pool, s)
    ]
    source_idx = np.random.choice(source.n_obs, len(search_space) * args.k)
    source = source[source_idx].copy()
    source.obs["design"] = np.repeat(search_space, args.k)
    source = aggregate_obs(source, "design").to_df()
    tasks = [task for task in source.index.str.split(",") if not (set(task) - pert_set)]
    result = gears_model.predict(tasks)

    result = pd.DataFrame.from_dict(
        result, orient="index", columns=pert_data.adata.var_names
    )
    result.index = result.index.map(lambda x: ",".join(sorted(set(x.split("_")))))
    assert (result.columns == source.columns).all()
    assert result.index.isin(source.index).all()
    source.loc[result.index] = result  # Unpredictable conditions are left unchanged

    neighbor = NearestNeighbors(n_neighbors=args.n_neighbors).fit(source.to_numpy())
    nni = neighbor.kneighbors(target.X, return_distance=False)
    votes = source.index[nni.ravel()].value_counts()
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
