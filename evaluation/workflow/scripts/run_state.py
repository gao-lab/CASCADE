import argparse
import subprocess
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import numpy as np
import toml
import yaml
from scipy.sparse import csr_matrix, issparse


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--input-train", type=Path, required=True)
    train.add_argument("--input-test", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--interv-key", type=str, required=True)
    train.add_argument("--cell-set-len", type=int, default=32)  # Table 3
    train.add_argument("--hidden-dim", type=int, default=128)  # Table 3
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--info", type=Path, default=None)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--data", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--info", type=Path, default=None)
    # design = subparsers.add_parser("design")
    # design.add_argument("--input-data", type=Path, required=True)
    # design.add_argument("--input-model", type=Path, required=True)
    # design.add_argument("--input-target", type=Path, required=True)
    # design.add_argument("--pert-emb", type=Path, required=True)
    # design.add_argument("--pool", type=Path, required=True)
    # design.add_argument("--output", type=Path, required=True)
    # design.add_argument("--design-size", type=int, required=True)
    # design.add_argument("-k", type=int, default=30)
    # design.add_argument("--n-neighbors", type=int, default=30)
    # design.add_argument("--seed", type=int, default=0)
    # design.add_argument("--info", type=Path, required=True)
    return parser.parse_args()


def get_emb(pert, emb):
    pert = sorted(set(pert.split("+")) - {"ctrl"})
    if len(pert):
        return emb.reindex(pert).fillna(0).mean().to_numpy()
    return np.zeros(emb.shape[1])


def train(args):
    adata_train = ad.read_h5ad(args.input_train)
    adata_test = ad.read_h5ad(args.input_test)
    cell_line = adata_train.obs["cell_line"].cat.categories
    assert len(cell_line) == 1
    cell_line = cell_line[0]
    test_perts = adata_test.obs[args.interv_key].unique().tolist()
    if "gemgroup" in adata_train.obs:
        batch = "gemgroup"
    else:
        batch = "batch"
        if "batch" not in adata_train.obs:
            adata_train.obs[batch] = "batch"
            adata_test.obs[batch] = "batch"
    if issparse(adata_train.X):
        adata_train.X = csr_matrix(adata_train.X)
    if issparse(adata_test.X):
        adata_test.X = csr_matrix(adata_test.X)
    adata_train.X = adata_train.X.astype(np.float32)
    adata_test.X = adata_test.X.astype(np.float32)

    # Write toml file
    fewshot = {
        "datasets": {
            "combined": str(args.output / "combined"),
        },
        "training": {
            "combined": "train",
        },
        "fewshot": {
            f"combined.{cell_line}": {
                "test": test_perts,
            }
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "fewshot.toml", "w") as f:
        toml.dump(fewshot, f)

    # Merge train & test and write into a cell_type directory
    combined = ad.concat([adata_train, adata_test], merge="first")
    (args.output / "combined").mkdir(exist_ok=True)
    combined.write(args.output / "combined" / f"{cell_line}.h5ad", compression="gzip")

    # Call state CLI to train model
    start_time = time()
    subprocess.run(
        [
            "state",
            "tx",
            "train",
            f"data.kwargs.toml_config_path='{args.output}/fewshot.toml'",
            "data.kwargs.embed_key=null",
            "data.kwargs.cell_type_key=cell_line",
            f"data.kwargs.batch_col={batch}",
            f"data.kwargs.pert_col={args.interv_key}",
            "data.kwargs.control_pert=''",
            "data.kwargs.perturbation_features_file=/opt/SE-600M/protein_embeddings.pt",
            f"training.train_seed={args.seed}",
            f"model.kwargs.cell_set_len={args.cell_set_len}",
            f"model.kwargs.hidden_dim={args.hidden_dim}",
            f"output_dir='{args.output}'",
            "use_wandb=false",
            "name=model",
            # f"training.max_steps=500",  # DEBUG
        ]
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
    start_time = time()
    subprocess.run(
        [
            "state",
            "tx",
            "predict",
            "--output_dir",
            args.model / "model",
            "--checkpoint",
            "final.ckpt",
        ]
    )
    elapsed_time = time() - start_time

    data = ad.read_h5ad(args.data)
    pred = ad.read_h5ad(args.model / "model" / "eval_final.ckpt" / "adata_pred.h5ad")
    pred.var = data.var.copy()  # Var including var_names would be emptied
    pred = pred[pred.obs[args.interv_key] != ""].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pred.write(args.output, compression="gzip")
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


# TODO
# def design(args):
#     source = ad.read_h5ad(args.input_data)
#     if issparse(source.X):
#         source.X = csr_matrix(source.X)
#     target = ad.read_h5ad(args.input_target)
#     if issparse(target.X):
#         target.X = target.X.toarray()
#     pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()
#     pert_emb = pd.read_csv(args.pert_emb, index_col=0)
#     model = biolord.Biolord.load(args.input_model)
#
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     start_time = time()
#
#     search_space = [
#         ",".join(sorted(c))
#         for s in range(args.design_size + 1)
#         for c in combinations(pool, s)
#     ]
#     source_idx = np.random.choice(source.n_obs, len(search_space) * args.k)
#     source = source[source_idx].copy()
#     source.obs["design"] = np.repeat(search_space, args.k)
#     source.obs["condition"] = source.obs["design"].map(interv2condition)
#     source.obsm["condition_emb"] = np.stack(
#         source.obs["condition"].astype(str).map(lambda x: get_emb(x, pert_emb))
#     )
#     biolord.Biolord.setup_anndata(
#         source,
#         ordered_attributes_keys="condition_emb",
#     )
#
#     adata_pred = []
#     for batch_idx in np.array_split(
#         np.arange(source.n_obs), np.ceil(source.n_obs / 2048)
#     ):  # Avoid VRAM overflow
#         adata_pred.append(
#             model.compute_prediction_adata(source[batch_idx], source[batch_idx], [])
#         )
#     adata_pred = ad.concat(adata_pred)
#     neighbor = NearestNeighbors(n_neighbors=args.n_neighbors).fit(adata_pred.X)
#     nni = neighbor.kneighbors(target.X, return_distance=False)
#     votes = source.obs["design"].iloc[nni.ravel()].value_counts()
#     outcast = [item for item in search_space if item not in votes.index]
#     outcast = pd.Series(0, index=outcast, name="count")
#     design = pd.concat([votes, outcast])
#     design = design.to_frame().rename(columns={"count": "votes"})
#     elapsed_time = time() - start_time
#
#     args.output.parent.mkdir(parents=True, exist_ok=True)
#     design.to_csv(args.output)
#     if args.info:
#         args.info.parent.mkdir(parents=True, exist_ok=True)
#         with args.info.open("w") as f:
#             yaml.dump(
#                 {
#                     "cmd": " ".join(argv),
#                     "args": vars(args),
#                     "time": elapsed_time,
#                 },
#                 f,
#             )


if __name__ == "__main__":
    args = parse_args()
    if args.subcommand == "train":
        train(args)
    elif args.subcommand == "predict":
        predict(args)
    else:  # args.subcommand == "design"
        raise RuntimeError("Not implemented yet")
        # design(args)
