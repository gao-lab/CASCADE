import argparse
import copy
import json
import shutil
import warnings
from itertools import combinations
from pathlib import Path
from sys import argv
from time import time

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from gears import PertData  # type: ignore
from gears.utils import create_cell_graph_dataset_for_prediction  # type: ignore
from scgpt.loss import masked_mse_loss, masked_relative_error  # type: ignore
from scgpt.model import TransformerGenerator  # type: ignore
from scgpt.tokenizer.gene_tokenizer import GeneVocab  # type: ignore
from scgpt.utils import map_raw_id_to_vocab_id, set_seed  # type: ignore
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch_geometric.loader import DataLoader  # type: ignore

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

set_seed(42)

pretrained_model = "/opt/scGPT/scGPT_human"

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 2

n_hvg = 0  # number of highly variable genes
include_zero_gene = "all"  # include zero expr genes in training input,
# "all", "batch-wise", "row-wise", or False
max_seq_len = 1536

# settings for training
MLM = True  # whether to use masked language modeling, currently it is always on.
CLS = False  # celltype classification objective
CCE = False  # Contrastive cell embedding objective
MVC = False  # Masked value prediction for cell embedding
ECS = False  # Elastic cell similarity objective
cell_emb_style = "cls"
mvc_decoder_style = "inner product, detach"
amp = True
load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 64
eval_batch_size = 64
schedule_interval = 1
early_stop = 5

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0.2  # dropout probability
use_fast_transformer = True  # whether to use fast transformer
pool_size = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()


# logging
log_interval = 100


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--go-path", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--data-path", type=Path, required=True)
    train.add_argument("--interv-key", type=str, required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--info", type=Path, default=None)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--input-data", type=Path, required=True)
    predict.add_argument("--input-model", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--data-path", type=Path, required=True)
    predict.add_argument("--interv-key", type=str, required=True)
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--info", type=Path, default=None)
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


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    n_genes: int,
    gene_ids,
    scaler,
    optimizer,
    scheduler,
) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
            )
            output_values = output_dict["mlm_output"]

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
        scaler.step(optimizer)
        scaler.update()

        # torch.cuda.empty_cache()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            # ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time()


def evaluate(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, n_genes: int, gene_ids
) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0

    with torch.no_grad():
        for batch, batch_data in enumerate(val_loader):
            batch_size = len(batch_data.y)
            batch_data.to(device)
            x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, n_genes)
            pert_flags = x[:, 1].long().view(batch_size, n_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            if include_zero_gene in ["all", "batch-wise"]:
                if include_zero_gene == "all":
                    input_gene_ids = torch.arange(n_genes, device=device)
                else:  # when batch-wise
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # sample input_gene_id
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                        :max_seq_len
                    ]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # src_key_padding_mask = mapped_input_gene_ids.eq(vocab[pad_token])
                src_key_padding_mask = torch.zeros_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                    do_sample=True,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool, device=input_values.device
                )
                loss = criterion(output_values, target_values, masked_positions)
            total_loss += loss.item()
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item()
    return total_loss / len(val_loader), total_error / len(val_loader)


def train(args):
    epochs = args.epochs
    best_val_loss = float("inf")
    best_model = None
    patience = 0
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

    model_dir = Path(pretrained_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file) as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    n_genes = len(genes)

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )
    if load_param_prefixs is not None and pretrained_model is not None:
        # only load params that start with the prefix
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if any([k.startswith(prefix) for prefix in load_param_prefixs])
        }
        for k, v in pretrained_dict.items():
            print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    elif pretrained_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            print(f"Loading all model params from {model_file}")
        except Exception:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time()
        train_loader = pert_data.dataloader["train_loader"]
        valid_loader = pert_data.dataloader["val_loader"]

        train_epoch(
            model, train_loader, epoch, n_genes, gene_ids, scaler, optimizer, scheduler
        )

        val_loss, val_mre = evaluate(model, valid_loader, n_genes, gene_ids)
        elapsed = time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} |"
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            print(f"Best model with score {best_val_loss:5.4f}")
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                print(f"Early stop at epoch {epoch}")
                break

        scheduler.step()

    elapsed_time = time() - start_time
    output_path = Path(args.output)
    # if output_path.suffix == "":  # 检查路径是否是目录
    #     output_path = output_path / "model.pt"  # 将文件名拼接上
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), output_path)

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
    pool_size = 300
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

    model_dir = Path(pretrained_model)
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file) as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    ntokens = len(vocab)  # size of vocabulary

    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )

    model.load_state_dict(torch.load(Path(args.input_model)))
    model.eval()
    model.to(device)

    pert_set = set(genes)
    start_time = time()
    tasks = [task for task in ctfact.index.str.split(",") if not (set(task) - pert_set)]

    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()

    with torch.no_grad():
        results_pred = {}
        for pert in tasks:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    elapsed_time = time() - start_time
    result = pd.DataFrame.from_dict(
        results_pred, orient="index", columns=pert_data.adata.var_names
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
    pool_size = 300
    source = ad.read_h5ad(args.input_data)
    if issparse(source.X):
        source.X = csr_matrix(source.X)
    if isinstance(source.X, np.ndarray):
        source.X = sp.csr_matrix(source.X)
    target = ad.read_h5ad(args.input_target)
    if issparse(target.X):
        target.X = csr_matrix(target.X)
    if isinstance(target.X, np.ndarray):
        target.X = sp.csr_matrix(target.X)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist()

    pert_data = PertData(args.data_path)
    pert_data.load(data_path=(args.data_path / "custom").as_posix())
    pert_data.prepare_split(split="no_test", seed=args.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_dir = Path(pretrained_model)
    model_config_file = model_dir / "args.json"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    pert_data.adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
    ]
    genes = pert_data.adata.var["gene_name"].tolist()

    with open(model_config_file) as f:
        model_configs = json.load(f)
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )
    ntokens = len(vocab)  # size of vocabulary

    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        do_mvc=MVC,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        use_fast_transformer=use_fast_transformer,
    )

    model.load_state_dict(torch.load(Path(args.input_model)))
    model.eval()
    model.to(device)

    pert_set = set(genes)
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

    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()

    with torch.no_grad():
        results_pred = {}
        for pert in tasks:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []
            for batch_data in loader:
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    result = pd.DataFrame.from_dict(
        results_pred, orient="index", columns=pert_data.adata.var_names
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
