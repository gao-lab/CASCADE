r"""
CLI entry point
"""

from argparse import ArgumentParser, Namespace, _ArgumentGroup
from collections.abc import Iterable
from functools import wraps
from pathlib import Path
from random import randint, sample
from sys import argv
from time import sleep, time


# fmt: off
def add_configure_args(group: _ArgumentGroup) -> None:
    group.add_argument(
        "--interv-key", type=str, default=None,
        help="Interventional target key in adata.obs",
    )
    group.add_argument(
        "--use-covariate", type=str, default=None,
        help="Covariate key in adata.obs",
    )
    group.add_argument(
        "--use-size", type=str, default=None,
        help="Size key in adata.obs",
    )
    group.add_argument(
        "--use-weight", type=str, default=None,
        help="Weight key in adata.obs",
    )
    group.add_argument(
        "--use-layer", type=str, default=None,
        help="Data key in adata.layers",
    )


def add_construct_args(group: _ArgumentGroup) -> None:
    group.add_argument(
        "--n-particles", type=int, default=4,
        help="Number of SVGD particles",
    )
    group.add_argument(
        "--n-layers", type=int, default=1,
        help="Number of MLP layers in the structural equations",
    )
    group.add_argument(
        "--hidden-dim", type=int, default=16,
        help="MLP hidden layer dimension in the structural equations",
    )
    group.add_argument(
        "--latent-dim", type=int, default=16,
        help="Dimension of the latent variable",
    )
    group.add_argument(
        "--dropout", type=float, default=0.2,
        help="Dropout rate",
    )
    group.add_argument(
        "--beta", type=float, default=0.1,
        help="KL weight of the latent variable",
    )
    group.add_argument(
        "--scaffold-mod", type=str, default="Edgewise",
        choices={"Edgewise", "Bilinear"},
        help="Scaffold graph module",
    )
    group.add_argument(
        "--sparse-mod", type=str, default="L1",
        choices={"L1", "ScaleFree"},
        help="Sparse prior module",
    )
    group.add_argument(
        "--acyc-mod", type=str, default="SpecNorm",
        choices={"TrExp", "SpecNorm", "LogDet"},
        help="Acyclic prior module",
    )
    group.add_argument(
        "--latent-mod", type=str, default="EmbLatent",
        choices={"NilLatent", "EmbLatent", "GCNLatent"},
        help="Latent module",
    )
    group.add_argument(
        "--lik-mod", type=str, default="NegBin",
        choices={"Normal", "NegBin"},
        help="Causal likelihood module",
    )
    group.add_argument(
        "--kernel-mod", type=str, default="RBF",
        choices={"KroneckerDelta", "RBF"},
        help="SVGD kernel module",
    )
    group.add_argument(
        "--scaffold-graph", type=Path, default=None,
        help="Scaffold graph of the scaffold graph module (.gml)",
    )
    group.add_argument(
        "--scaffold-tau", type=float, default=None,
        help="Gumbel sigmoid temperature of the scaffold graph module",
    )
    group.add_argument(
        "--bilinear-emb-dim", type=int, default=None,
        help="Embedding dimension of the `Bilinear` scaffold graph module "
             "(only effective when the `Bilinear` module is used)",
    )
    group.add_argument(
        "--spec-norm-n-iter", type=int, default=None,
        help="Number of power iterations for the `SpecNorm` acyclic prior module "
             "(only effective when the `SpecNorm` module is used)",
    )
    group.add_argument(
        "--latent-data", type=Path, default=None,
        help="Depending on the latent module used, it can be "
             "the latent embedding of the `EmbLatent` module (.csv), or "
             "the latent graph of the `GCNLatent` module (.gml)",
    )
    group.add_argument(
        "--gcn-latent-emb-dim", type=int, default=None,
        help="Embedding dimension of the `GCNLatent` module "
             "(only effective when the `GCNLatent` module is used)"
    )
    group.add_argument(
        "--gcn-latent-n-layers", type=int, default=None,
        help="Number of layers for the `GCNLatent` module"
             "(only effective when the `GCNLatent` module is used)",
    )
    group.add_argument(
        "--random-seed", type=int, default=0,
        help="Random seed",
    )
    group.add_argument(
        "--log-dir", type=Path, default=".",
        help="Directory to store tensorboard logs",
    )


def add_fit_args(group: _ArgumentGroup) -> None:
    group.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay",
    )
    group.add_argument(
        "--accumulate-grad-batches", type=int, default=1,
        help="Number of batches to accumulate before optimizer step",
    )
    group.add_argument(
        "--log-adj", type=str, default="mean",
        choices={"none", "mean", "particles", "both"},
        help="Type of adjacency matrix to write to tensorboard logs",
    )
    group.add_argument(
        "--val-check-interval", type=int, default=300,
        help="Validation check interval in training steps",
    )
    group.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of data to use for validation",
    )
    group.add_argument(
        "--max-epochs", type=int, default=1000,
        help="Maximal number of training epochs",
    )
    group.add_argument(
        "--n-devices", type=int, default=1,
        help="Number of GPU devices to use",
    )


def add_pred_args(group: _ArgumentGroup) -> None:
    group.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size",
    )
    group.add_argument(
        "--n-devices", type=int, default=1,
        help="Number of GPU devices to use",
    )


def add_misc_args(group: _ArgumentGroup) -> None:
    group.add_argument(
        "--random-sleep", type=int, default=0,
        help="Sleep a random amount of time before starting",
    )
    group.add_argument(
        "-v", "--verbose", default=False, action="store_true",
        help="Enable verbose output",
    )


def parse_discover(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Input dataset (.h5ad)",
    )
    io.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Output discovered model (.pt)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    configure = parser.add_argument_group("Dataset configuration options")
    add_configure_args(configure)
    construct = parser.add_argument_group("Model construction options")
    add_construct_args(construct)
    fit = parser.add_argument_group("Model fitting options")
    fit.add_argument(
        "--lam", type=float, default=0.1,
        help="Sparse gradient coefficient",
    )
    fit.add_argument(
        "--alpha", type=float, default=0.5,
        help="Acyclicity gradient coefficient",
    )
    fit.add_argument(
        "--gamma", type=float, default=1.0,
        help="Kernel gradient coefficient",
    )
    fit.add_argument(
        "--cyc-tol", type=float, default=1e-4,
        help="Tolerance for cyclic penalty",
    )
    fit.add_argument(
        "--prefit", default=False, action="store_true",
        help="Whether to prefit the model on covariates only",
    )
    fit.add_argument(
        "--opt", type=str, default="AdamW",
        help="Optimizer type",
    )
    fit.add_argument(
        "--lr", type=float, default=5e-3,
        help="Learning rate",
    )
    fit.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size",
    )
    add_fit_args(fit)
    fit.add_argument(
        "--log-subdir", type=str, default="discover",
        help="Subdirectory to store tensorboard logs",
    )
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_acyclify(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Input discovered model (.pt)",
    )
    io.add_argument(
        "-g", "--graph", type=Path, required=True,
        help="Output acyclified causal graph (.gml)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_tune(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Input dataset (.h5ad)",
    )
    io.add_argument(
        "-g", "--graph", type=Path, required=True,
        help="Input acyclified causal graph (.gml)",
    )
    io.add_argument(
        "-m", "--input-model", type=Path, required=True,
        help="Input discovered model (*.pt)",
    )
    io.add_argument(
        "-o", "--output-model", type=Path, required=True,
        help="Output tuned model (.pt)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    configure = parser.add_argument_group("Dataset configuration options")
    add_configure_args(configure)
    fit = parser.add_argument_group("Model fitting options")
    fit.add_argument(
        "--tune-ctfact", default=False, action="store_true",
        help="Tune the model in counterfactual mode",
    )
    fit.add_argument(
        "--stratify", type=str, default=None,
        help="Stratify counterfactual pairs based on the given key in adata.obs",
    )
    fit.add_argument(
        "--opt", type=str, default="AdamW",
        help="Optimizer type",
    )
    fit.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate",
    )
    fit.add_argument(
        "--batch-size", type=int, default=128,
        help="Mini-batch size",
    )
    add_fit_args(fit)
    fit.add_argument(
        "--log-subdir", type=str, default="tune",
        help="Subdirectory to store tensorboard logs",
    )
    fit.add_argument(
        "--random-seed", type=int, default=None,
        help="Random seed",
    )
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_counterfactual(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Input dataset (.h5ad)",
    )
    io.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Input tuned model (*.pt)",
    )
    io.add_argument(
        "-u", "--design-module", type=Path, default=None,
        help="Input intervention design module (*.pt)",
    )
    io.add_argument(
        "-p", "--pred", type=Path, required=True,
        help="Output counterfactual prediction (.h5ad)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    configure = parser.add_argument_group("Dataset configuration options")
    add_configure_args(configure)
    pred = parser.add_argument_group("Model prediction options")
    pred.add_argument(
        "--fixed-genes", type=str, default=None,
        help="Comma-separated genes to fix in counterfactual prediction",
    )
    pred.add_argument(
        "--sample", default=False, action="store_true",
        help="Use random samples rather than mean for counterfactual prediction",
    )
    pred.add_argument(
        "--ablate-latent", default=False, action="store_true",
        help="Ablate latent contribution during counterfactual prediction",
    )
    pred.add_argument(
        "--ablate-interv", default=False, action="store_true",
        help="Ablate direct intervention during counterfactual prediction",
    )
    pred.add_argument(
        "--ablate-graph", default=False, action="store_true",
        help="Ablate graph contribution during counterfactual prediction",
    )
    add_pred_args(pred)
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_design(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Input source dataset (.h5ad)",
    )
    io.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Input tuned model (*.pt)",
    )
    io.add_argument(
        "-t", "--target", type=Path, required=True,
        help="Input design target (*.h5ad)",
    )
    io.add_argument(
        "--pool", type=Path, default=None,
        help="Input candidate variable pool to intervene (*.txt)",
    )
    io.add_argument(
        "--init", type=Path, default=None,
        help="Input initial design (*.txt)",
    )
    io.add_argument(
        "-o", "--output-design", type=Path, required=True,
        help="Output interventional design (.csv)",
    )
    io.add_argument(
        "-u", "--output-module", type=Path, default=None,
        help="Output intervention design module (*.pt)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    configure = parser.add_argument_group("Dataset configuration options")
    add_configure_args(configure)
    fit = parser.add_argument_group("Model fitting options")
    fit.add_argument(
        "--design-size", type=int, default=1,
        help="Expected number of perturbation targets to design",
    )
    fit.add_argument(
        "--design-scale-bias", default=False, action="store_true",
        help="Whether to design interventional scale and bias",
    )
    fit.add_argument(
        "--target-weight", type=str, default=None,
        help="Key in target.var containing the weight of each gene",
    )
    fit.add_argument(
        "--stratify", type=str, default=None,
        help="Stratify design pairs based on the given key in adata.obs",
    )
    fit.add_argument(
        "--opt", type=str, default="AdamW",
        help="Optimizer type",
    )
    fit.add_argument(
        "--lr", type=float, default=5e-2,
        help="Learning rate",
    )
    fit.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size",
    )
    add_fit_args(fit)
    fit.add_argument(
        "--log-subdir", type=str, default="design",
        help="Subdirectory to store tensorboard logs",
    )
    fit.add_argument(
        "--random-seed", type=int, default=None,
        help="Random seed",
    )
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_design_brute_force(parser: ArgumentParser) -> None:
    io = parser.add_argument_group("Input/output options")
    io.add_argument(
        "-d", "--data", type=Path, required=True,
        help="Input source dataset (.h5ad)",
    )
    io.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Input tuned model (*.pt)",
    )
    io.add_argument(
        "-t", "--target", type=Path, required=True,
        help="Input design target (*.h5ad)",
    )
    io.add_argument(
        "--pool", type=Path, default=None,
        help="Input candidate variable pool to intervene (*.txt)",
    )
    io.add_argument(
        "-o", "--output-design", type=Path, required=True,
        help="Output interventional design (.csv)",
    )
    io.add_argument(
        "-p", "--pred", type=Path, default=None,
        help="Output counterfactual prediction (.h5ad)",
    )
    io.add_argument(
        "-i", "--info", type=Path, default=None,
        help="Output run information (.yaml)",
    )
    configure = parser.add_argument_group("Dataset configuration options")
    add_configure_args(configure)
    pred = parser.add_argument_group("Model prediction options")
    pred.add_argument(
        "--design-size", type=int, default=1,
        help="Expected number of perturbation targets to design",
    )
    pred.add_argument(
        "-k", type=int, default=30,
        help="Number of cells to predict for each possible intervention",
    )
    pred.add_argument(
        "--n-neighbors", type=int, default=30,
        help="Number of counterfactual neighbors to consider for each target cell",
    )
    add_pred_args(pred)
    misc = parser.add_argument_group("Miscellaneous options")
    add_misc_args(misc)


def parse_upgrade(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-m", "--model", type=Path, required=True,
        help="Model to be upgraded (*.pt)"
    )


def parse_devmgr(parser: ArgumentParser) -> None:
    subparsers = parser.add_subparsers(
        description="Select a function", dest="sub_cmd", required=True
    )
    init = subparsers.add_parser("init", description="Initialize devices")
    init.add_argument(
        "--n-devices", type=int, required=True,
        help="Number of devices to initialize",
    )
    acquire = subparsers.add_parser("acquire", description="Acquire devices")
    acquire.add_argument(
        "--n-devices", type=int, required=True,
        help="Number of devices to acquire",
    )
    release = subparsers.add_parser("release", description="Release devices")
    release.add_argument(
        "--devices", type=str, required=True,
        help="List of comma-separated devices to release",
    )


def parse_args(args: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=(
            "CASCADE: Causality Aware Single-Cell Automatic "
            "Discovery/Deduction/Design Engine"
        ),
        epilog="Please check the help message of each subcommand for more details.",
    )
    subparsers = parser.add_subparsers(
        description="Select a function", dest="cmd", required=True
    )
    discover = subparsers.add_parser(
        "discover", description="Run causal discovery"
    )
    parse_discover(discover)
    acyclify = subparsers.add_parser(
        "acyclify", description="Acyclify discovered causal graph"
    )
    parse_acyclify(acyclify)
    tune = subparsers.add_parser(
        "tune", description="Tune acyclified model"
    )
    parse_tune(tune)
    counterfactual = subparsers.add_parser(
        "counterfactual", description="Run counterfactual prediction"
    )
    parse_counterfactual(counterfactual)
    design = subparsers.add_parser(
        "design", description="Targeted intervention design"
    )
    parse_design(design)
    design_brute_force = subparsers.add_parser(
        "design_brute_force",
        description="Targeted intervention design with brute-force search",
    )
    parse_design_brute_force(design_brute_force)
    upgrade = subparsers.add_parser(
        "upgrade", description="Upgrade saved CASCADE model"
    )
    parse_upgrade(upgrade)
    devmgr = subparsers.add_parser(
        "devmgr", description="Device manager"
    )
    parse_devmgr(devmgr)
    return parser.parse_args(args=args)
# fmt: on


def lock(f):
    try:
        from filelock import FileLock, SoftFileLock
    except ImportError as e:  # pragma: no cover
        raise ImportError("`devmgr` requires installing `filelock`") from e

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            with FileLock("devices.lock"):
                return f(*args, **kwargs)
        except NotImplementedError:
            with SoftFileLock("devices.soft-lock"):
                return f(*args, **kwargs)

    return wrapper


def read(f):
    f.seek(0)
    return [int(device.strip()) for device in f.readlines()]


def write(f, devices):
    f.seek(0)
    f.truncate()
    f.writelines(f"{i}\n" for i in sorted(devices))


@lock
def init(n_devices: int) -> None:
    with open("devices.txt", "w") as f:
        write(f, range(n_devices))


@lock
def acquire(n_devices: int) -> str:
    with open("devices.txt", "r+") as f:
        available = read(f)
        acquired = sample(available, n_devices)
        write(f, set(available) - set(acquired))
        return acquired


@lock
def release(devices: Iterable[str]) -> None:
    with open("devices.txt", "r+") as f:
        available = read(f)
        write(f, set(available) | set(devices))


def run_discover(args: Namespace) -> None:
    import anndata as ad
    import networkx as nx
    import numpy as np
    import pandas as pd
    import yaml
    from kneed import KneeLocator
    from loguru import logger
    from sklearn.decomposition import TruncatedSVD

    from .data import _get_covariate, aggregate_obs, configure_dataset, encode_regime
    from .model import CASCADE, LogAdj

    adata = ad.read_h5ad(args.data)
    scaffold_graph = nx.read_gml(args.scaffold_graph) if args.scaffold_graph else None
    latent_data = None
    if args.latent_mod == "EmbLatent":
        if args.latent_data is None:
            logger.warning(
                "EmbLatent requested but missing latent data. "
                "Using perturbation SVD to compute latent embedding..."
            )
            adata_agg = aggregate_obs(adata, args.interv_key, X_agg="mean").to_df()
            adata_diff = adata_agg.loc[adata_agg.index != ""] - adata_agg.loc[""]
            adata_diff /= adata_diff.std()
            n_comps = min(adata_diff.shape) - 1
            trunc_svd = TruncatedSVD(n_components=n_comps, algorithm="arpack")
            svd = trunc_svd.fit_transform(adata_diff.T)
            argsort = trunc_svd.explained_variance_ratio_.argsort(kind="stable")[::-1]
            exp_var_ratio = trunc_svd.explained_variance_ratio_[argsort]
            svd = svd[:, argsort]
            knee = KneeLocator(
                np.arange(n_comps),
                exp_var_ratio,
                curve="convex",
                direction="decreasing",
            )
            if knee.knee is None:
                logger.warning("No knee point detected. Using all dimensions...")
                knee.knee = n_comps - 1
            logger.info(f"Using SVD dimensions: {knee.knee + 1}")
            exp_var = exp_var_ratio[: knee.knee + 1].sum()
            logger.info(f"Total explained variance: {exp_var}")
            latent_data = pd.DataFrame(svd[:, : knee.knee + 1], index=adata.var_names)
        else:
            latent_data = pd.read_csv(args.latent_data, index_col=0)
    elif args.latent_mod == "GCNLatent" and args.latent_data is not None:
        latent_data = nx.read_gml(args.latent_data)

    sleep(randint(0, args.random_sleep))
    start_time = time()

    encode_regime(adata, "interv", args.interv_key)
    configure_dataset(
        adata,
        use_regime="interv",
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_weight=args.use_weight,
        use_layer=args.use_layer,
    )

    scaffold_kws = {}
    if args.scaffold_tau is not None:
        scaffold_kws["tau"] = args.scaffold_tau
    if args.scaffold_mod == "Bilinear" and args.bilinear_emb_dim is not None:
        scaffold_kws["emb_dim"] = args.bilinear_emb_dim
    acyc_kws = {}
    if args.acyc_mod == "SpecNorm" and args.spec_norm_n_iter is not None:
        acyc_kws["n_iter"] = args.spec_norm_n_iter
    latent_kws = {}
    if args.latent_mod == "GCNLatent" and args.gcn_latent_emb_dim is not None:
        latent_kws["emb_dim"] = args.gcn_latent_emb_dim
    if args.latent_mod == "GCNLatent" and args.gcn_latent_n_layers is not None:
        latent_kws["n_layers"] = args.gcn_latent_n_layers

    model = CASCADE(
        vars=adata.var_names,
        n_particles=args.n_particles,
        n_covariates=_get_covariate(adata).shape[1],
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        beta=args.beta,
        scaffold_mod=args.scaffold_mod,
        sparse_mod=args.sparse_mod,
        acyc_mod=args.acyc_mod,
        latent_mod=args.latent_mod,
        lik_mod=args.lik_mod,
        kernel_mod=args.kernel_mod,
        scaffold_graph=scaffold_graph,
        latent_data=latent_data,
        scaffold_kws=scaffold_kws,
        acyc_kws=acyc_kws,
        latent_kws=latent_kws,
        random_state=args.random_seed,
        log_dir=args.log_dir,
    )

    model.discover(
        adata,
        lam=args.lam,
        alpha=args.alpha,
        gamma=args.gamma,
        cyc_tol=args.cyc_tol,
        prefit=args.prefit,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_adj=LogAdj[args.log_adj],
        batch_size=args.batch_size,
        val_check_interval=args.val_check_interval,
        val_frac=args.val_frac,
        max_epochs=args.max_epochs,
        n_devices=args.n_devices,
        log_subdir=args.log_subdir,
        verbose=args.verbose,
    )
    elapsed_time = time() - start_time

    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model)
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


def _acyclify(g):
    from loguru import logger

    from .graph import acyclify, filter_edges

    g = filter_edges(g, cutoff=0.5)
    acyc = acyclify(g)
    logger.info(
        f"Removed {g.number_of_edges() - acyc.number_of_edges()} cyclic edges..."
    )
    return acyc


def run_acyclify(args: Namespace) -> None:
    from multiprocessing import Pool

    import networkx as nx
    import yaml

    from .graph import demultiplex, multiplex
    from .model import CASCADE

    model = CASCADE.load(args.model)

    sleep(randint(0, args.random_sleep))
    start_time = time()
    graph = model.export_causal_graph()
    with Pool(model.net.n_particles) as p:
        graph_acyc_list = p.map(_acyclify, demultiplex(graph))
    graph_acyc = multiplex(*graph_acyc_list)
    elapsed_time = time() - start_time

    args.graph.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gml(graph_acyc, args.graph)
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


def run_tune(args: Namespace) -> None:
    import anndata as ad
    import networkx as nx
    import yaml

    from .data import configure_dataset, encode_regime
    from .model import CASCADE, LogAdj

    adata = ad.read_h5ad(args.data)
    graph = nx.read_gml(args.graph)
    model = CASCADE.load(args.input_model)

    sleep(randint(0, args.random_sleep))
    start_time = time()
    model.import_causal_graph(graph)
    encode_regime(adata, "interv", args.interv_key)
    configure_dataset(
        adata,
        use_regime="interv",
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_weight=args.use_weight,
        use_layer=args.use_layer,
    )
    if args.random_seed is not None:
        model.rnd.seed(args.random_seed)
    model.tune(
        adata,
        tune_ctfact=args.tune_ctfact,
        stratify=args.stratify,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_adj=LogAdj[args.log_adj],
        batch_size=args.batch_size,
        val_check_interval=args.val_check_interval,
        val_frac=args.val_frac,
        max_epochs=args.max_epochs,
        n_devices=args.n_devices,
        log_subdir=args.log_subdir,
        verbose=args.verbose,
    )
    elapsed_time = time() - start_time

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output_model)
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


def run_counterfactual(args: Namespace) -> None:
    import anndata as ad
    import yaml

    from .data import configure_dataset, encode_regime
    from .model import CASCADE
    from .nn import IntervDesign

    adata = ad.read_h5ad(args.data)
    model = CASCADE.load(args.model)
    design = IntervDesign.load(args.design_module) if args.design_module else None

    sleep(randint(0, args.random_sleep))
    start_time = time()
    encode_regime(adata, "interv", args.interv_key)
    configure_dataset(
        adata,
        use_regime="interv",
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_weight=args.use_weight,
        use_layer=args.use_layer,
    )
    ctfact = model.counterfactual(
        adata,
        batch_size=args.batch_size,
        n_devices=args.n_devices,
        design=design,
        fixed_genes=args.fixed_genes.split(",") if args.fixed_genes else None,
        sample=args.sample,
        ablate_latent=args.ablate_latent,
        ablate_interv=args.ablate_interv,
        ablate_graph=args.ablate_graph,
    )
    elapsed_time = time() - start_time

    args.pred.parent.mkdir(parents=True, exist_ok=True)
    ctfact.write(args.pred, compression="gzip")
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


def run_design(args: Namespace) -> None:
    import anndata as ad
    import numpy as np
    import yaml

    from .data import configure_dataset, encode_regime
    from .model import CASCADE

    source = ad.read_h5ad(args.data)
    target = ad.read_h5ad(args.target)
    model = CASCADE.load(args.model)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist() if args.pool else None
    init = np.loadtxt(args.init, dtype=str, ndmin=1).tolist() if args.init else None

    sleep(randint(0, args.random_sleep))
    start_time = time()
    encode_regime(source, "interv", args.interv_key)
    configure_dataset(
        source,
        use_regime="interv",
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_weight=args.use_weight,
        use_layer=args.use_layer,
    )
    configure_dataset(
        target,
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_layer=args.use_layer,
    )
    if args.random_seed is not None:
        model.rnd.seed(args.random_seed)
    design, design_mod = model.design(
        source,
        target,
        pool=pool,
        init=init,
        design_size=args.design_size,
        design_scale_bias=args.design_scale_bias,
        target_weight=args.target_weight,
        stratify=args.stratify,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        accumulate_grad_batches=args.accumulate_grad_batches,
        batch_size=args.batch_size,
        val_check_interval=args.val_check_interval,
        val_frac=args.val_frac,
        max_epochs=args.max_epochs,
        n_devices=args.n_devices,
        log_subdir=args.log_subdir,
        verbose=args.verbose,
    )
    elapsed_time = time() - start_time

    args.output_design.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output_design)
    if args.output_module:
        args.output_module.parent.mkdir(parents=True, exist_ok=True)
        design_mod.save(args.output_module)
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


def run_design_brute_force(args: Namespace) -> None:
    import anndata as ad
    import numpy as np
    import yaml

    from .data import configure_dataset, encode_regime
    from .model import CASCADE

    source = ad.read_h5ad(args.data)
    target = ad.read_h5ad(args.target)
    model = CASCADE.load(args.model)
    pool = np.loadtxt(args.pool, dtype=str, ndmin=1).tolist() if args.pool else None

    sleep(randint(0, args.random_sleep))
    start_time = time()
    encode_regime(source, "interv", args.interv_key)
    configure_dataset(
        source,
        use_regime="interv",
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_weight=args.use_weight,
        use_layer=args.use_layer,
    )
    configure_dataset(
        target,
        use_covariate=args.use_covariate,
        use_size=args.use_size,
        use_layer=args.use_layer,
    )
    design, ctfact = model.design_brute_force(
        source,
        target,
        pool=pool,
        design_size=args.design_size,
        k=args.k,
        counterfactual_kws={
            "batch_size": args.batch_size,
            "n_devices": args.n_devices,
        },
        neighbor_kws={"n_neighbors": args.n_neighbors},
    )
    elapsed_time = time() - start_time

    args.output_design.parent.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output_design)
    if args.pred:
        args.pred.parent.mkdir(parents=True, exist_ok=True)
        ctfact.write(args.pred, compression="gzip")
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


def run_upgrade(args: Namespace) -> None:  # pragma: no cover
    from .model import upgrade_saved_model

    upgrade_saved_model(args.model)


def run_devmgr(args: Namespace) -> str | None:
    if args.sub_cmd == "init":
        return init(args.n_devices)
    if args.sub_cmd == "acquire":
        acquired = acquire(args.n_devices)
        acquired = ",".join(str(device) for device in sorted(acquired))
        print(acquired)
        return acquired
    # args.sub_cmd == "release"
    return release(set(map(int, args.devices.split(","))))


def main(args: list[str] | None = None) -> str | None:
    args = parse_args(args=args)
    if args.cmd == "discover":
        return run_discover(args)
    if args.cmd == "acyclify":
        return run_acyclify(args)
    if args.cmd == "tune":
        return run_tune(args)
    if args.cmd == "counterfactual":
        return run_counterfactual(args)
    if args.cmd == "design":
        return run_design(args)
    if args.cmd == "design_brute_force":
        return run_design_brute_force(args)
    if args.cmd == "upgrade":
        return run_upgrade(args)
    # args.cmd == "devmgr"
    return run_devmgr(args)


if __name__ == "__main__":
    main()
