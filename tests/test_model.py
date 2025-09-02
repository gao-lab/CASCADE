from os import environ
from random import randint
from time import sleep

import networkx as nx
import pandas as pd
import torch
from allpairspy import AllPairs
from pytest import mark, raises
from scipy.sparse import csr_matrix
from torch.nn import init

from cascade.data import configure_dataset, encode_regime
from cascade.graph import (
    acyclify,
    annotate_explanation,
    core_explanation_graph,
    demultiplex,
    filter_edges,
    multiplex,
    prep_cytoscape,
)
from cascade.model import CASCADE, LogAdj

from .utils import array_close, graph_cmp, model_close


@mark.parametrize(
    [
        "scaffold_mod",
        "sparse_mod",
        "acyc_mod",
        "latent_mod",
        "lik_mod",
        "kernel_mod",
    ],
    [
        values
        for values in AllPairs(
            [
                ["Edgewise", "Bilinear"],
                ["L1", "ScaleFree"],
                ["TrExp", "SpecNorm", "LogDet"],
                ["NilLatent", "EmbLatent", "GCNLatent"],
                ["Normal", "NegBin"],
                ["KroneckerDelta", "RBF"],
            ]
        )
    ],
)
def test_reproducibility(
    adata,
    dag,
    tmp_path,
    scaffold_mod,
    sparse_mod,
    acyc_mod,
    latent_mod,
    lik_mod,
    kernel_mod,
):
    worker_count = int(environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
    fit_kws = {"max_epochs": 20, "verbose": True}
    sleep(randint(0, 2 * (worker_count - 1)))
    encode_regime(adata, "interv", key="interv")
    configure_dataset(
        adata, use_regime="interv", use_size="size" if lik_mod == "NegBin" else None
    )
    if latent_mod == "EmbLatent":
        latent_data = pd.DataFrame(adata.varm["repr"], index=adata.var_names)
    elif latent_mod == "GCNLatent":
        latent_data = nx.Graph(dag)
    else:  # latent_mod == "NilLatent"
        latent_data = None
    model1 = CASCADE(
        adata.var_names,
        latent_dim=4,
        scaffold_mod=scaffold_mod,
        sparse_mod=sparse_mod,
        acyc_mod=acyc_mod,
        latent_mod=latent_mod,
        lik_mod=lik_mod,
        kernel_mod=kernel_mod,
        scaffold_graph=nx.Graph(dag),
        latent_data=latent_data,
        log_dir=tmp_path,
    )
    model1.discover(adata, cyc_tol=1e-3, **fit_kws)
    digraph1 = model1.export_causal_graph()
    model1.save(tmp_path / "save.pt")
    loaded1 = CASCADE.load(tmp_path / "save.pt")
    assert model_close(model1.net, loaded1.net)

    model2 = CASCADE(
        adata.var_names,
        latent_dim=4,
        scaffold_mod=scaffold_mod,
        sparse_mod=sparse_mod,
        acyc_mod=acyc_mod,
        latent_mod=latent_mod,
        lik_mod=lik_mod,
        kernel_mod=kernel_mod,
        scaffold_graph=nx.Graph(dag),
        latent_data=latent_data,
        log_dir=tmp_path,
    )
    model2.discover(adata, cyc_tol=1e-3, **fit_kws)
    digraph2 = model2.export_causal_graph()
    assert graph_cmp(digraph1, digraph2, "weight", lambda x, y: array_close(x, y))
    assert model_close(model1.net, model2.net)

    acyc_digraph2 = multiplex(
        *(acyclify(filter_edges(g, cutoff=0.5)) for g in demultiplex(digraph2))
    )
    model2.import_causal_graph(acyc_digraph2)
    model2.tune(adata, tune_ctfact=True, log_adj=LogAdj.both, **fit_kws)
    digraph3 = model2.export_causal_graph()
    assert graph_cmp(acyc_digraph2, digraph3, "weight", lambda x, y: array_close(x, y))
    assert not model_close(model1.net, model2.net)


def test_err(adata, dag, tmp_path):
    encode_regime(adata, "interv", key="interv")
    configure_dataset(adata, use_regime="interv")
    with raises(TypeError):
        CASCADE(
            adata.var_names,
            latent_dim=4,
            scaffold_mod="L1",
            sparse_mod="L1",
            acyc_mod="TrExp",
            latent_mod="EmbLatent",
            lik_mod="Normal",
            kernel_mod="RBF",
            scaffold_graph=nx.Graph(dag),
            latent_data=pd.DataFrame(adata.varm["repr"], index=adata.var_names),
            log_dir=tmp_path,
        )
    dag.add_edge("A", "A")
    with raises(ValueError):
        CASCADE(
            adata.var_names,
            latent_dim=4,
            latent_mod="EmbLatent",
            scaffold_graph=nx.Graph(dag),
            latent_data=pd.DataFrame(adata.varm["repr"], index=adata.var_names),
            log_dir=tmp_path,
        )


def test_design(adata, dag, tmp_path):
    encode_regime(adata, "interv", key="interv")
    configure_dataset(adata, use_regime="interv", use_size="size")

    model = CASCADE(
        adata.var_names,
        n_particles=2,
        latent_dim=4,
        latent_mod="EmbLatent",
        scaffold_graph=nx.Graph(dag),
        latent_data=pd.DataFrame(adata.varm["repr"], index=adata.var_names),
        log_dir=tmp_path,
    )
    model.discover(adata, cyc_tol=0, max_epochs=20)
    digraph = model.export_causal_graph()
    digraph = multiplex(
        *(acyclify(filter_edges(g, cutoff=0.5)) for g in demultiplex(digraph))
    )
    model.import_causal_graph(digraph)
    model.tune(adata, tune_ctfact=True, max_epochs=20)
    model.save(tmp_path / "save.pt")
    loaded = CASCADE.load(tmp_path / "save.pt")
    assert model_close(model.net, loaded.net)

    target0 = adata[adata.obs["interv"] == "D,C"].copy()
    design0, _ = model.design_brute_force(adata, target0, design_size=2, k=10)
    assert design0.shape[1] == 1

    target1 = adata[adata.obs["interv"] == "B"].copy()
    model.rnd.seed(0)
    design1, mod1 = model.design(
        adata,
        target1,
        design_size=2,
        design_scale_bias=True,
        lr=0.05,
        max_epochs=20,
    )
    model.net.design = None
    loaded.net.design = None
    assert model_close(model.net, loaded.net)
    _ = model.design_error_curve(adata, target1, mod1)

    target2 = adata[adata.obs["interv"] == "B"].copy()
    model.rnd.seed(0)
    design2, mod2 = model.design(
        adata,
        target2,
        design_size=2,
        design_scale_bias=True,
        lr=0.05,
        max_epochs=20,
    )
    model.net.design = None
    assert model_close(model.net, loaded.net)
    assert design2.equals(design1)
    assert model_close(mod2, mod1)

    target3 = adata[adata.obs["interv"] == "B"].copy()
    model.rnd.seed(0)
    design3, mod3 = model.design(
        adata,
        target3,
        design_size=2,
        design_scale_bias=False,
        lr=0.05,
        max_epochs=20,
    )
    model.net.design = None
    assert model_close(model.net, loaded.net)
    assert not design3.equals(design1)
    assert not model_close(mod3, mod1)

    target4 = adata[adata.obs["interv"] == "C,D"].copy()
    model.rnd.seed(0)
    design4, mod4 = model.design(
        adata,
        target4,
        design_size=2,
        design_scale_bias=False,
        lr=0.05,
        max_epochs=20,
    )
    model.net.design = None
    assert model_close(model.net, loaded.net)
    assert not design4.equals(design3)
    assert not model_close(mod4, mod3)
    init.zeros_(mod3.logits)
    init.zeros_(mod4.logits)
    assert model_close(mod4, mod3)


def test_counterfactual(adata, dag, tmp_path):
    encode_regime(adata, "interv", key="interv")
    encode_regime(adata, "ctfact", key="ctfact")
    configure_dataset(
        adata, use_regime="interv", use_size="size", use_covariate="covariate"
    )
    model = CASCADE(
        adata.var_names,
        n_particles=2,
        n_covariates=adata.obsm["covariate"].shape[1],
        latent_dim=4,
        latent_mod="EmbLatent",
        lik_mod="NegBin",
        scaffold_graph=nx.Graph(dag),
        latent_data=pd.DataFrame(adata.varm["repr"], index=adata.var_names),
        log_dir=tmp_path,
    )
    model.discover(adata, cyc_tol=0, prefit=True, max_epochs=1)
    model.discover(adata, cyc_tol=0, max_epochs=20)
    digraph = model.export_causal_graph()
    digraph = multiplex(
        *(acyclify(filter_edges(g, cutoff=0.5)) for g in demultiplex(digraph))
    )
    model.import_causal_graph(digraph)
    model.tune(adata, tune_ctfact=True, max_epochs=20)

    configure_dataset(adata, use_regime="ctfact", use_size="size")
    adata_ctfact = model.counterfactual(adata, batch_size=16)
    assert not array_close(adata_ctfact.X, adata.X)

    adata.X = csr_matrix(adata.X)
    adata_ctfact1 = model.counterfactual(adata, batch_size=12)
    assert array_close(adata_ctfact1.X, adata_ctfact.X)

    model.save(tmp_path / "test.pt")
    model = CASCADE.load(tmp_path / "test.pt")
    adata_ctfact2 = model.counterfactual(adata, batch_size=16)
    assert array_close(adata_ctfact2.X, adata_ctfact.X)

    if torch.cuda.is_available():
        model.net = model.net.cuda()
        adata_ctfact3 = model.counterfactual(adata, batch_size=16)
        assert array_close(adata_ctfact3.X, adata_ctfact.X)

    encode_regime(adata, "ctfact", key="complete")
    adata_ctfact4 = model.counterfactual(adata, batch_size=16)
    assert not array_close(adata_ctfact4.X, adata_ctfact.X)

    configure_dataset(adata, use_regime="interv", use_size="size")
    configure_dataset(adata_ctfact, use_layer="X_ctfact", use_size="size")
    adata_explain = model.explain(adata, adata_ctfact, batch_size=16)
    digraph_explain = annotate_explanation(
        digraph, adata_explain, model.export_causal_map()
    )
    digraph_core = core_explanation_graph(digraph_explain, list(digraph_explain.nodes))
    _ = prep_cytoscape(
        digraph_core, digraph, list(digraph_core.nodes), list(digraph.nodes)
    )


def test_diagnose(adata, dag, tmp_path):
    encode_regime(adata, "interv", key="interv")
    encode_regime(adata, "ctfact", key="ctfact")
    configure_dataset(
        adata, use_regime="interv", use_size="size", use_covariate="covariate"
    )
    model = CASCADE(
        adata.var_names,
        n_particles=2,
        n_covariates=adata.obsm["covariate"].shape[1],
        latent_dim=4,
        latent_mod="EmbLatent",
        lik_mod="NegBin",
        scaffold_graph=nx.Graph(dag),
        latent_data=pd.DataFrame(adata.varm["repr"], index=adata.var_names),
        log_dir=tmp_path,
    )
    model.discover(adata, cyc_tol=0, prefit=False, lr=0.1, max_epochs=200, verbose=True)

    adata_diag = model.diagnose(adata[:32], batch_size=4)
    assert adata_diag.obsm["Z_mean_diag"].shape == (32, 4, 2)
    assert adata_diag.obsm["Z_std_diag"].shape == (32, 4, 2)
    assert adata_diag.layers["X_mean_diag"].shape == (32, adata.n_vars, 2)
    assert adata_diag.layers["X_std_diag"].shape == (32, adata.n_vars, 2)
    assert adata_diag.layers["X_disp_diag"].shape == (32, adata.n_vars, 2)

    adata_jac = model.jacobian(adata[:32], batch_size=4)
    assert adata_jac.layers["X_jac"].shape == (32, adata.n_vars, adata.n_vars, 2)
