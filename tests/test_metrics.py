import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from pytest import mark, raises

from cascade.graph import multiplex
from cascade.metrics import (
    annot_resp,
    ctfact_delta_pcc,
    ctfact_dir_acc,
    ctfact_mse,
    disc_acc,
    disc_ap,
    disc_auroc,
    disc_f1,
    disc_prec,
    disc_recall,
    disc_resp_acc,
    disc_resp_dist,
    disc_resp_dist_diff,
    disc_shd,
    disc_sid,
    dsgn_auhrc_exact,
    dsgn_auhrc_partial,
)


@mark.parametrize("use_multiplex", [True, False])
def test_disc_acc(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_acc(true, pred) == 1.0
    assert 0.0 < disc_acc(true.reverse(), pred) < 1.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_prec(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_prec(true, pred) == 1.0
    assert disc_prec(true.reverse(), pred) == 0.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_recall(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_recall(true, pred) == 1.0
    assert disc_recall(true.reverse(), pred) == 0.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_f1(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_f1(true, pred) == 1.0
    assert disc_f1(true.reverse(), pred) == 0.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_auroc(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_auroc(true, pred) == 1.0
    assert 0.0 < disc_auroc(true.reverse(), pred) < 1.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_ap(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_ap(true, pred) == 1.0
    assert 0.0 < disc_ap(true.reverse(), pred) < 1.0


@mark.parametrize("use_multiplex", [True, False])
def test_disc_shd(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_shd(true, pred) == 0.0
    assert disc_shd(true.reverse(), pred, cutoff=0.5) == 2 * dag.number_of_edges()


@mark.use_ri
@mark.parametrize("use_multiplex", [True, False])
def test_disc_sid(dag, use_multiplex):
    true = dag
    pred = multiplex(dag, dag) if use_multiplex else dag
    assert disc_sid(true, pred) == 0.0
    assert disc_sid(true.reverse(), pred) > 0.0
    assert disc_sid(true, nx.compose(pred, pred.reverse())) > 0.0
    with raises(ValueError):
        disc_sid(nx.Graph(true), pred)


def test_disc_resp(dag, adata):
    annot_resp(dag, adata, "interv")
    disc_resp_acc(dag)
    disc_resp_dist(dag)
    disc_resp_dist_diff(dag)


def test_ctfact_mse(adata):
    sc.tl.rank_genes_groups(adata, "interv", reference="", method="t-test")
    mse_df = ctfact_mse(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
    )
    assert np.isclose(mse_df["normalized_mse"], 0).all()
    mse_df = ctfact_mse(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
        top_de=5,
    )
    assert np.isclose(mse_df["normalized_mse"], 0).all()


def test_ctfact_delta_pcc(adata):
    sc.tl.rank_genes_groups(adata, "interv", reference="", method="t-test")
    pcc_df = ctfact_delta_pcc(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
    )
    assert np.isclose(pcc_df["delta_pcc"], 1).all()
    pcc_df = ctfact_delta_pcc(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
        top_de=5,
    )
    assert np.isclose(pcc_df["delta_pcc"], 1).all()


def test_ctfact_dir_acc(adata):
    sc.tl.rank_genes_groups(adata, "interv", reference="", method="t-test")
    acc_df = ctfact_dir_acc(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
    )
    assert np.isclose(acc_df["dir_acc"], 1).all()
    acc_df = ctfact_dir_acc(
        adata[adata.obs["interv"] == ""],
        adata[adata.obs["interv"] != ""],
        adata[adata.obs["interv"] != ""],
        "interv",
        top_de=5,
    )
    assert np.isclose(acc_df["dir_acc"], 1).all()


def test_dsgn_auhrc():
    designs = pd.DataFrame(
        {
            "a": [0.5, 0.7, 0.9, 0.8, 0.6],
            "b": [0.5, 0.7, 0.9, 0.8, 0.6],
            "d": [0.5, 0.7, 0.9, 0.8, 0.6],
        },
        index=["a", "b", "c", "d", "e"],
    )
    exact = dsgn_auhrc_exact(designs)
    partial = dsgn_auhrc_partial(designs)
    assert 0 < exact < 1
    assert 0 < partial < 1
    assert exact == partial
    designs = pd.DataFrame(
        {
            "a,b": [0.5, 0.7, 0.9, 0.8, 0.6],
            "b,c": [0.5, 0.7, 0.9, 0.8, 0.6],
            "d,e": [0.5, 0.7, 0.9, 0.8, 0.6],
        },
        index=["a,b", "b,c", "c,d", "d,e", "e,f"],
    )
    exact = dsgn_auhrc_exact(designs)
    partial = dsgn_auhrc_partial(designs)
    assert 0 < exact < partial < 1
