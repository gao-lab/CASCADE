import numpy as np
from pytest import raises

from cascade.data import (
    Targets,
    _get_covariate,
    _get_regime,
    _get_size,
    _get_weight,
    _get_X,
    _set_covariate,
    _set_regime,
    _set_size,
    _set_weight,
    _set_X,
    configure_dataset,
    encode_regime,
    filter_unobserved_targets,
    get_all_targets,
    get_configuration,
    neighbor_impute,
    simple_design,
)

from .utils import array_close


def test_targets():
    assert Targets() == Targets("") == Targets([]) == Targets(",,")
    assert Targets("A,B") == Targets("B,A")
    assert Targets("A,B") >= Targets("B,A")

    assert Targets("A,B,A") == Targets("B,A")
    assert Targets("A,B,A") <= Targets("B,A")

    assert Targets("A,B") < Targets("A,B,C")
    assert not Targets("A,B") > Targets("A,B,C")
    assert Targets("A,B,C") < Targets("A,B,D")
    assert not Targets("A,B,C") > Targets("A,B,D")

    assert Targets("A,B,C") > Targets("A,B")
    assert not Targets("A,B,C") < Targets("A,B")
    assert Targets("A,B,D") > Targets("A,B,C")
    assert not Targets("A,B,D") < Targets("A,B,C")

    assert Targets("A,B,C") & ["C", "D"] == Targets("C")
    assert Targets("A,B,C") | ["C", "D"] == Targets("A,B,C,D")
    assert Targets("A,B,C") ^ ["C", "D"] == Targets("A,B,D")
    assert Targets("A,B,C") - ["A"] == Targets("B,C")

    assert repr(Targets("A,B,A")) == "A,B"


def test_get_all_targets(adata):
    assert get_all_targets(adata, "interv") == Targets("B,C,D,E")
    assert get_all_targets(adata, "ctfact") == Targets("D,F,G,H")


def test_filter_unobserved_targets(adata):
    adata1 = filter_unobserved_targets(adata, "interv")
    assert (adata1.obs_names == adata.obs_names).all()
    adata2 = filter_unobserved_targets(adata, "ctfact")
    assert (adata2.obs_names == adata.obs_names).all()
    adata.obs["interv"] = adata.obs["interv"].replace({"B": "Z"})
    adata3 = filter_unobserved_targets(adata, "interv")
    assert adata.n_obs - adata3.n_obs == 200
    adata.obs["ctfact"] = adata.obs["ctfact"].replace({"F,G": "F,Z"})
    adata4 = filter_unobserved_targets(adata, "ctfact")
    assert adata.n_obs - adata4.n_obs == 100


def test_encode_regime(adata):
    encode_regime(adata, "regime", key="interv")
    correct = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * 200
        + [[0, 1, 0, 0, 0, 0, 0, 0, 0]] * 200
        + [[0, 0, 0, 1, 0, 0, 0, 0, 0]] * 200
        + [[0, 0, 1, 1, 0, 0, 0, 0, 0]] * 200
        + [[0, 0, 0, 0, 1, 0, 0, 0, 0]] * 200
    ).astype(bool)
    assert array_close(adata.layers["regime"].toarray(), correct)
    adata.obs["interv"] = adata.obs["interv"] + ",Z"
    with raises(ValueError):
        encode_regime(adata, "regime", key="interv")


def test_configure_dataset(adata):
    with raises(KeyError):
        get_configuration(adata)

    with raises(KeyError):
        configure_dataset(adata, use_regime="unknown")
    with raises(KeyError):
        configure_dataset(adata, use_covariate="unknown")
    with raises(KeyError):
        configure_dataset(adata, use_size="unknown")
    with raises(KeyError):
        configure_dataset(adata, use_weight="unknown")
    with raises(KeyError):
        configure_dataset(adata, use_layer="unknown")

    _get_X(adata)
    _get_regime(adata)
    _get_covariate(adata)
    _get_size(adata)
    _get_weight(adata)

    encode_regime(adata, "interv", key="interv")
    encode_regime(adata, "ctfact", key="ctfact")
    configure_dataset(adata, use_regime="interv")
    configure_dataset(adata, use_regime="ctfact")

    adata.obsm["covariate2"] = adata.obsm["covariate"]
    adata.obs["size2"] = adata.obs["size"]
    adata.obs["weight2"] = adata.obs["weight"]
    adata.layers["layer"] = adata.X
    adata.layers["layer2"] = adata.X
    configure_dataset(adata, use_covariate="covariate")
    configure_dataset(adata, use_covariate="covariate2")
    configure_dataset(adata, use_size="size")
    configure_dataset(adata, use_size="size2")
    configure_dataset(adata, use_weight="weight")
    configure_dataset(adata, use_weight="weight2")
    configure_dataset(adata, use_layer="layer")
    configure_dataset(adata, use_layer="layer2")

    _set_X(adata, _get_X(adata))
    _set_regime(adata, _get_regime(adata))
    _set_covariate(adata, _get_covariate(adata))
    _set_size(adata, _get_size(adata))
    _set_weight(adata, _get_weight(adata))


def test_neighbor_impute(adata):
    adata_impute1 = neighbor_impute(adata, 1, "covariate", use_batch="interv")
    adata_impute2 = neighbor_impute(adata, 5, "covariate", use_batch="interv")
    adata_impute3 = neighbor_impute(adata, 5, "covariate")
    assert array_close(adata_impute1.X, adata.X)
    assert not array_close(adata_impute2.X, adata.X)
    assert not array_close(adata_impute3.X, adata.X)
    assert not array_close(adata_impute2.X, adata_impute3.X)


def test_simple_design(adata):
    target = adata.copy()
    del target.obs["interv"]
    simple_design(adata, target, "interv")
