import numpy as np
from pytest import raises
from scipy.sparse import csr_matrix

from cascade.utils import (
    autodevice,
    covariance,
    get_random_state,
    hclust,
    partial_correlation,
    pearson_correlation,
    spearman_correlation,
    str_to_bool,
    variance,
)

from .utils import array_close


def test_str_to_bool():
    assert str_to_bool("T")
    assert str_to_bool("True")
    assert str_to_bool("true")
    assert str_to_bool("1")
    assert not str_to_bool("F")
    assert not str_to_bool("False")
    assert not str_to_bool("false")
    assert not str_to_bool("0")
    with raises(ValueError):
        str_to_bool("none")


def test_get_random_state():
    get_random_state()
    get_random_state(None)
    get_random_state(0)
    get_random_state(np.random.RandomState(0))


def test_variance(adata):
    var1 = variance(adata.X)
    var2 = variance(csr_matrix(adata.X))
    assert array_close(var1, var2)


def test_covariance(adata):
    cov1 = covariance(adata.X)
    cov2 = covariance(csr_matrix(adata.X))
    assert array_close(cov1, cov2)


def test_pearson_correlation(adata):
    cor1 = pearson_correlation(adata.X)
    cor2 = pearson_correlation(csr_matrix(adata.X))
    assert array_close(cor1, cor2)


def test_spearman_correlation(adata):
    cor1 = spearman_correlation(adata.X)
    cor2 = spearman_correlation(csr_matrix(adata.X))
    assert array_close(cor1, cor2)


def test_partial_correlation(adata):
    pcor1 = partial_correlation(adata.X)
    pcor2 = partial_correlation(csr_matrix(adata.X))
    assert array_close(pcor1, pcor2)


def test_hclust(adata):
    hclust(adata.to_df())


def test_autodevice():
    autodevice(2)
    autodevice(1)
    autodevice(0)
