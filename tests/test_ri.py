import networkx as nx
from pytest import mark


@mark.use_ri
def test_dag2essgraph(dag):
    from cascade.ri import dag2essgraph

    essgraph = dag2essgraph(dag)
    assert essgraph.number_of_nodes() == dag.number_of_nodes()
    assert essgraph.number_of_edges() >= dag.number_of_edges()


@mark.use_ri
def test_structIntervDist(dag):
    from cascade.ri import structIntervDist

    true = nx.to_scipy_sparse_array(dag, weight=None)
    pred = nx.to_scipy_sparse_array(dag, weight=None)
    assert structIntervDist(true, pred) == 0


@mark.use_ri
def test_pc(adata, dg):
    from cascade.ri import pc

    digraph = pc(adata, nx.Graph(dg), alpha=0.5, verbose=True)
    assert set(digraph.nodes) == set(adata.var_names)


@mark.use_ri
def test_ges(adata, dg):
    from cascade.ri import ges

    digraph = ges(adata, nx.Graph(dg), verbose=True)
    assert set(digraph.nodes) == set(adata.var_names)


@mark.use_ri
def test_gies(adata, dg):
    from cascade.ri import gies

    digraph = gies(adata, "interv", nx.Graph(dg), verbose=True)
    assert set(digraph.nodes) == set(adata.var_names)
