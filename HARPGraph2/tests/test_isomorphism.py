import pytest
from ..mixeddigraph import MixedDiGraph

@pytest.fixture
def graph1():
    graph = MixedDiGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('B', 'C'))
    graph.add_edge(('C', 'A'))
    return graph

@pytest.fixture
def graph2():
    graph = MixedDiGraph()
    graph.add_edge(('X', 'Y'))
    graph.add_edge(('Y', 'Z'))
    graph.add_edge(('Z', 'X'))
    return graph

@pytest.fixture
def graph3():
    graph = MixedDiGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('B', 'C'))
    return graph

@pytest.fixture
def graph4():
    graph = MixedDiGraph()
    graph.add_edge(('X', 'Y'))
    graph.add_edge(('Y', 'Z'))
    return graph

@pytest.fixture
def graph5():
    graph = MixedDiGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('B', 'C'))
    graph.add_edge(('C', 'D'))
    graph.add_edge(('D', 'A'))
    return graph

@pytest.fixture
def graph6():
    graph = MixedDiGraph()
    graph.add_edge(('X', 'Y'))
    graph.add_edge(('Y', 'Z'))
    graph.add_edge(('Z', 'W'))
    graph.add_edge(('W', 'X'))
    return graph

def test_isomorphic_graphs(graph1, graph2):
    assert graph1.is_isomorphic(graph2) == True

def test_non_isomorphic_graphs(graph1, graph3):
    assert graph1.is_isomorphic(graph3) == False

def test_isomorphic_subgraphs(graph3, graph4):
    assert graph3.is_isomorphic(graph4) == True

def test_isomorphic_larger_graphs(graph5, graph6):
    assert graph5.is_isomorphic(graph6) == True

def test_non_isomorphic_larger_graphs(graph5, graph1):
    assert graph5.is_isomorphic(graph1) == False
