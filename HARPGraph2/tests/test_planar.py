import pytest
import networkx as nx
from ..planarpdgraph import PlanarPDGraph

@pytest.mark.dependency()
def test_to_networkx_conversion():
    graph = PlanarPDGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('B', 'C'))
    graph.add_edge(('C', 'D'))
    graph.add_edge(('D', 'A'))
    graph.add_edge(('A', 'C'))

    nx_graph = graph._to_networkx()

    expected_graph = nx.Graph()
    expected_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')])

    assert nx.is_isomorphic(nx_graph, expected_graph), "The NetworkX graph does not match the expected graph."

@pytest.fixture
def planar_graph():
    graph = PlanarPDGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('B', 'C'))
    graph.add_edge(('C', 'D'))
    graph.add_edge(('D', 'A'))
    return graph

@pytest.fixture
def k5_graph():
    graph = PlanarPDGraph()
    graph.add_edge(('A', 'B'))
    graph.add_edge(('A', 'C'))
    graph.add_edge(('A', 'D'))
    graph.add_edge(('A', 'E'))
    graph.add_edge(('B', 'C'))
    graph.add_edge(('B', 'D'))
    graph.add_edge(('B', 'E'))
    graph.add_edge(('C', 'D'))
    graph.add_edge(('C', 'E'))
    graph.add_edge(('D', 'E'))
    return graph

@pytest.fixture
def k33_graph():
    graph = PlanarPDGraph()
    graph.add_edge(('A', 'D'))
    graph.add_edge(('A', 'E'))
    graph.add_edge(('A', 'F'))
    graph.add_edge(('B', 'D'))
    graph.add_edge(('B', 'E'))
    graph.add_edge(('B', 'F'))
    graph.add_edge(('C', 'D'))
    graph.add_edge(('C', 'E'))
    graph.add_edge(('C', 'F'))
    return graph


@pytest.mark.dependency(depends=["test_to_networkx_conversion"])
def test_k5_detection(k5_graph):
    assert k5_graph._has_k5() == True, "Expected k5_graph to contain a K5 subgraph"

@pytest.mark.dependency(depends=["test_k5_detection"])
def test_no_k5_in_planar_graph(planar_graph):
    assert planar_graph._has_k5() == False, "Expected planar_graph to not contain a K5 subgraph"

@pytest.mark.dependency(depends=["test_to_networkx_conversion"])
def test_k33_detection(k33_graph):
    assert k33_graph._has_k33() == True, "Expected k33_graph to contain a K33 subgraph"

@pytest.mark.dependency(depends=["test_k33_detection"])
def test_no_k33_in_planar_graph(planar_graph):
    assert planar_graph._has_k33() == False, "Expected planar_graph to not contain a K33 subgraph"

@pytest.mark.dependency(depends=["test_k33_detection"])
def test_is_not_planar_k33(k33_graph):
    assert k33_graph.is_planar() == False, "Expected k33_graph to be non-planar due to K33 subgraph"

@pytest.mark.dependency(depends=["test_to_networkx_conversion"])
def test_get_internal_faces(planar_graph):
    faces = planar_graph.get_internal_faces()
    expected_faces = [['A', 'B', 'C', 'D']]
    assert all(set(face) in [set(exp_face) for exp_face in expected_faces] for face in faces), f"Expected faces {expected_faces}, but got {faces}"

@pytest.mark.dependency(depends=["test_to_networkx_conversion"])
def test_is_bipartite(planar_graph, k33_graph):
    assert planar_graph.is_bipartite() == True, "Expected planar_graph to be bipartite"
    assert k33_graph.is_bipartite() == True, "Expected k33_graph to be bipartite"

@pytest.mark.dependency(depends=["test_is_bipartite"])
def test_is_not_bipartite(k5_graph):
    assert k5_graph.is_bipartite() == False, "Expected k5_graph to be non-bipartite"


@pytest.mark.dependency(depends=["test_to_networkx_conversion"])
def test_is_planar(planar_graph):
    assert planar_graph.is_planar() == True, "Expected planar_graph to be planar"

@pytest.mark.dependency(depends=["test_is_planar"])
def test_is_not_planar_k5(k5_graph):
    assert k5_graph.is_planar() == False, "Expected k5_graph to be non-planar due to K5 subgraph"
