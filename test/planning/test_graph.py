import numpy as np
import os
import pytest
import tempfile

from pyroboplan.planning.graph import Node, Edge, Graph


def test_create_node():
    node = Node(np.array([1.0, 2.0, 3.0]))

    assert np.all(node.q == np.array([1.0, 2.0, 3.0]))
    assert node.parent is None
    assert node.cost == 0.0
    assert len(node.neighbors) == 0


def test_create_node_nondefault_args():
    parent_node = Node([1.0, 2.0, 3.0])
    node = Node([4.0, 5.0, 6.0], parent=parent_node, cost=42.0)

    assert np.all(node.q == np.array([4.0, 5.0, 6.0]))
    assert node.parent == parent_node
    assert np.all(node.parent.q == np.array([1.0, 2.0, 3.0]))
    assert node.cost == 42.0
    assert len(node.neighbors) == 0


def test_create_edge():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])
    edge = Edge(nodeA, nodeB)

    assert edge.nodeA == nodeA
    assert edge.nodeB == nodeB
    assert edge.cost == 0.0


def test_node_equalities():
    nodeA = Node([1.0, 1.0], 1.0)
    nodeB = Node([1.0, 1.0], 1.0)
    nodeC = Node([2.0, 2.0], 2.0)
    assert nodeA == nodeB
    assert nodeA != nodeC


def test_create_edge_nondefault_args():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])
    edge = Edge(nodeA, nodeB, cost=21.0)

    assert edge.nodeA == nodeA
    assert edge.nodeB == nodeB
    assert edge.cost == 21.0


def test_create_empty_graph():
    graph = Graph()
    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_add_nodes():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])
    graph = Graph()

    # Asking for a non-existent node will throw an exception
    with pytest.raises(ValueError):
        graph.get_node(nodeA)

    graph.add_node(nodeA)
    assert len(graph.nodes) == 1

    graph.add_node(nodeB)
    assert len(graph.nodes) == 2

    # Adding the same node again does not count as a repeat since the underlying storage is a set.
    nodeA_dup = Node([1.0, 2.0, 3.0])
    graph.add_node(nodeA_dup)
    assert len(graph.nodes) == 2

    assert len(graph.edges) == 0


def test_add_edges():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])
    nodeC = Node([7.0, 8.0, 9.0])
    graph = Graph()
    graph.add_node(nodeA)
    graph.add_node(nodeB)
    graph.add_node(nodeC)

    edgeAB = graph.add_edge(nodeA, nodeB)
    assert len(graph.edges) == 1
    assert nodeB in nodeA.neighbors
    assert nodeA in nodeB.neighbors
    assert edgeAB.cost == pytest.approx(np.linalg.norm([3.0, 3.0, 3.0]))

    # Adding the same edge should not modify the graph.
    edgeAB = graph.add_edge(nodeA, nodeB)
    assert len(graph.edges) == 1

    edgeAC = graph.add_edge(nodeA, nodeC)
    assert len(graph.edges) == 2
    assert nodeC in nodeA.neighbors
    assert nodeA in nodeC.neighbors
    assert edgeAC.cost == pytest.approx(np.linalg.norm([6.0, 6.0, 6.0]))


def test_remove_edges():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])
    graph = Graph()
    graph.add_node(nodeA)
    graph.add_node(nodeB)
    edgeAB = graph.add_edge(nodeA, nodeB)
    assert len(graph.edges) == 1

    # Remove a valid edge.
    assert graph.remove_edge(edgeAB)
    assert len(graph.edges) == 0
    assert nodeB not in nodeA.neighbors
    assert nodeA not in nodeB.neighbors

    # Remove an edge that was already removed.
    assert not graph.remove_edge(edgeAB)
    assert len(graph.edges) == 0


def test_get_nearest_node():
    nodeA = Node([1.0, 2.0, 3.0])
    nodeB = Node([4.0, 5.0, 6.0])

    graph = Graph()
    graph.add_node(nodeA)
    graph.add_node(nodeB)

    nearest_node = graph.get_nearest_node([5.0, 4.0, 3.0])
    assert nearest_node == nodeB


def test_get_nearest_node_empty_graph():
    graph = Graph()
    nearest_node = graph.get_nearest_node([1.0, 2.0, 3.0])
    assert nearest_node is None


def test_get_nearest_neighbors():
    graph = Graph()
    neighbors = graph.get_nearest_neighbors([1.0], radius=1.0)

    assert len(neighbors) == 0

    nodeA = Node([0.0])
    nodeB = Node([1.0])
    nodeC = Node([3.0])
    nodeD = Node([4.0])

    graph.add_node(nodeA)
    graph.add_node(nodeB)
    graph.add_node(nodeC)
    graph.add_node(nodeD)

    neighbors = graph.get_nearest_neighbors([1.0], radius=2.0)
    assert len(neighbors) == 2
    assert neighbors[0] == (nodeA, 1.0)
    assert neighbors[1] == (nodeC, 2.0)


def test_str_and_parse():
    nodeA = Node([1.0, 2.0, 3.0], cost=1.0)
    nodeB = Node([4.0, 5.0, 6.0], cost=1.0)
    edgeAB = Edge(nodeA, nodeB)

    sA = str(nodeA)
    sB = str(nodeB)
    eAB = str(edgeAB)

    assert sA != sB
    assert nodeA == Node.parse(sA)
    assert nodeA != Node.parse(sB)

    chk_edge = Edge.parse(eAB)
    assert nodeA == chk_edge.nodeA
    assert nodeB == chk_edge.nodeB


def test_save_and_load():
    nodeA = Node([1.0, 1.0], cost=1.0)
    nodeB = Node([1.0, 2.0], cost=1.0)
    g = Graph()
    g.add_node(nodeA)
    g.add_node(nodeB)
    g.add_edge(nodeA, nodeB)

    # Construct simple graph and save to tempfile.
    f = os.path.join(tempfile.mkdtemp(), "graph")
    g.save_to_file(f)

    # Assert that we get the same graph back.
    g_check = Graph.load_from_file(f)
    assert len(g_check.nodes) == 2
    assert len(g_check.edges) == 1
    assert nodeA in g_check.nodes
    assert nodeB in g_check.nodes
