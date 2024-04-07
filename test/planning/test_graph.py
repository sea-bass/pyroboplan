import numpy as np
import pytest

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

    graph.add_node(nodeA)
    assert len(graph.nodes) == 1

    graph.add_node(nodeB)
    assert len(graph.nodes) == 2

    # Adding the same node again does not count as a repeat since the underlying storage is a set.
    graph.add_node(nodeA)
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

    edgeAC = graph.add_edge(nodeA, nodeC)
    assert len(graph.edges) == 2
    assert nodeC in nodeA.neighbors
    assert nodeA in nodeC.neighbors
    assert edgeAC.cost == pytest.approx(np.linalg.norm([6.0, 6.0, 6.0]))


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
