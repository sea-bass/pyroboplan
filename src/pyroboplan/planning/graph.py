import numpy as np

from ..core.utils import configuration_distance


class Node:
    """Describes an individual node in a graph."""

    def __init__(self, q, parent=None, cost=0.0):
        """
        Creates a graph node instance.

        Parameters
        ----------
            q : array-like
                The joint configuration associated with the node.
            parent : `pyroboplan.planning.graph.Node`, optional.
                The parent of the node, which is either None or another Node object.
            cost : float, optional
                The cost associated with the node.
        """
        self.q = q
        self.parent = parent
        self.cost = cost
        self.neighbors = set()


class Edge:
    """Describes an individual edge in a graph."""

    def __init__(self, nodeA, nodeB, cost=0.0):
        """
        Creates a graph edge instance.

        Parameters
        ----------
            nodeA : `pyroboplan.planning.graph.Node`
                The first node in the edge.
            nodeB : `pyroboplan.planning.graph.Node`
                The second node in the edge.
            cost : float, optional
                The cost associated with the edge.
        """
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.cost = cost


class Graph:
    """Describes a graph of robot configuration states for motion planning."""

    def __init__(self):
        """
        Creates a graph instance with no edges or nodes.
        """
        self.nodes = set()
        self.edges = set()

    def add_node(self, node):
        """
        Adds a node to the graph.

        Parameters
        ----------
            node : `pyroboplan.planning.graph.Node`
                The node to add to the graph.
        """
        self.nodes.add(node)

    def add_edge(self, nodeA, nodeB):
        """
        Adds an edge to the graph.

        Parameters
        ----------
            nodeA : `pyroboplan.planning.graph.Node`
                The first node in the edge.
            nodeB : `pyroboplan.planning.graph.Node`
                The second node in the edge.
        """
        cost = configuration_distance(nodeA.q, nodeB.q)
        self.edges.add(Edge(nodeA, nodeB, cost))
        nodeA.neighbors.add(nodeB)
        nodeB.neighbors.add(nodeA)

    def get_nearest_node(self, q):
        """
        Gets the nearest node to a specified robot configuration.

        Parameters
        ----------
            q : array-like
                The robot configuration to use in this query.

        Returns
        -------
            `pyroboplan.planning.graph.Node` or None
                The nearest node to the input configuration, or None if the graph is empty.
        """
        nearest_node = None
        min_dist = np.inf

        for node in self.nodes:
            dist = configuration_distance(q, node.q)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node
