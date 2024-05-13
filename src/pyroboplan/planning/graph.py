""" Generic graph class for use in robot motion planning algorithms. """

import numpy as np
import ast
import re

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
        self.q = np.array(q)
        self.parent = parent
        self.cost = cost

        # Dictionary of neighboring nodes and their distances
        self.neighbors = {}

    def __hash__(self):
        """Hash on joint configurations only."""
        return hash(self.q.tobytes())

    def __eq__(self, other):
        """A node is equal to another node if and only if their joint configurations are equal."""
        return np.array_equal(self.q, other.q)

    def __str__(self):
        """Return a string representation of the node that includes joint configuration and cost."""
        return f"Node(q={self.q.tolist()}, cost={self.cost})"

    @staticmethod
    def parse(s):
        """Reconstruct a Node object from its string representation."""
        pattern = r"Node\(q=(.*?), cost=(.*?)\)"
        match = re.match(pattern, s)
        if match:
            q = ast.literal_eval(match.group(1))
            cost = float(match.group(2))
            return Node(q, None, cost)
        raise ValueError(f"Failed to reconstruct node from string {s}.")


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

    def __str__(self):
        """Return a string representation of the edge."""
        return f"Edge(nodeA=({self.nodeA}), nodeB=({self.nodeB}), cost={self.cost})"

    @staticmethod
    def parse(s):
        """Reconstruct an Edge object from its string representation."""
        pattern = r"Edge\(nodeA=\((.*?)\), nodeB=\((.*?)\), cost=(.*?)\)"
        match = re.match(pattern, s)
        if match:
            nodeA = Node.parse(match.group(1))
            nodeB = Node.parse(match.group(2))
            cost = float(match.group(3))
            return Edge(nodeA, nodeB, cost)
        raise ValueError(f"Failed to reconstruct edge from string {s}.")


class Graph:
    """Describes a graph of robot configuration states for motion planning."""

    def __init__(self):
        """
        Creates a graph instance with no edges or nodes.
        """
        self.nodes = {}
        self.edges = set()

    def add_node(self, node):
        """
        Adds a node to the graph.

        Parameters
        ----------
            node : `pyroboplan.planning.graph.Node`
                The node to add to the graph.
        """
        if node in self.nodes:
            return

        self.nodes[node] = node

    def get_node(self, q):
        """
        Returns the node corresponding to the provided robot configuration, or None if not present.

        Raises a `ValueError` if the specified configuration is not in the Graph.

        Parameters
        ----------
            q : array-like
                The robot configuration for the desired node.

        Returns
        -------
            `pyroboplan.planning.graph.Node`
                The node with the specified robot configuration.
        """
        node = Node(q)
        if node not in self.nodes:
            raise ValueError("Specified robot configuration not found in Graph.")

        return self.nodes[node]

    def add_edge(self, nodeA, nodeB):
        """
        Adds an edge to the graph.

        Parameters
        ----------
            nodeA : `pyroboplan.planning.graph.Node`
                The first node in the edge.
            nodeB : `pyroboplan.planning.graph.Node`
                The second node in the edge.

        Returns
        -------
            `pyroboplan.planning.graph.Edge`
                The edge that was added.
        """
        if nodeA not in self.nodes or nodeB not in self.nodes:
            raise ValueError("Specified nodes are not in Graph, cannot add edge.")
        nodeA = self.nodes[nodeA]
        nodeB = self.nodes[nodeB]

        cost = configuration_distance(nodeA.q, nodeB.q)
        edge = Edge(nodeA, nodeB, cost)
        self.edges.add(edge)
        nodeA.neighbors[nodeB] = cost
        nodeB.neighbors[nodeA] = cost
        return edge

    def remove_edge(self, edge):
        """
        Attempts to remove an edge from a graph, if it exists.

        Parameters
        ----------
            edge : `pyroboplan.planning.graph.Edge`
                The edge to remove from the graph.

        Returns
        -------
            bool
                True if the edge was successfully removed, else False.

        """
        if edge not in self.edges:
            return False

        del self.nodes[edge.nodeA].neighbors[edge.nodeB]
        del self.nodes[edge.nodeB].neighbors[edge.nodeA]
        self.edges.remove(edge)
        return True

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

    def get_nearest_neighbors(self, q, radius):
        """
        Gets a list of the nearest neighbors to the specified robot configuration.

        Neighboring nodes will be returned as a sorted list of tuples of nodes along with the distance
        to the specified configuration.

        Parameters
        ----------
            q : array-like
                The robot configuration to use in this query.
            radius: float
                The maximum radius in which to consider neighbors

        Returns
        -------
            List of (`float`, `pyroboplan.planning.graph.Node`)
                The set of all n
        """
        neighbors = []
        chk_node = Node(q)
        for node in self.nodes:
            if node == chk_node:
                continue
            d = configuration_distance(node.q, chk_node.q)
            if d <= radius:
                neighbors.append((node, d))
        return sorted(neighbors, key=lambda n: n[1])

    def save_to_file(self, filename):
        """
        Write the graph to file.

        Only node and edge data is persisted. Neighbors can be rederived, but all parent information is lost.

        Parameters
        ----------
            filename : str
                The full filepath in which to save the Graph.
        """
        with open(filename, "w") as file:
            file.writelines([str(node) + "\n" for node in self.nodes])
            file.writelines([str(edge) + "\n" for edge in self.edges])

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads a graph from a file.

        Parameters
        ----------
            filename : str
                The full filepath from which to read a Graph.

        Returns
        -------
            `pyroboplan.planning.graph.Graph`
                The reconstructed Graph from file.
        """
        g = cls()
        with open(filename, "r") as file:
            for line in file:
                if line.startswith("Node"):
                    node = Node.parse(line.strip())
                    g.add_node(node)
                elif line.startswith("Edge"):
                    edge = Edge.parse(line.strip())
                    g.add_edge(edge.nodeA, edge.nodeB)
        return g
