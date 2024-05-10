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
        self.neighbors = set()

    def __hash__(self):
        """Hash on joint configurations only."""
        return hash(self.q.tobytes())

    def __eq__(self, other):
        """A node is equal to another node iff their joint configurations are equal."""
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

    def __hash__(self):
        """Compute the hash of this edge based on nodes and cost."""
        return hash((self.nodeA, self.nodeB, self.cost))

    def __eq__(self, other):
        """An edge is equal to the other iff it has the same endpoints and cost, in order."""
        return (
            self.nodeA == other.nodeA
            and self.nodeB == other.nodeB
            and self.cost == other.cost
        )

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
        """
        node = Node(q)
        if node in self.nodes:
            return self.nodes[node]
        else:
            return None

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
        nodeA.neighbors.add(nodeB)
        nodeB.neighbors.add(nodeA)
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

        self.nodes[edge.nodeA].neighbors.remove(edge.nodeB)
        self.nodes[edge.nodeB].neighbors.remove(edge.nodeA)
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

    def save_to_file(self, filename):
        """
        Write the graph to file.

        Only node and edge data is persisted. Neighbors can be rederived, but all parent information is lost.
        """
        with open(filename, "w") as file:
            file.writelines([str(node) + "\n" for node in self.nodes])
            file.writelines([str(edge) + "\n" for edge in self.edges])

    @classmethod
    def load_from_file(cls, filename):
        """Loads a graph from a file."""
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
