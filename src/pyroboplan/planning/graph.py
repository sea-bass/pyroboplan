import numpy as np


class Node:
    def __init__(self, q, parent=None, cost=0.0):
        """
        TODO: Creates a graph node
        """
        self.q = q
        self.parent = parent
        self.cost = cost
        self.neighbors = set()


def default_cost_fn(node_start, node_end):
    return np.linalg.norm(node_end.q - node_start.q)


class Edge:
    def __init__(self, nodeA, nodeB, cost_fn=default_cost_fn):
        self.nodeA = nodeA
        self.nodeB = nodeB
        self.cost = cost_fn(self.nodeA, self.nodeB)


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def add_node(self, node):
        self.nodes.add(node)

    def add_edge(self, nodeA, nodeB):
        self.edges.add(Edge(nodeA, nodeB))
        nodeA.neighbors.add(nodeB)
        nodeB.neighbors.add(nodeA)

    def get_nearest_node(self, q):
        nearest_node = None

        min_dist = np.inf
        for node in self.nodes:
            dist = np.linalg.norm(q - node.q)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node
