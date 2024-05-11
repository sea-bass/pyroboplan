""" Implementations of graph search algorithms for planning applications. """

import numpy as np

from heapq import heappop, heappush

from pyroboplan.core.utils import configuration_distance
from pyroboplan.planning.utils import retrace_path


def dfs(graph, start_node, goal_node):
    """
    Find a path between the `start_pose` and `goal_pose` using a depth first search (DFS).

    Both the start and goal poses must be present in the Graph.

    Parameters
    ----------
    graph : `pyroboplan.planning.graph.Graph`
        The graph containing the nodes and edges.
    start_pose : `pyroboplan.planning.graph.Node`
        The starting robot configuration.
    goal_pose : `pyroboplan.planning.graph.Node`
        The goal robot configuration.

    Returns
    -------
    path : list of `pyroboplan.planning.graph.Node`
        The sequence of nodes representing a path from start_node to goal_node.
    """
    # Ensure the requested configurations are present
    start_node = graph.get_node(start_node.q)
    goal_node = graph.get_node(goal_node.q)

    # Mark the start node as the beginning
    stack = [start_node]
    start_node.parent = None
    start_node.cost = 0.0

    visited = set()
    while stack:
        current = stack.pop()
        if current == goal_node:
            return retrace_path(current)

        if current not in visited:
            visited.add(current)
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    # Update the path and mark it to visit
                    neighbor.parent = current
                    neighbor.cost = current.cost + current.neighbors[neighbor]
                    stack.append(neighbor)

    # If we reach this point, then we have visited all nodes that are reachable from the start
    # node without finding a path, so no path exists.
    return None


def astar(
    graph,
    start_node,
    goal_node,
    heuristic=lambda n1, n2: configuration_distance(n1.q, n2.q),
):
    """
    Finds the shortest path between the start_pose and goal_pose using the A* algorithm.

    Both the start and goal poses must be present in the Graph.

    Parameters
    ----------
    graph : `pyroboplan.planning.graph.Graph`
        The graph containing the nodes and edges.
    start_pose : `pyroboplan.planning.graph.Node`
        The starting robot configuration.
    goal_pose : `pyroboplan.planning.graph.Node`
        The goal robot configuration.
    heuristic : function(`pyroboplan.planning.graph.Node`, `pyroboplan.planning.graph.Node`)
        Heuristic function for estimating the distance between two nodes. For A* this will be
        used to compute the configuration space distance between a visited node (arg1) and the
        goal node (arg2). Defaults to :func:`pyroboplan.core.utils.configuration_distance`.

    Returns
    -------
    path : list of `pyroboplan.planning.graph.Node`
        The sequence of nodes representing a path from start_node to goal_node.
    """
    # Ensure the requested configurations are present
    start_node = graph.get_node(start_node.q)
    goal_node = graph.get_node(goal_node.q)

    # Store nodes as a tuple with cost as the first element, heapq will automatically
    # sort based on the first element. We use a counter as the second element to avoid
    # having to rely on node values for tie-breaking in pushes.
    open_set_nodes = set()  # For faster element lookup
    open_set = []
    counter = 0
    heappush(open_set, (0, counter, start_node))
    open_set_nodes.add(start_node)

    # Initialize cost dictionaries
    g_scores = {start_node: 0}
    f_scores = {start_node: heuristic(start_node, goal_node)}

    while open_set:
        # Grab the estimated best node
        _, _, current = heappop(open_set)
        open_set_nodes.remove(current)

        # Then we're done
        if current == goal_node:
            return retrace_path(goal_node)

        for neighbor in current.neighbors:
            # If we haven't visited then the assumed cost is infinity
            if neighbor not in g_scores:
                g_scores[neighbor] = np.inf

            # Check is this is a better parent
            tentative_g_score = g_scores[current] + current.neighbors[neighbor]
            if tentative_g_score < g_scores[neighbor]:
                neighbor.parent = current
                g_scores[neighbor] = tentative_g_score
                neighbor.cost = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal_node)
                f_scores[neighbor] = f_score
                counter += 1
                if neighbor not in open_set_nodes:
                    open_set_nodes.add(neighbor)
                    heappush(open_set, (f_score, counter, neighbor))

    # If we reach this point, then we have visited all nodes that are reachable from the start
    # node without finding a path, so no path exists.
    return None
