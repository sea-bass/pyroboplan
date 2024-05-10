def reconstruct_path(goal):
    """Constructs a path back to the start node from the goal node."""
    path = []
    current = goal
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def dfs(graph, start_node, goal_node):
    """
    Find a path between the start_pose and goal_pose using a depth first search.

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
    if not start_node or not goal_node:
        raise ValueError("Nodes must be present in graph to execute pathfinding.")

    # Mark the start node as the beginning
    stack = [start_node]
    start_node.parent = None

    visited = set()
    while stack:
        current = stack.pop()
        if current == goal_node:
            return reconstruct_path(current)

        if current not in visited:
            visited.add(current)
            for neighbor in current.neighbors:
                if neighbor not in visited:
                    # Update the path and mark it to visit
                    neighbor.parent = current
                    stack.append(neighbor)

    return None
