from pyroboplan.planning.graph import Node, Graph
from pyroboplan.planning.graph_search import dfs


def construct_graph(N):
    """Creates a connected NxN graph for testing."""
    g = Graph()

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            node = Node([i, j])
            g.add_node(node)

    for node in g.nodes:
        x, y = node.q[0], node.q[1]
        for px, py in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            neighbor = Node([px, py])
            if neighbor in g.nodes:
                g.add_edge(node, neighbor)

    return g


def test_dfs():
    g = construct_graph(5)

    start_pose = Node([1, 1])
    goal_pose = Node([5, 5])

    path = dfs(g, start_pose, goal_pose)

    assert path is not None
    assert path[0] == start_pose
    assert path[-1] == goal_pose
