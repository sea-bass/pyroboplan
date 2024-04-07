import numpy as np

from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)


# Use a fixed seed for random number generation in tests.
np.random.seed(1234)


def test_plan_trivial_rrt():
    model, collision_model, _ = load_models()

    # Initial joint states
    q_start = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])
    q_goal = q_start + 0.01

    # Plan with default parameters
    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_goal)

    assert path is not None
    assert len(path) == 2
    assert np.all(path[0] == q_start)
    assert np.all(path[1] == q_goal)


def test_plan_vanilla_rrt():
    model, collision_model, visual_model = load_models()
    add_self_collisions(collision_model)
    add_object_collisions(collision_model, visual_model)

    # Plan with default parameters
    q_start = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])
    q_goal = np.array([1.57, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_goal)

    # The path must exist and have more than
    assert path is not None
    assert len(path) > 2
    assert np.all(path[0] == q_start)
    assert np.all(path[-1] == q_goal)

    # Vanilla RRT should only grow the start tree, so we should check that the goal tree is just the goal node.
    assert len(planner.start_tree.nodes) > 1
    assert len(planner.start_tree.edges) > 1
    assert len(planner.goal_tree.nodes) == 1
    assert len(planner.goal_tree.edges) == 0


def test_plan_rrt_connect():
    model, collision_model, visual_model = load_models()
    add_self_collisions(collision_model)
    add_object_collisions(collision_model, visual_model)

    # Plan with RRTConnect and bidirectional RRT
    q_start = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])
    q_goal = np.array([1.57, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    options = RRTPlannerOptions()
    options.rrt_connect = True
    options.bidirectional_rrt = True

    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_goal, options=options)

    # The path must exist and have more than
    assert path is not None
    assert len(path) > 2
    assert np.all(path[0] == q_start)
    assert np.all(path[-1] == q_goal)

    # RRTConnect should have nodes and edges for both trees
    assert len(planner.start_tree.nodes) > 1
    assert len(planner.start_tree.edges) > 1
    assert len(planner.goal_tree.nodes) > 1
    assert len(planner.goal_tree.edges) > 1
