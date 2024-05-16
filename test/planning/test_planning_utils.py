import numpy as np

from pyroboplan.planning.graph import Node
from pyroboplan.planning.utils import (
    discretized_joint_space_generator,
    discretize_joint_space_path,
    extend_robot_state,
    retrace_path,
)


def test_extend_robot_state():
    q_parent = np.array([0.0, 0.0])
    q_sample = np.array([2.0, 0.0])
    max_connection_distance = 1.0

    # Extending to itself should do nothing
    q_extend = extend_robot_state(q_parent, q_parent, max_connection_distance)
    assert np.array_equal(q_extend, q_parent)

    # Otherwise extend 1 rad towards the sample
    q_extend = extend_robot_state(q_parent, q_sample, max_connection_distance)
    assert np.array_equal(q_extend, np.array([1.0, 0.0]))


def test_discretize_joint_space_path():
    q_start = np.array([0.0, 0.0])
    q_end = np.array([2.0, 0.0])
    max_angle_distance = 1.0

    # Starting from the same point should just return the point.
    path = discretize_joint_space_path(q_start, q_start, max_angle_distance)
    assert len(path) == 1
    assert np.array_equal(path, [q_start])

    # Discretize at 0.5 radians and verify the expected result.
    path = discretize_joint_space_path(q_start, q_end, max_angle_distance)
    expected_path = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([2.0, 0.0]),
    ]
    assert np.array_equal(path, expected_path)


def test_retrace_path():
    nodeA = Node([1])
    nodeB = Node([2])
    nodeC = Node([3])
    nodeB.parent = nodeA
    nodeC.parent = nodeB

    path = retrace_path(None)
    assert len(path) == 0

    # A node with no parent is a path unto itself.
    path = retrace_path(nodeA)
    assert len(path) == 1
    assert path == [nodeA]

    # Verify we get the expected path from A to C.
    path = retrace_path(nodeC)
    assert len(path) == 3
    assert path == [nodeA, nodeB, nodeC]


class DummyModel:
    def __init__(self, lower_limit, upper_limit):
        self.lowerPositionLimit = lower_limit
        self.upperPositionLimit = upper_limit


def test_discretized_joint_space_simple():
    # Discretize 1-DOF from 0 to 3 at step size 1.0.
    model = DummyModel(np.array([0]), np.array([3]))
    generator = discretized_joint_space_generator(model, 1, False)
    points = [x for x in generator]

    # Result should be [0, 1, 2, 3] in arrays.
    points_check = [np.array([x]) for x in range(4)]
    assert len(points) == 4
    assert np.array_equal(points, points_check)


def test_discretized_joint_space_random():
    model = DummyModel(np.array([0]), np.array([1]))
    generator = discretized_joint_space_generator(model, 1, True)

    assert next(generator) == np.array([0])
    assert next(generator) == np.array([1])

    # Once the space has been sampled it should continue returning random states.
    random = next(generator)
    assert random is not None
    assert random[0] > 0.0 and random[0] < 1.0
