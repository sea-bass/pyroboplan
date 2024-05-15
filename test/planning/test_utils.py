import numpy as np

from pyroboplan.planning.utils import discretized_joint_space_generator


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
