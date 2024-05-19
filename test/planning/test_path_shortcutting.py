import numpy as np
import pytest

from pyroboplan.core.utils import get_path_length
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.planning.path_shortcutting import (
    get_normalized_path_scaling,
    get_configuration_from_normalized_path_scaling,
    shortcut_path,
)


def test_get_normalized_path_scaling():
    # Trivial empty path case
    assert get_normalized_path_scaling([]) == []

    # Trivial single-element path case
    assert get_normalized_path_scaling([np.array([1.0, 2.0, 3.0])]) == [1.0]

    # Nontrivial and realistic case with a multi-waypoint path
    scaling = get_normalized_path_scaling(
        [
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 0.0]),
            np.array([1.0, 2.0, -1.0]),
        ]
    )
    assert scaling == pytest.approx([0.0, 0.25, 0.75, 1.0])


def test_get_configuration_from_normalized_path_scaling():
    path = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([1.0, 2.0, -1.0]),
    ]
    scaling = get_normalized_path_scaling(path)

    # Invalid values
    with pytest.raises(ValueError):
        get_configuration_from_normalized_path_scaling(path, scaling, -1.0)
        get_configuration_from_normalized_path_scaling(path, scaling, 2.0)

    # Start and end of the path
    q, idx = get_configuration_from_normalized_path_scaling(path, scaling, 0.0)
    assert q == pytest.approx(path[0])
    assert idx == 0

    q, idx = get_configuration_from_normalized_path_scaling(path, scaling, 1.0)
    assert q == pytest.approx(path[-1])
    assert idx == len(path) - 1

    # Points along the path
    q, idx = get_configuration_from_normalized_path_scaling(path, scaling, 0.25)
    assert q == pytest.approx(path[1])
    assert idx == 1

    q, idx = get_configuration_from_normalized_path_scaling(path, scaling, 0.5)
    assert q == pytest.approx(0.5 * (path[1] + path[2]))
    assert idx == 2


def test_path_shortcutting():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)

    # Define a multi-configuration path
    q_path = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.00, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0]),
        np.array([0.785, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0]),
        np.array([0.785, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, -0.785]),
    ]

    # Check that the shortened path is shorter than original, but cannot be shorter than a direct path from start to end.
    q_shortened = shortcut_path(model, collision_model, q_path)
    assert get_path_length(q_shortened) <= get_path_length(q_path)
    assert get_path_length(q_shortened) >= get_path_length([q_path[0], q_path[-1]])
