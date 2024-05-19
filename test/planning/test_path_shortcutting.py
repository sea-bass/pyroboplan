import numpy as np

from pyroboplan.core.utils import get_path_length
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.planning.path_shortcutting import shortcut_path


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

    q_shortened = shortcut_path(model, collision_model, q_path)
    print(
        f"Path shortened from {get_path_length(q_path)} to {get_path_length(q_shortened)}"
    )
    print(f"Direct path length = {get_path_length([q_path[0], q_path[-1]])}")
    # Check that the shortened path is shorter than original, but cannot be shorter than a direct path from start to end.
    assert get_path_length(q_shortened) <= get_path_length(q_path)
    assert get_path_length(q_shortened) >= get_path_length([q_path[0], q_path[-1]])


if __name__ == "__main__":
    test_path_shortcutting()
