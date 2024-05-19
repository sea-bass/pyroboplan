import copy
import numpy as np

from pyroboplan.core.utils import configuration_distance, get_path_length
from pyroboplan.planning.utils import (
    check_collisions_along_path,
    discretize_joint_space_path,
)


def get_normalized_path_scaling(q_path):
    """Helper function"""
    path_length = 0.0
    path_length_list = [path_length]
    for idx in range(len(q_path) - 1):
        path_length += configuration_distance(q_path[idx], q_path[idx + 1])
        path_length_list.append(path_length)
    return np.array(path_length_list) / path_length


def get_configuration_from_normalized_path_scaling(q_path, path_lengths, value):
    """Helper function"""
    if value < 0 or value > 1:
        raise ValueError("Scaling value must be in the range [0, 1]")

    for idx in range(len(path_lengths)):
        if value > path_lengths[idx]:
            continue

        delta_scale = (value - path_lengths[idx - 1]) / (
            path_lengths[idx] - path_lengths[idx - 1]
        )
        return q_path[idx - 1] + delta_scale * (q_path[idx] - q_path[idx - 1]), idx


def shortcut_path(
    model, collision_model, q_path, max_iters=100, num_restarts=1, max_angle_step=0.05
):
    """
    Does path shortcutting wow

    Some good sources:
      * Section 3.5.3 in https://motion.cs.illinois.edu/RoboticSystems/MotionPlanningHigherDimensions.html
    """
    best_path = q_path
    best_path_length = get_path_length(best_path)

    for _ in range(num_restarts + 1):
        q_shortened = copy.deepcopy(q_path)
        for _ in range(max_iters):
            # Sample two points along the path length
            path_lengths = get_normalized_path_scaling(q_shortened)
            while True:
                low_point = np.random.random()
                high_point = np.random.random()
                if low_point > high_point:
                    low_point, high_point = high_point, low_point
                q_low, idx_low = get_configuration_from_normalized_path_scaling(
                    q_shortened, path_lengths, low_point
                )
                q_high, idx_high = get_configuration_from_normalized_path_scaling(
                    q_shortened, path_lengths, high_point
                )
                if idx_low < idx_high:
                    break

            # Check if the points are collision free. If they are, shortcut the path.
            path_to_goal = discretize_joint_space_path([q_low, q_high], max_angle_step)
            if not check_collisions_along_path(model, collision_model, path_to_goal):
                q_shortened = (
                    q_shortened[:idx_low] + [q_low, q_high] + q_shortened[idx_high:]
                )

        # Check if this is the best path so far
        if get_path_length(q_shortened) < best_path_length:
            best_path = copy.deepcopy(q_shortened)
            best_path_length = get_path_length(q_shortened)

    return best_path
