import copy
import numpy as np

from pyroboplan.core.utils import configuration_distance, get_path_length
from pyroboplan.planning.utils import (
    check_collisions_along_path,
    discretize_joint_space_path,
)


def get_normalized_path_scaling(q_path):
    """
    Returns a list of length-normalized scaling values along a joint configuration path,
    from 0.0 at the start to 1.0 at the end.

    Parameters
    ----------
        q_path : list[array-like]
            The list of joint configurations describing the path.

    Return
    ------
        list[float]
            The list of scaling values, between 0.0 and 1.0, at each of the path waypoints.
    """
    if len(q_path) == 0:
        return []
    if len(q_path) == 1:
        return [1.0]

    path_length = 0.0
    path_length_list = [path_length]
    for idx in range(len(q_path) - 1):
        path_length += configuration_distance(q_path[idx], q_path[idx + 1])
        path_length_list.append(path_length)
    return np.array(path_length_list) / path_length


def get_configuration_from_normalized_path_scaling(q_path, path_scalings, value):
    """
    Given a path and its corresponding length-normalized scalings between 0.0 and 1.0,
    get the joint configuration at a specific scale value.

    For example:
      * 0.0 will return the start configuration of the path
      * 1.0 will return the end configuration of the path
      * 0.5 will return the configuration halfway along the length of the path

    Parameters
    ----------
        q_path : list[array-like]
            The list of joint configurations describing the path.
        path_scalings : list[float]
            The list of scaling values, between 0.0 and 1.0, at each of the path waypoints.
        value : float
            The value, between 0.0 and 1.0, at which to get the joint configuration along the path.

    Return
    ------
        tuple (array-like, int)
            A tuple containing the joint configuration at the specified scaling value along the path,
            as well as the index corresponding to the next point along the path.
    """
    if value < 0 or value > 1:
        raise ValueError("Scaling value must be in the range [0, 1]")

    for idx in range(len(path_scalings)):
        if value > path_scalings[idx]:
            continue

        delta_scale = (value - path_scalings[idx - 1]) / (
            path_scalings[idx] - path_scalings[idx - 1]
        )
        return q_path[idx - 1] + delta_scale * (q_path[idx] - q_path[idx - 1]), idx


def shortcut_path(
    model, collision_model, q_path, max_iters=100, num_restarts=2, max_angle_step=0.05
):
    """
    Performs path shortcutting by sampling two random points along the path and verifying
    whether those two points can be connected directly without collisions.

    This implementation is based on section 3.5.3 of:
    https://motion.cs.illinois.edu/RoboticSystems/MotionPlanningHigherDimensions.html

    Parameters
    ----------
        model : `pinocchio.Model`
            The model of the robot.
        collision_model : `pinocchio.Model`.
            The model to use for collision checking.
        q_path : list[array-like]
            The list of joint configurations describing the path.
        max_iters : int
            The maximum iterations for randomly sampling shortcut points.
        num_restarts : int
            The number of restarts to try different random path shortcutting solutions.
        max_angle_step : float
            Maximum angle step, in radians, for collision checking along path segments.

    Return
    ------
        list[array-like]
            The shortest length path over all the shortcutting attempts.
    """
    best_path = q_path
    best_path_length = get_path_length(best_path)

    for _ in range(num_restarts + 1):
        q_shortened = copy.deepcopy(q_path)
        for _ in range(max_iters):
            # Sample two points along the path length
            path_scalings = get_normalized_path_scaling(q_shortened)
            while True:
                low_point = np.random.random()
                high_point = np.random.random()
                if low_point > high_point:
                    low_point, high_point = high_point, low_point
                q_low, idx_low = get_configuration_from_normalized_path_scaling(
                    q_shortened, path_scalings, low_point
                )
                q_high, idx_high = get_configuration_from_normalized_path_scaling(
                    q_shortened, path_scalings, high_point
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
