""" Utilities for motion planning. """

import numpy as np

from ..core.utils import (
    check_collisions_at_state,
    check_collisions_along_path,
    configuration_distance,
    get_random_state,
)

from itertools import product


def extend_robot_state(q_parent, q_sample, max_connection_distance):
    """
    Determines an incremental robot configuration between the parent and sample states, if one exists.

    Parameters
    ----------
        q_parent : array-like
            The starting robot configuration.
        q_sample : array-like
            The candidate sample configuration to extend towards.
        max_connection_distance : float
            Maximum angular distance, in radians, for connecting nodes.

    Returns
    -------
        array-like
            The resulting robot configuration, or None if it is not feasible.
    """
    q_diff = q_sample - q_parent
    q_increment = max_connection_distance * q_diff / np.linalg.norm(q_diff)

    q_cur = q_parent
    # Clip the distance between nearest and sampled nodes to max connection distance.
    # If we have reached the sampled node, then we just check that.
    if configuration_distance(q_cur, q_sample) > max_connection_distance:
        q_extend = q_cur + q_increment
    else:
        q_extend = q_sample

    # Then there are no collisions so the extension is valid
    return q_extend


def has_collision_free_path(q1, q2, max_angle_step, model, collision_model):
    """
    Determines if there is a collision free path between the provided nodes and models.

    Parameters
    ----------
        q1 : array-like
            The starting robot configuration.
        q2 : array-like
            The destination robot configuration.
        max_angle_step : float
            Maximum angle step, in radians, for collision checking along path segments.
        model : `pinocchio.Model`
            The model for the robot configuration.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.

    Returns
    -------
        bool
            True if the configurations can be connected, False otherwise.
    """
    # Ensure the destination is collision free.
    if check_collisions_at_state(model, collision_model, q2):
        return False

    # Ensure the discretized path is collision free.
    path_to_q_extend = discretize_joint_space_path([q1, q2], max_angle_step)
    if check_collisions_along_path(model, collision_model, path_to_q_extend):
        return False

    return True


def discretize_joint_space_path(q_path, max_angle_distance):
    """
    Discretizes a joint space path given a maximum angle distance between samples.

    This is used primarily for producing paths for collision checking.

    Parameters
    ----------
        q_path : list[array-like]
            A list of the joint configurations describing a path.
        max_angle_distance : float
            The maximum angular displacement, in radians, between samples.

    Returns
    -------
        list[array-like]
            A list of joint configuration arrays between the start and end points, inclusive.
    """
    q_discretized = []
    for idx in range(1, len(q_path)):
        q_start = q_path[idx - 1]
        q_end = q_path[idx]
        q_diff = q_end - q_start
        num_steps = int(np.ceil(np.linalg.norm(q_diff) / max_angle_distance)) + 1
        step_vec = np.linspace(0.0, 1.0, num_steps)
        q_discretized.extend([q_start + step * q_diff for step in step_vec])
    return q_discretized


def retrace_path(goal_node):
    """
    Retraces a path to the specified `goal_node` from a root node (a node with no parent).

    The resulting path will be returned in order form the start at index `0` to the `goal_node`
    at the index `-1`.

    Parameters
    ----------
        goal_node : `pyroboplan.planning.graph.Node`
            The starting joint configuration.

    Returns
    -------
        list[`pyroboplan.planning.graph.Node`]
            A list a nodes from the root to the specified `goal_node`.

    """
    path = []
    current = goal_node
    while current:
        path.append(current)
        current = current.parent
    path.reverse()
    return path


def discretized_joint_space_generator(model, step_size, generate_random=True):
    """
    Discretizes the entire joint space of the model at step_size increments.
    Once the entire space has been returned, the generator can optionally continue
    returning random samples from the configuration space - in which case this
    generator will never terminate.

    This is an extraordinarily expensive operation for high DOF manipulators
    and small step sizes!

    Parameters
    ----------
        model : `pinocchio.Model`
            The robot model containing lower and upper position limits.
        step_size : float
            The step size for sampling.
        generate_random : bool
            If True, continue randomly sampling the configuration space.
            Otherwise this generator will terminate.

    Yields
    ------
        np.ndarray
            The next point in the configuration space.
    """
    lower = model.lowerPositionLimit
    upper = model.upperPositionLimit

    # Ensure the range is inclusive of endpoints
    ranges = [np.arange(l, u + step_size, step_size) for l, u in zip(lower, upper)]
    for point in product(*ranges):
        yield np.array(point)

    # Once we have iterated through all available points we return random samples.
    while generate_random:
        yield get_random_state(model)
