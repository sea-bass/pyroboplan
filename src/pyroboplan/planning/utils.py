""" Utilities for motion planning. """

import numpy as np


def discretize_joint_space_path(q_start, q_end, max_angle_distance):
    """
    Discretizes a joint space path from `q_start` to `q_end` given a maximum angle distance between samples.

    This is used primarily for producing paths for collision checking.

    Parameters
    ----------
        q_start : array-like
            The starting joint configuration.
        q_end : array-like
            The final joint configuration.
        max_angle_distance : float
            The maximum angular displacement, in radians, between samples.

    Returns
    -------
        list[array-like]
            A list of joint configuration arrays between the start and end points, inclusive.
    """
    q_diff = q_end - q_start
    num_steps = int(np.ceil(np.linalg.norm(q_diff) / max_angle_distance)) + 1
    step_vec = np.linspace(0.0, 1.0, num_steps)
    return [q_start + step * q_diff for step in step_vec]
