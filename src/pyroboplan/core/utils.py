""" Core utilities for robot modeling. """

import copy
import numpy as np
import pinocchio


def check_collisions_at_state(model, collision_model, q):
    data = model.createData()
    collision_data = collision_model.createData()

    pinocchio.computeCollisions(model, data, collision_model, collision_data, q, False)
    return np.any([cr.isCollision() for cr in collision_data.collisionResults])


def check_collisions_along_path(model, collision_model, q_path):
    """
    Checks collisions along a path.
    """
    data = model.createData()
    collision_data = collision_model.createData()

    for q in q_path:
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, False
        )
        if np.any([cr.isCollision() for cr in collision_data.collisionResults]):
            return True

    return False


def get_random_state(model, padding=0.0):
    """
    Returns a random state that is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        padding : float or array-like, optional
            The padding to use around the sampled joint limits.

    Returns
    -------
        array-like
            A set of randomly generated joint variables.
    """
    return np.random.uniform(
        model.lowerPositionLimit + padding, model.upperPositionLimit - padding
    )


def get_random_collision_free_state(model, collision_model, padding=0.0, max_tries=100):
    """
    Returns a random state that is within the model's joint limits and is collision-free according to the collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        padding : float or array-like, optional
            The padding to use around the sampled joint limits.
        max_tries : int, optional
            The maximum number of tries for sampling a collision-free state.

    Returns
    -------
        array-like
            A set of randomly generated collision-free joint variables, or None if one cannot be found.
    """
    num_tries = 0
    while num_tries < max_tries:
        state = get_random_state(model, padding=padding)
        if not check_collisions_at_state(model, collision_model, state):
            return state
        num_tries += 1

    print(f"Could not generate collision-free state after {max_tries} tries.")
    return None


def get_random_transform(model, target_frame, joint_padding=0.0):
    """
    Returns a random transform for a target frame that is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random transform.
        target_frame : str
            The name of the frame for which to generate a random transform.
        joint_padding : float or array-like, optional
            The padding to use around the sampled joint limits.

    Returns
    -------
        `pinocchio.SE3`
            A randomly generated transform for the specified target frame.
    """
    q_target = get_random_state(model, joint_padding)
    data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_target)
    return data.oMf[target_frame_id]


def check_within_limits(model, q):
    """
    Checks whether a particular joint configuration is within the model's joint limits.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model.

    Returns
    -------
        bool
            True if the configuration is within joint limits, otherwise False.
    """
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )


def extract_cartesian_poses(model, target_frame, q_vec):
    """
    Extracts the Cartesian poses of a specified model frame given a list of joint configurations.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to perform forward kinematics.
        target_frame : str
            The name of the target frame.
        q_vec : array-like
            A list of joint configuration values describing the path.

    Returns
    -------
        list[`pinocchio.SE3`]
            A list of transforms describing the Cartesian poses of the specified frame at the provided joint configurations.
    """
    data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    tforms = []
    for q in q_vec:
        pinocchio.framesForwardKinematics(model, data, q)
        tforms.append(copy.deepcopy(data.oMf[target_frame_id]))
    return tforms
