""" Core utilities for robot modeling. """

import copy
import numpy as np
import pinocchio


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
