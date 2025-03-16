"""Library of common nullspace components for inverse kinematics."""

import numpy as np
import pinocchio

from pyroboplan.core.utils import calculate_collision_vector_and_jacobians


def zero_nullspace_component(model, q):
    """
    Returns a zero nullspace component, which is effectively a no-op.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model. Not used, but required to match the function interface.

    Returns
    -------
        array-like
            An array of zeros whose length is the number of joint variables in the model.
    """
    return np.zeros_like(model.lowerPositionLimit)


def joint_limit_nullspace_component(model, q, gain=1.0, padding=0.0):
    """
    Returns a joint limits avoidance nullspace component.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model.
        gain : float, optional
            A gain to modify the relative weight of this term.
        padding : float, optional
            Optional padding around the joint limits.

    Returns
    -------
        array-like
            An array containing the joint space avoidance nullspace terms.
    """
    upper_limits = model.upperPositionLimit - padding
    lower_limits = model.lowerPositionLimit + padding

    grad = zero_nullspace_component(model, q)
    for idx in range(len(grad)):
        if q[idx] > upper_limits[idx]:
            grad[idx] = -gain * (q[idx] - upper_limits[idx])
        elif q[idx] < lower_limits[idx]:
            grad[idx] = -gain * (q[idx] - lower_limits[idx])
    return grad


def joint_center_nullspace_component(model, q, gain=1.0):
    """
    Returns a joint centering nullspace component.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        q : array-like
            The joint configuration for the model.
        gain : float, optional
            A gain to modify the relative weight of this term.

    Returns
    -------
        array-like
            An array containing the joint centering nullspace terms.
    """
    joint_center_positions = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
    return gain * (joint_center_positions - q)


def collision_avoidance_nullspace_component(
    model,
    data,
    collision_model,
    collision_data,
    q,
    dist_padding=0.05,
    gain=1.0,
):
    """
    Returns a collision avoidance nullspace component.

    These calculations are based off the following resources:
       * https://typeset.io/pdf/a-collision-free-mpc-for-whole-body-dynamic-locomotion-and-1l6itpfk.pdf
       * https://laas.hal.science/hal-04425002

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        data : `pinocchio.Data`
            The model data to use for collision distance checks.
        collision_model : `pinocchio.GeometryModel`
            The model with which to check collision distances.
        collision_data : `pinocchio.GeometryData`
            The collision model data to use for collision distance checks.
        q : array-like
            The joint configuration for the model.
        dist_padding : float
            The distance padding, in meters, on the collision distances.
            For example, a distance padding of 0.1 means collisions have an influence 10 cm away from actual collision..
        gain : float, optional
            A gain to modify the relative weight of this term.

    Returns
    -------
        array-like
            An array containing the collision avoidance nullspace terms.
    """
    coll_component = np.zeros_like(model.lowerPositionLimit)

    # Find all the collision distances at the current state.
    pinocchio.framesForwardKinematics(model, data, q)
    pinocchio.computeCollisions(model, data, collision_model, collision_data, q, False)
    pinocchio.computeDistances(model, data, collision_model, collision_data, q)

    # For each collision pair within a distance threshold, calculate its collision Jacobians
    # and use them to push the corresponding joint values away from collision.
    for idx in range(len(collision_model.collisionPairs)):
        dist = collision_data.distanceResults[idx].min_distance
        if dist > dist_padding:
            continue

        distance_vec, Jcoll1, Jcoll2 = calculate_collision_vector_and_jacobians(
            model, collision_model, data, collision_data, idx, q
        )
        coll_component -= (distance_vec - dist_padding) @ (Jcoll2 - Jcoll1)

    coll_component = gain * coll_component

    return coll_component
