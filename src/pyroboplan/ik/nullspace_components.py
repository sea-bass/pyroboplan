""" Library of common nullspace components for inverse kinematics. """

import numpy as np
import pinocchio


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
    max_vel=1.0,
    damping=0.0001,
    gain=1.0,
):
    """
    Returns a collision avoidance nullspace component.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        data : `pinocchio.Data`
            The model data to use for collision distance checks.
        collision_model : `pinocchio.GeometryModel`
            The model with which to check collision distances.
        collision_data : `pinocchio.Data`
            The collision model data to use for collision distance checks.
        q : array-like
            The joint configuration for the model.
        dist_padding : float
            The distance padding, in meters, on the collision distances.
            For example, a distance padding of 0.1 means collisions have an influence 10 cm away from actual collision.
        max_vel : float
            The maximum velocity norm that can be returned by this component.
        damping : float
            Damping value, between 0 and 1, for the collision Jacobian pseudoinverse.
        gain : float, optional
            A gain to modify the relative weight of this term.

    Returns
    -------
        array-like
            An array containing the collision avoidance nullspace terms.
    """
    coll_component = np.zeros_like(model.lowerPositionLimit)

    # Find all the collision distances.
    pinocchio.computeDistances(model, data, collision_model, collision_data, q)

    # For each collision pair within a distance threshold, calculate its collision Jacobian
    # and use it to push the corresponding joint values away from collision.
    for k in range(len(collision_model.collisionPairs)):
        cp = collision_model.collisionPairs[k]
        dr = collision_data.distanceResults[k]

        if dr.min_distance > dist_padding:
            continue

        frame1 = collision_model.geometryObjects[cp.first].parentFrame
        # if frame1 < model.nframes:
        #     frame1_name = model.frames[frame1].name

        frame2 = collision_model.geometryObjects[cp.second].parentFrame
        # if frame2 < model.nframes:
        #     frame2_name = model.frames[frame2].name

        flip_order = (frame1 >= model.nframes) or (frame2 > frame1)
        if not flip_order:
            J = pinocchio.computeFrameJacobian(
                model, data, q, frame1, pinocchio.ReferenceFrame.LOCAL
            )
            distance_vec = dr.getNearestPoint2() - dr.getNearestPoint1()
            T = (
                (collision_model.geometryObjects[cp.first].placement * data.oMf[frame1])
                .inverse()
                .toActionMatrix()
            )
        else:
            J = pinocchio.computeFrameJacobian(
                model, data, q, frame2, pinocchio.ReferenceFrame.LOCAL
            )
            distance_vec = dr.getNearestPoint1() - dr.getNearestPoint2()
            T = (
                (
                    collision_model.geometryObjects[cp.second].placement
                    * data.oMf[frame2]
                )
                .inverse()
                .toActionMatrix()
            )

        # Now that we have the collision Jacobian, figure out the effective joint velocity to move away from collision.
        Jcoll = T[:3, :] @ J
        if dr.min_distance > 1e-6:
            distance_vec *= 1.0 - dist_padding / dr.min_distance
        delta_q = Jcoll.T @ np.linalg.solve(
            Jcoll.dot(Jcoll.T) + damping**2 * np.eye(3), distance_vec
        )
        coll_component -= delta_q

    coll_component = gain * coll_component

    # Limit the maximum velocity returned by this component.
    final_norm = np.linalg.norm(coll_component)
    if final_norm > max_vel:
        coll_component *= max_vel / final_norm

    return coll_component
