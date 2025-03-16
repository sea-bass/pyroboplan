"""Core utilities for robot modeling."""

import copy
import numpy as np
import pinocchio


def check_collisions_at_state(
    model,
    collision_model,
    q,
    data=None,
    collision_data=None,
    distance_padding=0.0,
):
    """
    Checks whether a specified joint configuration is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for collision checking.
        collision_model : `pinocchio.Model`
            The collision model to use for collision checking.
        q : array-like
            The joint configuration of the model.
        data : `pinocchio.Data`, optional
            The model data to use for collision checking. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for collision checking. If None, data is created automatically.
        distance_padding : float, optional
            The padding, in meters, to use for distance to nearest collision.

    Returns
    -------
        bool
            True if there are any collisions or minimum distance violations, otherwise False.
    """
    if not data:
        data = model.createData()
    if not collision_data:
        collision_data = collision_model.createData()
    stop_at_first_collision = True  # For faster computation

    for elem in collision_data.collisionRequests:
        elem.security_margin = distance_padding

    pinocchio.computeCollisions(
        model, data, collision_model, collision_data, q, stop_at_first_collision
    )
    return np.any([cr.isCollision() for cr in collision_data.collisionResults])


def get_minimum_distance_at_state(
    model, collision_model, q, data=None, collision_data=None
):
    """
    Gets the minimum distance to collision at a specified state.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for collision checking.
        collision_model : `pinocchio.Model`
            The collision model to use for collision checking.
        q : array-like
            The joint configuration of the model.
        data : `pinocchio.Data`, optional
            The model data to use for collision checking. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for collision checking. If None, data is created automatically.

    Returns
    -------
        float
            The minimum distance to collision, in meters.
    """
    if len(collision_model.collisionPairs) == 0:
        return np.inf

    if not data:
        data = model.createData()
    if not collision_data:
        collision_data = collision_model.createData()

    pinocchio.computeDistances(model, data, collision_model, collision_data, q)
    return np.min([dr.min_distance for dr in collision_data.distanceResults])


def check_collisions_along_path(
    model, collision_model, q_path, data=None, collision_data=None, distance_padding=0.0
):
    """
    Checks whether a path consisting of multiple joint configurations is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        q_path : list[array-like]
            A list of joint configurations describing the path.
        data : `pinocchio.Data`, optional
            The model data to use for collision checking. If None, data is created automatically.
        collision_data : `pinocchio.GeometryData`, optional
            The collision_model data to use for collision checking. If None, data is created automatically.
        distance_padding : float, optional
            The padding, in meters, to use for distance to nearest collision.

    Returns
    -------
        bool
            True if there are any collisions or minimum distance violations, otherwise False.
    """
    if not data:
        data = model.createData()
    if not collision_data:
        collision_data = collision_model.createData()
    stop_at_first_collision = True  # For faster computation

    for elem in collision_data.collisionRequests:
        elem.security_margin = distance_padding

    for q in q_path:
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, stop_at_first_collision
        )
        if np.any([cr.isCollision() for cr in collision_data.collisionResults]):
            return True

    return False


def configuration_distance(q_start, q_end):
    """
    Returns the distance between two joint configurations.

    Parameters
    ----------
        q_start : array-like
            The start joint configuration.
        q_end : array-like
            The end joint configuration.

    Returns
    -------
        float
            The distance between the two joint configurations.
    """
    return np.linalg.norm(q_end - q_start)


def get_path_length(q_path):
    """
    Returns the configuration distance of a path.

    Parameters
    ----------
        q_path : list[array-like]
            A list of joint configurations describing a path.

    Returns
    -------
        float
            The total configuration distance of the entire path.
    """
    total_distance = 0.0
    for idx in range(1, len(q_path)):
        total_distance += configuration_distance(q_path[idx - 1], q_path[idx])
    return total_distance


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


def get_random_collision_free_state(
    model, collision_model, joint_padding=0.0, distance_padding=0.0, max_tries=100
):
    """
    Returns a random state that is within the model's joint limits and is collision-free according to the collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random state.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        joint_padding : float or array-like, optional
            The padding to use around the sampled joint limits.
        distance_padding : float, optional
            The padding, in meters, to use for distance to nearest collision.
        max_tries : int, optional
            The maximum number of tries for sampling a collision-free state.

    Returns
    -------
        array-like
            A set of randomly generated collision-free joint variables, or None if one cannot be found.
    """
    num_tries = 0
    while num_tries < max_tries:
        state = get_random_state(model, padding=joint_padding)
        if not check_collisions_at_state(
            model, collision_model, state, distance_padding=distance_padding
        ):
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


def get_random_collision_free_transform(
    model,
    collision_model,
    target_frame,
    joint_padding=0.0,
    distance_padding=0.0,
    max_tries=100,
):
    """
    Returns a random transform for a target frame that is within the model's joint limits and is collision-free.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to generate a random transform.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        target_frame : str
            The name of the frame for which to generate a random transform.
        joint_padding : float or array-like, optional
            The padding to use around the sampled joint limits.
        distance_padding : float, optional
            The padding, in meters, to use for distance to nearest collision.
        max_tries : int, optional
            The maximum number of tries for sampling a collision-free state.

    Returns
    -------
        `pinocchio.SE3`
            A randomly generated transform for the specified target frame.
    """
    q_target = get_random_collision_free_state(
        model,
        collision_model,
        joint_padding=joint_padding,
        distance_padding=distance_padding,
        max_tries=max_tries,
    )
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


def extract_cartesian_pose(model, target_frame, q, data=None):
    """
    Extracts the Cartesian pose of a specified model frame given a joint configuration.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model from which to perform forward kinematics.
        target_frame : str
            The name of the target frame.
        q : array-like
            The joint configuration values describing the robot state.
        data : `pinocchio.Data`, optional
            The model data to use. If not set, one will be created.

    Returns
    -------
        `pinocchio.SE3`
            The transform describing the Cartesian pose of the specified frame at the provided joint configuration.
    """
    if data is None:
        data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q)
    return copy.deepcopy(data.oMf[target_frame_id])


def extract_cartesian_poses(model, target_frame, q_vec, data=None):
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
        data : `pinocchio.Data`, optional
            The model data to use. If not set, one will be created.

    Returns
    -------
        list[`pinocchio.SE3`]
            A list of transforms describing the Cartesian poses of the specified frame at the provided joint configurations.
    """
    if data is None:
        data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    tforms = []
    for q in q_vec:
        pinocchio.framesForwardKinematics(model, data, q)
        tforms.append(copy.deepcopy(data.oMf[target_frame_id]))
    return tforms


def get_collision_geometry_ids(model, collision_model, body):
    """
    Gets a list of collision geometry model IDs for a specified body name.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for getting frame IDs.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        body : str
            The name of the body.
            This can be directly the name of a geometry in the collision model,
            or it can be the name of the frame in the main model.

    Return
    ------
        list[int]
            A list of collision geometry IDs corresponding to the body name.
    """
    body_collision_geom_ids = []

    # First, check if this is directly a collision geometry.
    body_collision_geom_id = collision_model.getGeometryId(body)
    if body_collision_geom_id < collision_model.ngeoms:
        body_collision_geom_ids.append(body_collision_geom_id)
    else:
        # Otherwise, look for the frame name in the model and return its associated collision objects.
        body_frame_id = model.getFrameId(body)
        if body_frame_id < model.nframes:
            for id, obj in enumerate(collision_model.geometryObjects):
                if obj.parentFrame == body_frame_id:
                    body_collision_geom_ids.append(id)

    return body_collision_geom_ids


def get_collision_pair_indices_from_bodies(model, collision_model, body_list):
    """
    Returns a list of all the collision pair indices involving a list of objects.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for getting frame IDs.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        body_list : list[str]
            A list containing the names of bodies.
            These can be directly the name of a geometry in the collision model,
            or they can be the name of the frame in the main model.

    Return
    ------
        list[int]
            The indices of the collision pairs list involving the bodies in the specified list.
    """
    collision_ids = set()
    for obj in body_list:
        ids = get_collision_geometry_ids(model, collision_model, obj)
        collision_ids.update(ids)

    pairs = []
    for idx, p in enumerate(collision_model.collisionPairs):
        if p.first in collision_ids or p.second in collision_ids:
            pairs.append(idx)

    return pairs


def set_collisions(model, collision_model, body1, body2, enable):
    """
    Sets collision checking between two bodies by searching for their corresponding geometry objects in the collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for getting frame IDs.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        body1 : str
            The name of the first body.
        body2 : str
            The name of the second body.
        enable : bool
            If True, enables collisions. If False, disables collisions.
    """
    body1_collision_ids = get_collision_geometry_ids(model, collision_model, body1)
    body2_collision_ids = get_collision_geometry_ids(model, collision_model, body2)

    for id1 in body1_collision_ids:
        for id2 in body2_collision_ids:
            pair = pinocchio.CollisionPair(id1, id2)
            if enable:
                collision_model.addCollisionPair(pair)
            else:
                collision_model.removeCollisionPair(pair)


def calculate_collision_vector_and_jacobians(
    model, collision_model, data, collision_data, pair_idx, q
):
    """
    Given collision and distance results from collision model data, computes the collision vector
    and collision Jacobians at both collision points.

    This is useful for algorithms that perform collision avoidance, such as IK and trajectory optimization.

    Note that forward kinematics, collision, and distance checks must be evaluated first to populate the
    `data` and `collision_data` variables.

    Parameters
    ----------
        model : `pinocchio.Model`
            The model to use for Jacobian computation.
        collision_model : `pinocchio.Model`
            The model to use for collision checking.
        data : `pinocchio.Data`
            The model data to use for Jacobian computation.
        collision_data : `pinocchio.GeometryData`
            The collision_model data to use for collision checking.
        pair_idx : int
            The index of the collision pair from which to extract information.
        q : array-like
            The joint configuration of the model.

    Return
    ------
        tuple(array-like, array-like, array-like)
            A tuple containing collision distance vector from frame1 to frame2,
            and the collision Jacobians at frame1 and frame2.
    """
    cp = collision_model.collisionPairs[pair_idx]
    dr = collision_data.distanceResults[pair_idx]

    # According to the coal documentation, the normal always points from object1 to object2.
    coll_points = [dr.getNearestPoint1(), dr.getNearestPoint2()]
    distance_vec = dr.normal
    distance = dr.min_distance

    # Calculate the Jacobians at the parent frames of both collision points.
    parent_frame1 = collision_model.geometryObjects[cp.first].parentFrame
    if parent_frame1 >= model.nframes:
        parent_frame1 = 0
    Jframe1 = pinocchio.computeFrameJacobian(
        model, data, q, parent_frame1, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    r1 = coll_points[0] - data.oMf[parent_frame1].translation
    Jcoll1 = pinocchio.SE3(np.eye(3), r1).toActionMatrix()[:3, :] @ Jframe1

    parent_frame2 = collision_model.geometryObjects[cp.second].parentFrame
    if parent_frame2 >= model.nframes:
        parent_frame2 = 0
    Jframe2 = pinocchio.computeFrameJacobian(
        model, data, q, parent_frame2, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    r2 = coll_points[1] - data.oMf[parent_frame2].translation
    Jcoll2 = pinocchio.SE3(np.eye(3), r2).toActionMatrix()[:3, :] @ Jframe2

    return distance_vec * distance, Jcoll1, Jcoll2
