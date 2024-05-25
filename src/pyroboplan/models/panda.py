""" Utilities to load example Franka Emika Panda model. """

import hppfcl
import numpy as np
import os
import pinocchio

from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models():
    """
    Gets the example Panda models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "panda_description")
    urdf_filename = os.path.join(package_dir, "urdf", "panda.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)


def add_self_collisions(model, collision_model):
    """
    Adds link self-collisions to the Panda collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
    """
    self_collision_pair_names = [
        ("panda_link0", "panda_link2"),
        ("panda_link0", "panda_link3"),
        ("panda_link0", "panda_link4"),
        ("panda_link0", "panda_link5"),
        ("panda_link0", "panda_link6"),
        ("panda_link0", "panda_link7"),
        ("panda_link1", "panda_link3"),
        ("panda_link1", "panda_link4"),
        ("panda_link1", "panda_link5"),
        ("panda_link1", "panda_link6"),
        ("panda_link1", "panda_link7"),
        ("panda_link2", "panda_link4"),
        ("panda_link2", "panda_link5"),
        ("panda_link2", "panda_link6"),
        ("panda_link2", "panda_link7"),
        ("panda_link3", "panda_link5"),
        ("panda_link3", "panda_link6"),
        ("panda_link3", "panda_link7"),
        ("panda_link4", "panda_link6"),
        ("panda_link4", "panda_link7"),
        ("panda_link5", "panda_link7"),
    ]
    for pair in self_collision_pair_names:
        set_collisions(model, collision_model, pair[0], pair[1], True)


def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    """
    Adds link self-collisions to the Panda collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        visual_model : `pinocchio.Model`
            The Panda visual geometry model.
        inflation_radius : float, optional
            An inflation radius, in meters, around the objects.
    """
    # Add the collision objects
    ground_plane = pinocchio.GeometryObject(
        "obstacle_ground",
        0,
        hppfcl.Box(2.0, 2.0, 0.3),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.151])),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        hppfcl.Sphere(0.2 + inflation_radius),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        hppfcl.Sphere(0.25 + inflation_radius),
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        hppfcl.Box(
            0.25 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
        ),
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        hppfcl.Box(
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
        ),
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_2)
    collision_model.addGeometryObject(obstacle_box_2)

    # Define the active collision pairs between the robot and obstacle links.
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = [
        "obstacle_ground",
        "obstacle_box_1",
        "obstacle_box_2",
        "obstacle_sphere_1",
        "obstacle_sphere_2",
    ]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)
