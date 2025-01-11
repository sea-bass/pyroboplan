""" Utilities to load example 2-DOF manipulator. """

import coal
import numpy as np
import os
import pinocchio

from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models():
    """
    Gets the example 2-DOF models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "two_dof_description")
    urdf_filename = os.path.join(package_dir, "two_dof.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)


def add_object_collisions(model, collision_model, visual_model):
    """
    Adds obstacles and collisions to the 2-DOF manipulator collision model.

    Parameters
    ----------
        model : `pinocchio.Model`
            The robot model.
        collision_model : `pinocchio.Model`
            The collision geometry model.
        visual_model : `pinocchio.Model`
            The visual geometry model.
    """
    obstacle_1 = pinocchio.GeometryObject(
        "obstacle_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([1.0, 1.0, 0.0])),
        coal.Cylinder(0.3, 0.1),
    )
    obstacle_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_1)
    collision_model.addGeometryObject(obstacle_1)

    obstacle_2 = pinocchio.GeometryObject(
        "obstacle_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([-1.0, -0.75, 0.0])),
        coal.Box(0.5, 0.5, 0.1),
    )
    obstacle_2.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_2)
    collision_model.addGeometryObject(obstacle_2)

    # Define the active collision pairs between the robot and obstacle links.
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "arm" in cobj.name
    ]
    obstacle_names = ["obstacle_1", "obstacle_2"]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)
