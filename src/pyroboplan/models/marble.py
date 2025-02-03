"""Utilities to load a 2d spherical body resembling a marble labyrinth."""

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
    package_dir = os.path.join(models_folder, "marble_description")
    urdf_filename = os.path.join(package_dir, "marble.urdf")

    return pinocchio.buildModelsFromUrdf(urdf_filename, package_dirs=models_folder)


def add_object_collisions(model, collision_model, visual_model):
    ground_plane = pinocchio.GeometryObject(
        "ground",
        0,
        pinocchio.SE3(np.eye(3), np.array([1.0, 0.5, -0.05 - 0.02])),
        coal.Box(2.1, 1.1, 0.1),
    )
    ground_plane.meshColor = np.array([0.8, 0.8, 0.8, 1.0])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    box_idx = 0

    def box_on_plane(x, y, w, h):
        nonlocal box_idx
        box_idx += 1
        box = pinocchio.GeometryObject(
            f"box{box_idx}",
            0,
            pinocchio.SE3(np.eye(3), np.array([x + w / 2, y + h / 2, 0.0])),
            coal.Box(w, h, 0.04),
        )
        box.meshColor = np.array([0.3, 0.3, 0.3, 1.0])
        visual_model.addGeometryObject(box)
        collision_model.addGeometryObject(box)

    # outer walls
    box_on_plane(-0.05, -0.05, 0.05, 1.1)
    box_on_plane(2.0, -0.05, 0.05, 1.1)
    box_on_plane(-0.05, -0.05, 2.1, 0.05)
    box_on_plane(-0.05, 1.0, 2.1, 0.05)

    # obstacles
    box_on_plane(0.3, 0.0, 0.05, 0.7)
    box_on_plane(0.75, 0.2, 0.05, 0.8)
    box_on_plane(1.3, 0.0, 0.05, 0.3)
    box_on_plane(0.8, 0.5, 0.5, 0.05)
    box_on_plane(1.6, 0.45, 0.05, 0.4)
    box_on_plane(1.65, 0.45, 0.35, 0.05)

    # split workspace so that many plans will not be feasible and the search trees fill the plane
    # box_on_plane(1.0, 0.0, 0.05, 0.5)

    # Define the active collision pairs between the robot and obstacle links.
    obstacle_names = [
        cobj.name
        for cobj in collision_model.geometryObjects
        if cobj.name.startswith("box")
    ]
    for obstacle_name in obstacle_names:
        set_collisions(model, collision_model, obstacle_name, "body", True)
