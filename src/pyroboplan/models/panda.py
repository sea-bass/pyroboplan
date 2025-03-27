"""Utilities to load example Franka Emika Panda model."""

import coal
import numpy as np
import os
import pinocchio
import hppfcl as fcl
from plyfile import PlyData


from ..core.utils import set_collisions
from .utils import get_example_models_folder


def load_models(use_sphere_collisions=False):
    """
    Gets the example Panda models.

    Returns
    -------
        tuple[`pinocchio.Model`]
            A 3-tuple containing the model, collision geometry model, and visual geometry model.
    """
    models_folder = get_example_models_folder()
    package_dir = os.path.join(models_folder, "panda_description")
    urdf_filename = "panda_spheres.urdf" if use_sphere_collisions else "panda.urdf"
    urdf_filepath = os.path.join(package_dir, "urdf", urdf_filename)

    return pinocchio.buildModelsFromUrdf(urdf_filepath, package_dirs=models_folder)


def add_self_collisions(model, collision_model, srdf_filename=None):
    """
    Adds link self-collisions to the Panda collision model.

    This uses an SRDF file to remove any excluded collision pairs.

    Parameters
    ----------
        model : `pinocchio.Model`
            The Panda model.
        collision_model : `pinocchio.Model`
            The Panda collision geometry model.
        srdf_filename : str, optional
            Path to the SRDF file describing the excluded collision pairs.
            If not specified, uses a default file included with the Panda model.
    """
    if srdf_filename is None:
        models_folder = get_example_models_folder()
        package_dir = os.path.join(models_folder, "panda_description")
        srdf_filename = os.path.join(package_dir, "srdf", "panda.srdf")

    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)


def add_object_collisions(model, collision_model, visual_model, inflation_radius=0.0):
    """
    Adds obstacles and collisions to the Panda collision model.

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
        "ground_plane",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.151])),
        coal.Box(2.0, 2.0, 0.3),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
        coal.Sphere(0.2 + inflation_radius),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
        coal.Sphere(0.25 + inflation_radius),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
        coal.Box(
            0.25 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
            0.55 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
        coal.Box(
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
            0.33 + 2.0 * inflation_radius,
        ),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_2)
    collision_model.addGeometryObject(obstacle_box_2)

    # Define the active collision pairs between the robot and obstacle links.
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = [
        "ground_plane",
        "obstacle_box_1",
        "obstacle_box_2",
        "obstacle_sphere_1",
        "obstacle_sphere_2",
    ]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)

    # Exclude the collision between the ground and the base link
    set_collisions(model, collision_model, "panda_link0", "ground_plane", False)


def add_octree_collisions(model, collision_model, visual_model):

    # Read the PLY file
    ply_data = PlyData.read('./living_room_furniture.ply')

    # Access vertex data (assuming it's a mesh file)
    vertices = ply_data['vertex']
    vertex_array = np.array([vertices['x'], vertices['y'], vertices['z']]).T
    octree = fcl.makeOctree(vertex_array, 0.01)

    # octree = fcl.makeOctree(np.random.rand(8, 3)+np.array([-0.5, -0.5, 0]), 0.01)
    octree_object = pinocchio.GeometryObject("octree", 0, pinocchio.SE3.Identity(), octree)
    octree_object.meshColor[0] = 1.0
    collision_model.addGeometryObject(octree_object)
    visual_model.addGeometryObject(octree_object)

    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]

    for collision_name in collision_names:
        set_collisions(model, collision_model, "octree", collision_name, True)
