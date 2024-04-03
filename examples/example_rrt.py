import hppfcl
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from os.path import dirname, join, abspath
import time

from pyroboplan.planning.rrt import RRTPlanner
from pyroboplan.planning.utils import discretize_joint_space_path


def prepare_scene(visual_model, collision_model):
    """Helper function to create a collision scene for this example."""

    # Add collision objects
    obstacle_0 = pinocchio.GeometryObject(
        "obstacle_sphere",
        0,
        hppfcl.Sphere(0.2),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
    )
    obstacle_0.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_0)
    collision_model.addGeometryObject(obstacle_0)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        hppfcl.Sphere(0.25),
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_1 = pinocchio.GeometryObject(
        "obstacle_box",
        0,
        hppfcl.Box(0.25, 0.25, 0.25),
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    )
    obstacle_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_1)
    collision_model.addGeometryObject(obstacle_1)

    obstacle_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        hppfcl.Box(0.33, 0.33, 0.33),
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
    )
    obstacle_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_2)
    collision_model.addGeometryObject(obstacle_2)


if __name__ == "__main__":
    # Create models and data
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "models")
    package_dir = join(pinocchio_model_dir, "panda_description")
    urdf_filename = join(package_dir, "urdf", "panda.urdf")

    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_filename, package_dirs=pinocchio_model_dir
    )
    data = model.createData()

    prepare_scene(visual_model, collision_model)

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the active collision pairs
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = [
        "obstacle_box",
        "obstacle_box_2",
        "obstacle_sphere",
        "obstacle_sphere_2",
    ]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            collision_model.addCollisionPair(
                pinocchio.CollisionPair(
                    collision_model.getGeometryId(collision_name),
                    collision_model.getGeometryId(obstacle_name),
                )
            )
    collision_data = pinocchio.GeometryData(collision_model)

    # Define a joint space path
    q_start = pinocchio.randomConfiguration(model)
    q_end = pinocchio.randomConfiguration(model)
    max_angle_step = 0.05

    viz.display(q_start)
    time.sleep(1.0)

    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_end)
    planner.visualize(viz, show_tree=True)

    # Animate the path
    for idx in range(1, len(path)):
        segment_start = path[idx - 1]
        segment_end = path[idx]
        q_path = discretize_joint_space_path(segment_start, segment_end, 0.05)
        for q in q_path:
            viz.display(q)
            time.sleep(0.05)
