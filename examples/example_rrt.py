import hppfcl
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from os.path import dirname, join, abspath
import time

from pyroboplan.core.utils import get_random_collision_free_state
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path


def prepare_scene(visual_model, collision_model):
    """Helper function to create a collision scene for this example."""

    # Add self collisions
    self_collision_pair_names = [
        ("panda_link0_0", "panda_link2_0"),
        ("panda_link0_0", "panda_link3_0"),
        ("panda_link0_0", "panda_link4_0"),
        ("panda_link0_0", "panda_link5_0"),
        ("panda_link0_0", "panda_link6_0"),
        ("panda_link0_0", "panda_link7_0"),
        ("panda_link1_0", "panda_link3_0"),
        ("panda_link1_0", "panda_link4_0"),
        ("panda_link1_0", "panda_link5_0"),
        ("panda_link1_0", "panda_link6_0"),
        ("panda_link1_0", "panda_link7_0"),
        ("panda_link2_0", "panda_link4_0"),
        ("panda_link2_0", "panda_link5_0"),
        ("panda_link2_0", "panda_link6_0"),
        ("panda_link2_0", "panda_link7_0"),
        ("panda_link3_0", "panda_link5_0"),
        ("panda_link3_0", "panda_link6_0"),
        ("panda_link3_0", "panda_link7_0"),
        ("panda_link4_0", "panda_link6_0"),
        ("panda_link4_0", "panda_link7_0"),
        ("panda_link5_0", "panda_link7_0"),
    ]
    for pair in self_collision_pair_names:
        collision_model.addCollisionPair(
            pinocchio.CollisionPair(
                collision_model.getGeometryId(pair[0]),
                collision_model.getGeometryId(pair[1]),
            )
        )

    # Add collision objects
    ground_plane = pinocchio.GeometryObject(
        "ground_plane",
        0,
        hppfcl.Box(2.0, 2.0, 0.01),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, -0.006])),
    )
    ground_plane.meshColor = np.array([0.5, 0.5, 0.5, 0.5])
    visual_model.addGeometryObject(ground_plane)
    collision_model.addGeometryObject(ground_plane)

    obstacle_sphere_1 = pinocchio.GeometryObject(
        "obstacle_sphere_1",
        0,
        hppfcl.Sphere(0.2),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
    )
    obstacle_sphere_1.meshColor = np.array([0.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_1)
    collision_model.addGeometryObject(obstacle_sphere_1)

    obstacle_sphere_2 = pinocchio.GeometryObject(
        "obstacle_sphere_2",
        0,
        hppfcl.Sphere(0.25),
        pinocchio.SE3(np.eye(3), np.array([0.5, 0.5, 0.5])),
    )
    obstacle_sphere_2.meshColor = np.array([1.0, 1.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_sphere_2)
    collision_model.addGeometryObject(obstacle_sphere_2)

    obstacle_box_1 = pinocchio.GeometryObject(
        "obstacle_box_1",
        0,
        hppfcl.Box(0.25, 0.25, 0.25),
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    )
    obstacle_box_1.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_1)
    collision_model.addGeometryObject(obstacle_box_1)

    obstacle_box_2 = pinocchio.GeometryObject(
        "obstacle_box_2",
        0,
        hppfcl.Box(0.33, 0.33, 0.33),
        pinocchio.SE3(np.eye(3), np.array([-0.5, -0.5, 0.75])),
    )
    obstacle_box_2.meshColor = np.array([0.0, 0.0, 1.0, 0.5])
    visual_model.addGeometryObject(obstacle_box_2)
    collision_model.addGeometryObject(obstacle_box_2)


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
        "ground_plane",
        "obstacle_box_1",
        "obstacle_box_2",
        "obstacle_sphere_1",
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

    # Define the start and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)
    viz.display(q_start)
    time.sleep(1.0)

    # Search for a path
    options = RRTPlannerOptions()
    options.max_angle_step = 0.05
    options.max_connection_dist = 0.1
    options.goal_biasing_probability = 0.15
    options.max_planning_time = 10.0
    options.rrt_connect = True
    options.bidirectional_rrt = True

    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_end, options=options)
    planner.visualize(viz, "panda_hand", show_tree=True)

    # Animate the path
    if path:
        input("Press 'Enter' to animate the path.")
        for idx in range(1, len(path)):
            segment_start = path[idx - 1]
            segment_end = path[idx]
            q_path = discretize_joint_space_path(
                segment_start, segment_end, options.max_angle_step
            )
            for q in q_path:
                viz.display(q)
                time.sleep(0.05)
