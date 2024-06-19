"""
This example shows PyRoboPlan capabilities around Pinocchio to import a model
and perform collision checking along a predefined path.
These capabilities form the basis of collision checking for validating other
motion planning components such as inverse kinematics and path planning.
"""

import hppfcl
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import numpy as np
import time

from pyroboplan.core.utils import extract_cartesian_poses, set_collisions
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames


def prepare_collision_scene(model, collision_model):
    """Helper function to create a collision scene for this example."""

    # Modify the collision model so all the Panda links are translucent
    for cobj in collision_model.geometryObjects:
        cobj.meshColor = np.array([0.7, 0.7, 0.7, 0.25])

    # Add collision objects
    obstacle_0 = pinocchio.GeometryObject(
        "obstacle_0",
        0,
        hppfcl.Sphere(0.2),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.1, 1.1])),
    )
    obstacle_0.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_0)

    obstacle_1 = pinocchio.GeometryObject(
        "obstacle_1",
        0,
        hppfcl.Box(0.3, 0.3, 0.3),
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    )
    obstacle_1.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_1)

    # Define the active collision pairs
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = ["obstacle_0", "obstacle_1"]
    for obstacle_name in obstacle_names:
        for collision_name in collision_names:
            set_collisions(model, collision_model, obstacle_name, collision_name, True)


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    prepare_collision_scene(model, collision_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayCollisions(True)
    viz.displayVisuals(False)

    # Define a joint space path
    q_start = pinocchio.randomConfiguration(model)
    q_end = pinocchio.randomConfiguration(model)
    max_step_size = 0.05
    q_path = discretize_joint_space_path([q_start, q_end], max_step_size)

    # Visualize the path
    target_tforms = extract_cartesian_poses(model, "panda_hand", q_path)
    visualize_frames(viz, "path", target_tforms, line_length=0.05, line_width=1)
    viz.display(q_start)
    time.sleep(0.5)

    # Collision check along the path
    for q in q_path:
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, False
        )

        contacts = []
        for k in range(len(collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = collision_model.collisionPairs[k]
            if cr.isCollision():
                print(
                    "collision between:",
                    collision_model.geometryObjects[cp.first].name,
                    " and ",
                    collision_model.geometryObjects[cp.second].name,
                )
                for contact in cr.getContacts():
                    contacts.extend(
                        [
                            contact.pos,
                            contact.pos - contact.normal * contact.penetration_depth,
                        ]
                    )
        if len(contacts) == 0:
            print("Found no collisions!")

        viz.viewer["collision_display"].set_object(
            mg.LineSegments(
                mg.PointsGeometry(
                    position=np.array(contacts).T,
                    color=np.array([[1.0, 0.0, 0.0] for _ in contacts]).T,
                ),
                mg.LineBasicMaterial(
                    linewidth=3,
                    vertexColors=True,
                ),
            )
        )

        viz.display(q)
        time.sleep(0.1)
