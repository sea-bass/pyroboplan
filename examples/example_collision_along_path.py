import hppfcl
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as mg
import numpy as np
from os.path import dirname, join, abspath
import time

from pyroboplan.core.utils import extract_cartesian_poses
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames


def prepare_collision_scene(collision_model):
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
        hppfcl.Box(0.25, 0.25, 0.25),
        pinocchio.SE3(np.eye(3), np.array([-0.5, 0.5, 0.7])),
    )
    obstacle_1.meshColor = np.array([0.0, 1.0, 0.0, 0.2])
    collision_model.addGeometryObject(obstacle_1)


if __name__ == "__main__":
    # Create models and data
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "models")
    package_dir = join(pinocchio_model_dir, "panda_description")
    urdf_filename = join(package_dir, "urdf", "panda.urdf")

    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_filename, package_dirs=pinocchio_model_dir
    )
    data = model.createData()

    prepare_collision_scene(collision_model)

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayCollisions(True)
    viz.displayVisuals(False)

    # Define the active collision pairs
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = ["obstacle_0", "obstacle_1"]
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
    q_path = discretize_joint_space_path(q_start, q_end, max_angle_step)

    # Visualize the path
    target_tforms = extract_cartesian_poses(model, "panda_hand", q_path)
    visualize_frames(viz, "path", target_tforms, line_length=0.05, line_width=1)
    viz.display(q_start)
    time.sleep(1.0)

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
