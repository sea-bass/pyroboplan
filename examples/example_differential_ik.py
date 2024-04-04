import hppfcl
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from os.path import dirname, join, abspath

from pyroboplan.core.utils import get_random_state, get_random_transform
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import joint_limit_nullspace_component


def prepare_scene(visual_model, collision_model):
    """Helper function to create a collision scene for this example."""

    # Add collision objects
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
    np.set_printoptions(precision=3)

    # Define the active collision pairs
    collision_names = [
        cobj.name for cobj in collision_model.geometryObjects if "panda" in cobj.name
    ]
    obstacle_names = [
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

    # Set up the IK solver
    ik = DifferentialIk(
        model, data=data, collision_model=collision_model, visualizer=viz, verbose=True
    )
    target_frame = "panda_hand"
    options = DifferentialIkOptions()
    nullspace_components = [
        lambda model, q: joint_limit_nullspace_component(model, q, gain=0.5)
    ]

    # Solve IK several times and print the results
    for _ in range(10):
        init_state = get_random_state(model)
        target_tform = get_random_transform(model, target_frame)
        q_sol = ik.solve(
            target_frame,
            target_tform,
            init_state=init_state,
            options=options,
            nullspace_components=[],
        )
        print(f"Solution configuration:\n{q_sol}\n")
