"""
This example shows PyRoboPlan capabilities for inverse kinematics (IK).
IK defines the task of finding a set of joint positions for a robot model to
achieve a desired target pose for a specific coordinate frame.
"""

from pinocchio.visualize import MeshcatVisualizer
import numpy as np

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    get_random_collision_free_transform,
)
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import joint_limit_nullspace_component
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    np.set_printoptions(precision=3)

    # Set up the IK solver
    options = DifferentialIkOptions()
    ik = DifferentialIk(
        model,
        data=data,
        collision_model=collision_model,
        options=options,
        visualizer=viz,
    )
    target_frame = "panda_hand"
    nullspace_components = [
        lambda model, q: joint_limit_nullspace_component(model, q, gain=0.5)
    ]

    # Solve IK several times and print the results
    for _ in range(10):
        init_state = get_random_collision_free_state(model, collision_model)
        target_tform = get_random_collision_free_transform(
            model, collision_model, target_frame
        )
        q_sol = ik.solve(
            target_frame,
            target_tform,
            init_state=init_state,
            nullspace_components=[],
            verbose=True,
        )
        print(f"Solution configuration:\n{q_sol}\n")
