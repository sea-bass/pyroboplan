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
from pyroboplan.ik.nullspace_components import (
    joint_limit_nullspace_component,
    collision_avoidance_nullspace_component,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model, inflation_radius=0.1)

    data = model.createData()
    collision_data = collision_model.createData()

    target_frame = "panda_hand"
    ignore_joint_indices = [
        model.getJointId("panda_finger_joint1") - 1,
        model.getJointId("panda_finger_joint2") - 1,
    ]

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    np.set_printoptions(precision=3)

    # Set up the IK solver
    options = DifferentialIkOptions(
        damping=0.0001,
        min_step_size=0.025,
        max_step_size=0.1,
        ignore_joint_indices=ignore_joint_indices,
        rng_seed=None,
    )
    ik = DifferentialIk(
        model,
        data=data,
        collision_model=collision_model,
        options=options,
        visualizer=viz,
    )
    nullspace_components = [
        lambda model, q: collision_avoidance_nullspace_component(
            model,
            data,
            collision_model,
            collision_data,
            q,
            gain=1.0,
            dist_padding=0.05,
        ),
        lambda model, q: joint_limit_nullspace_component(
            model, q, gain=0.1, padding=0.025
        ),
    ]

    # Solve IK several times and print the results
    for _ in range(10):
        init_state = get_random_collision_free_state(model, collision_model)
        target_tform = get_random_collision_free_transform(
            model, collision_model, target_frame, joint_padding=0.05,
        )
        q_sol = ik.solve(
            target_frame,
            target_tform,
            init_state=init_state,
            nullspace_components=nullspace_components,
            verbose=True,
        )
        print(f"Solution configuration:\n{q_sol}\n")
