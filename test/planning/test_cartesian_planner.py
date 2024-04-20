import numpy as np
import pinocchio
import pytest
from scipy.spatial.transform import Rotation

from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.planning.cartesian_planner import (
    CartesianPlanner,
    CartesianPlannerOptions,
)


@pytest.mark.parametrize(
    "max_vals",
    [
        (0.0, 1.0, 1.0, 1.0),
        (1.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 0.0),
    ],
)
def test_invalid_cartesian_planner_options(max_vals):
    with pytest.raises(ValueError):
        CartesianPlannerOptions(
            max_linear_velocity=max_vals[0],
            max_linear_acceleration=max_vals[1],
            max_angular_velocity=max_vals[2],
            max_angular_acceleration=max_vals[3],
        )


@pytest.mark.parametrize("use_trapezoidal_scaling", [(True,), (False,)])
def test_cartesian_planner(use_trapezoidal_scaling):
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()

    # Define the Cartesian path from a start joint configuration
    target_frame = "panda_hand"
    q_start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
    init = extract_cartesian_pose(model, target_frame, q_start, data=data)
    rot = Rotation.from_euler("z", 60, degrees=True).as_matrix()
    tforms = [
        init,
        init * pinocchio.SE3(rot, np.array([0.0, 0.0, 0.2])),
        init * pinocchio.SE3(np.eye(3), np.array([0.0, 0.2, 0.2])),
    ]

    # Define the IK solve to use.
    ik = DifferentialIk(
        model,
        data=data,
        collision_model=collision_model,
    )

    # Perform Cartesian planning.
    dt = 0.05
    options = CartesianPlannerOptions(
        use_trapezoidal_scaling=use_trapezoidal_scaling,
    )
    planner = CartesianPlanner(model, target_frame, tforms, ik, options=options)
    success, t_vec, q_vec = planner.generate(q_start, dt)

    assert success
    assert q_vec.shape[0] == model.nq
    assert q_vec.shape[1] == len(t_vec)


def test_cartesian_planner_failure():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()

    # Define the Cartesian path from a start joint configuration.
    # This pose is intentionally too far to be reachable, so we expect planning to fail.
    target_frame = "panda_hand"
    q_start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
    init = extract_cartesian_pose(model, target_frame, q_start, data=data)
    tforms = [
        init,
        init * pinocchio.SE3(np.eye(3), np.array([10.0, 10.0, 1.0])),
    ]

    # Define the IK solve to use.
    ik = DifferentialIk(
        model,
        data=data,
        collision_model=collision_model,
    )

    # Perform Cartesian planning.
    dt = 0.05
    planner = CartesianPlanner(model, target_frame, tforms, ik)
    success, _, _ = planner.generate(q_start, dt)

    assert not success
