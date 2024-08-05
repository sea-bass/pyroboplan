import copy
import numpy as np
import pinocchio
import pytest

from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import collision_avoidance_nullspace_component
from pyroboplan.models.panda import load_models, add_self_collisions


# Use a fixed seed for random number generation in tests.
np.random.seed(1234)


def test_ik_solve_trivial_ik():
    model, _, _ = load_models()
    data = model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform to the current joint state's FK
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q)
    target_tform = data.oMf[target_frame_id]

    # Solve IK
    options = DifferentialIkOptions()
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q,
        nullspace_components=[],
    )

    # The result should be very, very close to the initial state
    np.testing.assert_almost_equal(q, q_sol, decimal=6)


def test_ik_solve_ik():
    model, _, _ = load_models()
    data = model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q_init = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform 1 cm along z axis
    offset = 0.01
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_init)
    target_tform = copy.deepcopy(data.oMf[target_frame_id])
    target_tform.translation[2] = target_tform.translation[2] + offset

    # Solve IK
    options = DifferentialIkOptions()
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q_init,
        nullspace_components=[],
    )
    assert q_sol is not None

    # Get the resulting FK
    pinocchio.framesForwardKinematics(model, data, q_sol)
    result_tform = data.oMf[target_frame_id]

    # Assert that they are very, very close, lines up with the max pos/rot error from
    # the options.
    error = target_tform.actInv(result_tform)
    np.testing.assert_almost_equal(np.identity(3), error.rotation, decimal=3)
    np.testing.assert_almost_equal(np.zeros(3), error.translation, decimal=3)


def test_ik_solve_impossible_ik():
    model, _, _ = load_models()
    target_frame = "panda_hand"

    # Target is unreachable by the panda
    R = np.identity(3)
    T = np.array([10.0, 10.0, 10.0])
    target_tform = pinocchio.SE3(R, T)

    # Solve IK
    options = DifferentialIkOptions()
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=None,
        nullspace_components=[],
    )

    assert q_sol is None, "Solution should be impossible!"


def test_ik_in_collision():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)
    target_frame = "panda_hand"

    # Target is reachable, but in self-collision.
    R = np.array(
        [
            [0.206636, -0.430153, 0.878789],
            [-0.978337, -0.10235, 0.179946],
            [0.0125395, -0.896935, -0.441984],
        ],
    )
    T = np.array([0.155525, 0.0529695, 0.0259166])
    target_tform = pinocchio.SE3(R, T)

    # Solve IK
    options = DifferentialIkOptions()
    ik = DifferentialIk(
        model,
        collision_model=collision_model,
        data=None,
        options=options,
        visualizer=None,
    )
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=None,
        nullspace_components=[],
        verbose=False,
    )

    assert q_sol is None, "Solution should be in self-collision!"


def test_ik_with_nullspace_components():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()
    collision_data = collision_model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q_init = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform 1 cm along z axis
    offset = 0.01
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_init)
    target_tform = copy.deepcopy(data.oMf[target_frame_id])
    target_tform.translation[2] = target_tform.translation[2] + offset

    # Solve IK
    options = DifferentialIkOptions()
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q_init,
        nullspace_components=[
            lambda model, q: collision_avoidance_nullspace_component(
                model,
                data,
                collision_model,
                collision_data,
                q,
                gain=1.0,
                dist_padding=0.05,
            )
        ],
    )
    assert q_sol is not None

    # Get the resulting FK
    pinocchio.framesForwardKinematics(model, data, q_sol)
    result_tform = data.oMf[target_frame_id]

    # Assert that they are very, very close, lines up with the max pos/rot error from
    # the options.
    error = target_tform.actInv(result_tform)
    np.testing.assert_almost_equal(np.identity(3), error.rotation, decimal=3)
    np.testing.assert_almost_equal(np.zeros(3), error.translation, decimal=3)


def test_ik_solve_bad_joint_weights():
    model, _, _ = load_models()
    target_frame = "panda_hand"

    # Target is unreachable by the panda
    R = np.identity(3)
    T = np.array([10.0, 10.0, 10.0])
    target_tform = pinocchio.SE3(R, T)

    # Ignore the gripper joint indices.
    ignore_joint_indices = [
        model.getJointId("panda_finger_joint1") - 1,
        model.getJointId("panda_finger_joint2") - 1,
    ]

    # Solve IK with bad joint weight sizes.
    options = DifferentialIkOptions(
        joint_weights=[1.0, 2.0], ignore_joint_indices=ignore_joint_indices
    )
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    with pytest.raises(ValueError) as exc_info:
        ik.solve(
            target_frame,
            target_tform,
            init_state=None,
            nullspace_components=[],
        )

    assert (
        exc_info.value.args[0] == "Joint weights, if specified, must have 7 elements."
    )

    # Solve IK with non-positive joint weight values.
    options = DifferentialIkOptions(
        joint_weights=[0.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ignore_joint_indices=ignore_joint_indices,
    )
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    with pytest.raises(ValueError) as exc_info:
        ik.solve(
            target_frame,
            target_tform,
            init_state=None,
            nullspace_components=[],
        )

    assert exc_info.value.args[0] == "All joint weights must be strictly positive."


def test_ik_solve_ik_joint_weights():
    model, _, _ = load_models()
    data = model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q_init = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform 1 cm along z axis
    offset = 0.01
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_init)
    target_tform = copy.deepcopy(data.oMf[target_frame_id])
    target_tform.translation[2] = target_tform.translation[2] + offset

    # Ignore the gripper joint indices.
    ignore_joint_indices = [
        model.getJointId("panda_finger_joint1") - 1,
        model.getJointId("panda_finger_joint2") - 1,
    ]

    # Solve IK with joint weights for the 7 arm joints.
    options = DifferentialIkOptions(
        joint_weights=[10.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0],
        ignore_joint_indices=ignore_joint_indices,
    )
    ik = DifferentialIk(model, data=None, options=options, visualizer=None)
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q_init,
        nullspace_components=[],
    )
    assert q_sol is not None

    # Get the resulting FK
    pinocchio.framesForwardKinematics(model, data, q_sol)
    result_tform = data.oMf[target_frame_id]

    # Assert that they are very, very close, lines up with the max pos/rot error from
    # the options.
    error = target_tform.actInv(result_tform)
    np.testing.assert_almost_equal(np.identity(3), error.rotation, decimal=3)
    np.testing.assert_almost_equal(np.zeros(3), error.translation, decimal=3)
