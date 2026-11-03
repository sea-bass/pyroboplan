import numpy as np
import pinocchio
import pytest
from scipy.spatial.transform import Rotation

from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
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


@pytest.mark.parametrize("use_trapezoidal_scaling", [True, False])
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
        init * pinocchio.SE3(np.eye(3), np.array([10.0, 10.0, 10.0])),
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


def model_and_data():
    model = pinocchio.Model()
    first = model.addJoint(
        0, pinocchio.JointModelRZ(), pinocchio.SE3.Identity(), "first"
    )
    second = model.addJoint(
        first,
        pinocchio.JointModelRY(),
        pinocchio.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
        "second",
    )
    model.addFrame(
        pinocchio.Frame(
            "tip",
            second,
            0,
            pinocchio.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
            pinocchio.FrameType.OP_FRAME,
        )
    )
    model.lowerPositionLimit[:] = -np.pi
    model.upperPositionLimit[:] = np.pi
    return model, model.createData()


def pose(model, data, q):
    pinocchio.framesForwardKinematics(model, data, np.array(q))
    return pinocchio.SE3(data.oMf[model.getFrameId("tip")])


def solver(model, data, **kwargs):
    return DifferentialIk(
        model,
        data=data,
        options=DifferentialIkOptions(max_retries=0, rng_seed=1, **kwargs),
    )


def assert_pose_close(expected, actual):
    error = pinocchio.log(expected.actInv(actual)).vector
    assert np.linalg.norm(error[:3]) < 1e-3
    assert np.linalg.norm(error[3:]) < 1e-3


def test_borrowed_target_is_snapshotted_before_forward_kinematics():
    model, data = model_and_data()
    expected = pose(model, data, [0.4, 0.2])
    borrowed = data.oMf[model.getFrameId("tip")]
    result = solver(model, data).solve("tip", borrowed, init_state=np.array([0.1, 0.1]))
    assert result is not None
    # The scratch Data view may change; the desired pose must not drift with it.
    assert_pose_close(expected, pose(model, model.createData(), result))


@pytest.mark.parametrize("kind", ["array", "readonly", "list"])
def test_ik_does_not_modify_initial_state(kind):
    model, data = model_and_data()
    target = pose(model, data, [0.4, 0.2])
    initial = np.array([0.1, 0.1])
    if kind == "readonly":
        initial.flags.writeable = False
    elif kind == "list":
        initial = initial.tolist()
    result = solver(model, data).solve("tip", target, init_state=initial)
    assert result is not None
    np.testing.assert_array_equal(initial, [0.1, 0.1])
    assert_pose_close(target, pose(model, model.createData(), result))


def test_ignored_joint_stays_fixed_without_restarts():
    model, data = model_and_data()
    target = pose(model, data, [0.4, 0.2])
    initial = np.array([0.1, 0.2])
    result = solver(model, data, ignore_joint_indices=[1]).solve(
        "tip", target, init_state=initial
    )
    assert result is not None
    np.testing.assert_allclose(result[1], initial[1], rtol=0, atol=1e-14)
    np.testing.assert_array_equal(initial, [0.1, 0.2])
    assert_pose_close(target, pose(model, model.createData(), result))


def test_planner_owns_waypoint_snapshots():
    model, data = model_and_data()
    start = pose(model, data, [0.1, 0.1])
    borrowed = data.oMf[model.getFrameId("tip")]
    end = pose(model, model.createData(), [0.2, 0.15])
    waypoints = [borrowed, end]
    planner = CartesianPlanner(model, "tip", waypoints, solver(model, data))
    pose(model, data, [0.6, 0.5])
    end.translation[:] = 99.0
    waypoints.clear()
    np.testing.assert_array_equal(planner.tforms[0].homogeneous, start.homogeneous)
    assert len(planner.tforms) == 2
    assert np.linalg.norm(planner.tforms[1].translation) < 3.0


@pytest.mark.parametrize("trapezoidal", [True, False])
@pytest.mark.parametrize("seed_kind", ["array", "readonly", "list"])
def test_real_multi_waypoint_plan_tracks_targets_and_repeats(trapezoidal, seed_kind):
    model, data = model_and_data()
    initial = np.array([0.1, 0.2])
    waypoints = [pose(model, data, q) for q in ([0.1, 0.2], [0.15, 0.2], [0.2, 0.2])]
    # Borrow the first waypoint from the same scratch Data used by IK.
    pose(model, data, initial)
    waypoints[0] = data.oMf[model.getFrameId("tip")]
    original = [p.homogeneous.copy() for p in waypoints]
    if seed_kind == "readonly":
        initial.flags.writeable = False
    elif seed_kind == "list":
        initial = initial.tolist()
    ik = solver(model, data, ignore_joint_indices=[1])
    planner = CartesianPlanner(
        model,
        "tip",
        waypoints,
        ik,
        CartesianPlannerOptions(use_trapezoidal_scaling=trapezoidal),
    )
    success, times, states = planner.generate(initial, 0.02)
    assert success
    assert states.shape == (2, len(times))
    assert np.ptp(states[0]) > 0.08
    np.testing.assert_allclose(states[1], 0.2, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(initial, [0.1, 0.2])
    for expected, q in zip(planner.generated_tforms, states.T):
        assert_pose_close(expected, pose(model, model.createData(), q))
    again, times2, states2 = planner.generate(initial, 0.02)
    assert again
    np.testing.assert_array_equal(times, times2)
    np.testing.assert_array_equal(states, states2)
    # The caller's borrowed scratch view can move; owned planner snapshots cannot.
    for current, expected in zip(planner.tforms, original):
        np.testing.assert_array_equal(current.homogeneous, expected)
    for current, expected in zip(waypoints[1:], original[1:]):
        np.testing.assert_array_equal(current.homogeneous, expected)


def test_planner_advances_returned_solution_without_solver_side_effects():
    model, data = model_and_data()
    waypoints = [pose(model, data, q) for q in ([0.1, 0.2], [0.2, 0.2], [0.3, 0.2])]
    initial = np.array([0.1, 0.2])

    class RecordingSolver:
        def __init__(self):
            self.seeds = []
            self.results = []

        def solve(self, frame, target, init_state):
            self.seeds.append(np.array(init_state, copy=True))
            result = np.array(init_state, copy=True) + [0.01, 0.0]
            self.results.append(result.copy())
            return result

    ik = RecordingSolver()
    planner = CartesianPlanner(model, "tip", waypoints, ik)
    success, _, _ = planner.generate(initial, 0.05)
    assert success
    assert len(ik.seeds) > 2
    for seed, previous in zip(ik.seeds[1:], ik.results[:-1]):
        np.testing.assert_array_equal(seed, previous)
    np.testing.assert_array_equal(initial, [0.1, 0.2])


def test_failure_preserves_caller_state_and_return_shape():
    model, data = model_and_data()
    initial = np.array([0.1, 0.2])
    start = pose(model, data, initial)
    unreachable = pinocchio.SE3(np.eye(3), np.array([10.0, 0.0, 0.0]))
    planner = CartesianPlanner(
        model, "tip", [start, unreachable], solver(model, data, max_iters=2)
    )
    success, times, states = planner.generate(initial, 0.5)
    assert not success
    assert states.shape == (2, len(times))
    assert len(planner.generated_tforms) <= len(times)
    np.testing.assert_array_equal(initial, [0.1, 0.2])
