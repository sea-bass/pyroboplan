"""Ownership and continuation regressions with genuine Pinocchio kinematics."""

import numpy as np
import pinocchio as pin
import pytest

from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.planning.cartesian_planner import (
    CartesianPlanner,
    CartesianPlannerOptions,
)


def model_and_data():
    model = pin.Model()
    first = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "first")
    second = model.addJoint(
        first,
        pin.JointModelRY(),
        pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
        "second",
    )
    model.addFrame(
        pin.Frame(
            "tip",
            second,
            0,
            pin.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    model.lowerPositionLimit[:] = -np.pi
    model.upperPositionLimit[:] = np.pi
    return model, model.createData()


def pose(model, data, q):
    pin.framesForwardKinematics(model, data, np.array(q))
    return pin.SE3(data.oMf[model.getFrameId("tip")])


def solver(model, data, **kwargs):
    return DifferentialIk(
        model,
        data=data,
        options=DifferentialIkOptions(max_retries=0, rng_seed=1, **kwargs),
    )


def assert_pose_close(expected, actual):
    error = pin.log(expected.actInv(actual)).vector
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
    unreachable = pin.SE3(np.eye(3), np.array([10.0, 0.0, 0.0]))
    planner = CartesianPlanner(
        model, "tip", [start, unreachable], solver(model, data, max_iters=2)
    )
    success, times, states = planner.generate(initial, 0.5)
    assert not success
    assert states.shape == (2, len(times))
    assert len(planner.generated_tforms) <= len(times)
    np.testing.assert_array_equal(initial, [0.1, 0.2])
