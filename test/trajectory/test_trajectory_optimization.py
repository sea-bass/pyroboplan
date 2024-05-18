import numpy as np
import pytest

from pyroboplan.models.panda import load_models
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)


# Use a fixed seed for random number generation in tests.
np.random.seed(1234)


def test_bad_traj_opt_options():
    # Insufficient waypoints
    with pytest.raises(ValueError):
        CubicTrajectoryOptimizationOptions(num_waypoints=1)

    # Non-positive min segment time
    with pytest.raises(ValueError):
        CubicTrajectoryOptimizationOptions(min_segment_time=-0.1)


def test_start_goal_traj_opt():
    model, _, _ = load_models()

    # Define the start and goal configurations
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_goal = np.array([0.785, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Perform trajectory optimization
    options = CubicTrajectoryOptimizationOptions(
        num_waypoints=3,
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=10.0,
        min_vel=-1.5,
        max_vel=1.5,
        min_accel=-0.75,
        max_accel=0.75,
        min_jerk=-1.0,
        max_jerk=1.0,
    )
    planner = CubicTrajectoryOptimization(model, options)
    with pytest.warns(RuntimeWarning):  # There are some invalid multiply values
        traj = planner.plan([q_start, q_goal])

    assert traj is not None
    assert traj.num_dims == 9
    assert len(traj.coeffs[0]) == options.num_waypoints - 1

    q, qd, _, _ = traj.evaluate(0.0)
    assert q == pytest.approx(q_start)
    assert qd == pytest.approx(np.zeros_like(qd))

    q, qd, _, _ = traj.evaluate(traj.segment_times[-1][-1])
    assert q == pytest.approx(q_goal)
    assert qd == pytest.approx(np.zeros_like(qd))

    waypoint_times = [seg_times[0] for seg_times in traj.segment_times[1:]]
    for t in waypoint_times:
        q, qd, qdd, qddd = traj.evaluate(t)
        assert np.all(q >= model.lowerPositionLimit)
        assert np.all(q <= model.upperPositionLimit)
        assert np.all(qd >= options.min_vel)
        assert np.all(qd <= options.max_vel)
        assert np.all(qdd >= options.min_accel)
        assert np.all(qdd <= options.max_accel)
        assert np.all(qddd >= options.min_jerk)
        assert np.all(qddd <= options.max_jerk)


def test_full_path_traj_opt():
    model, _, _ = load_models()

    # Define a multi-configuration path
    q_path = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.00, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0]),
        np.array([0.785, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    # Perform trajectory optimization
    options = CubicTrajectoryOptimizationOptions(
        num_waypoints=len(q_path),
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=10.0,
        min_vel=-1.5,
        max_vel=1.5,
        min_accel=-0.75,
        max_accel=0.75,
        min_jerk=-1.0,
        max_jerk=1.0,
    )
    planner = CubicTrajectoryOptimization(model, options)
    with pytest.warns(RuntimeWarning):  # There are some invalid multiply values
        traj = planner.plan(q_path)

    assert traj is not None
    assert traj.num_dims == 9
    assert len(traj.coeffs[0]) == options.num_waypoints - 1

    t_final = traj.segment_times[-1][-1]
    q, qd, _, _ = traj.evaluate(0.0)
    assert q == pytest.approx(q_path[0])
    assert qd == pytest.approx(np.zeros_like(qd))

    q, qd, _, _ = traj.evaluate(t_final)
    assert q == pytest.approx(q_path[-1])
    assert qd == pytest.approx(np.zeros_like(qd))

    waypoint_times = [seg_times[0] for seg_times in traj.segment_times[1:]]
    for idx, t in enumerate(waypoint_times):
        q, qd, qdd, qddd = traj.evaluate(t)
        assert q == pytest.approx(q_path[idx + 1])
        assert np.all(q >= model.lowerPositionLimit)
        assert np.all(q <= model.upperPositionLimit)
        assert np.all(qd >= options.min_vel)
        assert np.all(qd <= options.max_vel)
        assert np.all(qdd >= options.min_accel)
        assert np.all(qdd <= options.max_accel)
        assert np.all(qddd >= options.min_jerk)
        assert np.all(qddd <= options.max_jerk)
