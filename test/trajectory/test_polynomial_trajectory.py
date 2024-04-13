import numpy as np
import pytest

from pyroboplan.trajectory.polynomial import QuinticPolynomialTrajectory


def test_single_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = np.array([0.0, -0.1, 0.1, 0.0])
    qdd = np.array([0.0, -0.25, 0.25, 0.0])

    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)


def test_multi_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array(
        [
            [0.0, 1.0, -1.2, 0.25],
            [0.0, -1.0, 1.0, 0.0],
        ]
    )
    qd = np.array(
        [
            [0.0, -0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    qdd = np.array([[0.0, -0.25, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0]])

    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)


def test_evaluate_quintic_trajectory_bad_time_values():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    qdd = 0.0
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    with pytest.warns(UserWarning):
        assert traj.evaluate(-1.0) is None

    with pytest.warns(UserWarning):
        assert traj.evaluate(42.0) is None


def test_evaluate_single_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    qdd = 0.0
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    q_eval, qd_eval, qdd_eval = traj.evaluate(3.0)
    assert q_eval == pytest.approx(1.0)
    assert qd_eval == pytest.approx(0.0)
    assert qdd_eval == pytest.approx(0.0)

    q_eval, qd_eval, qdd_eval = traj.evaluate(4.5)
    assert q_eval == pytest.approx(-0.1)
    assert qd_eval == pytest.approx(-1.375)
    assert qdd_eval == pytest.approx(0.0)


def test_generate_single_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    qdd = 0.0
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (1, num_pts)
    assert qd.shape == (1, num_pts)
    assert qdd.shape == (1, num_pts)
