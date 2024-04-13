import numpy as np
import pytest

from pyroboplan.trajectory.polynomial import (
    CubicPolynomialTrajectory,
    QuinticPolynomialTrajectory,
)


##########################
# CUBIC TRAJECTORY TESTS #
##########################
def test_single_dof_cubic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = np.array([0.0, -0.1, 0.1, 0.0])
    traj = CubicPolynomialTrajectory(t, q, qd)

    assert len(traj.segment_times) == 3
    assert traj.segment_times[0] == [0.0, 3.0]
    assert traj.segment_times[1] == [3.0, 6.0]
    assert traj.segment_times[2] == [6.0, 9.0]
    assert len(traj.coeffs) == 1
    assert len(traj.coeffs[0]) == 3  # 3 segments
    for coeffs in traj.coeffs[0]:
        assert len(coeffs) == 4  # 4 coefficients for cubics


def test_multi_dof_cubic_trajectory():
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
    traj = CubicPolynomialTrajectory(t, q, qd)

    assert len(traj.segment_times) == 3
    assert traj.segment_times[0] == [0.0, 3.0]
    assert traj.segment_times[1] == [3.0, 6.0]
    assert traj.segment_times[2] == [6.0, 9.0]
    assert len(traj.coeffs) == 2
    for single_dof_coeffs in traj.coeffs:
        assert len(single_dof_coeffs) == 3  # 3 segments
        for coeffs in single_dof_coeffs:
            assert len(coeffs) == 4  # 4 coefficients for cubics


def test_evaluate_cubic_trajectory_bad_time_values():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    traj = CubicPolynomialTrajectory(t, q, qd)

    with pytest.warns(UserWarning):
        assert traj.evaluate(-1.0) is None

    with pytest.warns(UserWarning):
        assert traj.evaluate(42.0) is None


def test_evaluate_single_dof_cubic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    traj = CubicPolynomialTrajectory(t, q, qd)

    q_eval, qd_eval, qdd_eval = traj.evaluate(3.0)
    assert q_eval == pytest.approx(1.0)
    assert qd_eval == pytest.approx(0.0)
    assert qdd_eval == pytest.approx(-2.0 / 3.0)

    q_eval, qd_eval, qdd_eval = traj.evaluate(4.5)
    assert q_eval == pytest.approx(-0.1)
    assert qd_eval == pytest.approx(-1.1)
    assert qdd_eval == pytest.approx(0.0)


def test_evaluate_multi_dof_cubic_trajectory():
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
    traj = CubicPolynomialTrajectory(t, q, qd)

    q_eval, qd_eval, qdd_eval = traj.evaluate(3.0)
    assert q_eval == pytest.approx([1.0, -1.0])
    assert qd_eval == pytest.approx([-0.1, 0.0])

    q_eval, qd_eval, qdd_eval = traj.evaluate(4.5)
    assert q_eval == pytest.approx([-0.175, 0.0])
    assert qd_eval == pytest.approx([-1.1, 1.0])


def test_generate_single_dof_cubic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = 0.0
    traj = CubicPolynomialTrajectory(t, q, qd)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (1, num_pts)
    assert qd.shape == (1, num_pts)
    assert qdd.shape == (1, num_pts)


def test_generate_multi_dof_cubic_trajectory():
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
    traj = CubicPolynomialTrajectory(t, q, qd)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (2, num_pts)
    assert qd.shape == (2, num_pts)
    assert qdd.shape == (2, num_pts)


############################
# QUINTIC TRAJECTORY TESTS #
############################
def test_single_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = np.array([0.0, -0.1, 0.1, 0.0])
    qdd = np.array([0.0, -0.25, 0.25, 0.0])
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    assert len(traj.segment_times) == 3
    assert traj.segment_times[0] == [0.0, 3.0]
    assert traj.segment_times[1] == [3.0, 6.0]
    assert traj.segment_times[2] == [6.0, 9.0]
    assert len(traj.coeffs) == 1
    assert len(traj.coeffs[0]) == 3  # 3 segments
    for coeffs in traj.coeffs[0]:
        assert len(coeffs) == 6  # 6 coefficients for quintics


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

    assert len(traj.segment_times) == 3
    assert traj.segment_times[0] == [0.0, 3.0]
    assert traj.segment_times[1] == [3.0, 6.0]
    assert traj.segment_times[2] == [6.0, 9.0]
    assert len(traj.coeffs) == 2
    for single_dof_coeffs in traj.coeffs:
        assert len(single_dof_coeffs) == 3  # 3 segments
        for coeffs in single_dof_coeffs:
            assert len(coeffs) == 6  # 6 coefficients for quintics


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


def test_evaluate_multi_dof_quintic_trajectory():
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
    qdd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.0],
        ]
    )
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    q_eval, qd_eval, qdd_eval = traj.evaluate(3.0)
    assert q_eval == pytest.approx([1.0, -1.0])
    assert qd_eval == pytest.approx([-0.1, 0.0])
    assert qdd_eval == pytest.approx([0.0, 0.25])

    q_eval, qd_eval, qdd_eval = traj.evaluate(4.5)
    assert q_eval == pytest.approx([-0.19375, 0.0703125])
    assert qd_eval == pytest.approx([-1.375, 1.25])
    assert qdd_eval == pytest.approx([0.1, -0.125])


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


def test_generate_multi_dof_quintic_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = np.array(
        [
            [0.0, -0.1, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    qdd = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.25, 0.0],
        ]
    )
    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)

    t, q, qd, qdd = traj.generate(dt=0.01)
    num_pts = len(t)
    assert q.shape == (1, num_pts)
    assert qd.shape == (1, num_pts)
    assert qdd.shape == (1, num_pts)
