import numpy as np
import pytest

from pyroboplan.trajectory.polynomial import QuinticPolynomialTrajectory


def test_single_dof_trajectory():
    t = np.array([0.0, 3.0, 6.0, 9.0])
    q = np.array([0.0, 1.0, -1.2, 0.25])
    qd = np.array([0.0, -0.1, 0.1, 0.0])
    qdd = np.array([0.0, -0.25, 0.25, 0.0])

    traj = QuinticPolynomialTrajectory(t, q, qd, qdd)
    traj.visualize()


def test_multi_dof_trajectory():
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
    traj.visualize()


if __name__ == "__main__":
    test_multi_dof_trajectory()
