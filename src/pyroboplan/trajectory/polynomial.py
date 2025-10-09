"""
Implementations of polynomial trajectories.

Some good resources:
  * Video with derivation: https://robotacademy.net.au/lesson/1d-polynomial-trajectory
  * Chapter 9 of this book: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
"""

from abc import ABC
from typing import List, Optional, Union
import numpy as np
import numpy.typing as npt
import warnings


class PolynomialTrajectoryBase(ABC):
    """Abstract base class for polynomial trajectories."""
    num_dims: int
    coeffs: List[List[npt.NDArray[np.float32]]]
    segment_times: List[List[np.float32]]

    def evaluate(self, t: float):
        """
        Evaluates the trajectory at a specific time.

        Parameters
        ----------
            t : float
                The time, in seconds, at which to evaluate the trajectory.

        Returns
        -------
            tuple(array-like or float, array-like or float, array-like or float, array-like or float)
                The tuple of (q, qd, qdd, qddd) trajectory values at the specified time.
                If the trajectory is single-DOF, these will be scalars, otherwise they are arrays.
        """
        q = np.zeros(self.num_dims)  # Position
        qd = np.zeros(self.num_dims)  # Velocity
        qdd = np.zeros(self.num_dims)  # Acceleration
        qddd = np.zeros(self.num_dims)  # Jerk

        # Handle edge cases.
        if t < self.segment_times[0][0]:
            warnings.warn("Cannot evaluate trajectory before its start time.")
            return None
        elif t > self.segment_times[-1][-1]:
            warnings.warn("Cannot evaluate trajectory after its end time.")
            return None

        segment_idx = 0
        evaluated = False
        while not evaluated:
            t_segment_start = self.segment_times[segment_idx][0]
            t_segment_final = self.segment_times[segment_idx][-1]
            if t <= t_segment_final:
                for dim in range(self.num_dims):
                    coeffs = self.coeffs[dim][segment_idx]
                    dt = t - t_segment_start
                    q[dim] = np.polyval(coeffs, dt)
                    qd[dim] = np.polyval(np.polyder(coeffs, 1), dt)
                    qdd[dim] = np.polyval(np.polyder(coeffs, 2), dt)
                    qddd[dim] = np.polyval(np.polyder(coeffs, 3), dt)
                evaluated = True
            else:
                segment_idx += 1

        # If the trajectory is single-DOF, return the values as scalars.
        if len(q) == 1:
            q = q[0]
            qd = qd[0]
            qdd = qdd[0]
            qddd = qddd[0]
        return q, qd, qdd, qddd

    def generate(self, dt: float):
        """
        Generates a full trajectory at a specified sample time.

        Parameters
        ----------
            dt : float
                The sample time, in seconds.

        Returns
        -------
            tuple(array-like, array-like, array-like, array-like, array-like)
                The tuple of (t, q, qd, qdd, qddd) trajectory values generated at the sample time.
                The time vector is 1D, and the others are 2D (time and dimension).
        """

        # Initialize the data structure
        t_start = self.segment_times[0][0]
        t_final = self.segment_times[-1][-1]
        t_vec = np.arange(t_start, t_final, dt)
        if t_vec[-1] != t_final:
            t_vec = np.append(t_vec, t_final)
        num_pts = len(t_vec)

        q = np.zeros([self.num_dims, num_pts])  # Position
        qd = np.zeros([self.num_dims, num_pts])  # Velocity
        qdd = np.zeros([self.num_dims, num_pts])  # Acceleration
        qddd = np.zeros([self.num_dims, num_pts])  # Jerk

        t_idx = 0
        segment_idx = 0

        while t_idx < num_pts:
            t = t_vec[t_idx]
            t_segment_start = self.segment_times[segment_idx][0]
            t_segment_final = self.segment_times[segment_idx][-1]
            if t <= t_segment_final:
                for dim in range(self.num_dims):
                    coeffs = self.coeffs[dim][segment_idx]
                    dt = t - t_segment_start
                    q[dim, t_idx] = np.polyval(coeffs, dt)
                    qd[dim, t_idx] = np.polyval(np.polyder(coeffs, 1), dt)
                    qdd[dim, t_idx] = np.polyval(np.polyder(coeffs, 2), dt)
                    qddd[dim, t_idx] = np.polyval(np.polyder(coeffs, 3), dt)
                t_idx += 1
            else:
                segment_idx += 1

        return t_vec, q, qd, qdd, qddd

    def visualize(
        self,
        dt: float = 0.01,
        joint_names: Optional[List[str]] = None,
        show_position: bool = True,
        show_velocity: bool = False,
        show_acceleration: bool = False,
        show_jerk: bool = False,
    ):
        """
        Visualizes a generated trajectory with one figure window per dimension.

        Parameters
        ----------
            dt : float, optional
                The sample time at which to generate the trajectory by evaluating the polynomial.
            joint_names : list[str], optional
                The joint names to use for the figure titles.
                If unset, uses the text "Dimension 1", "Dimension 2", etc.
            show_position : bool, optional
                If true, shows the trajectory position values.
            show_velocity : bool, optional
                If true, shows the trajectory velocity values.
            show_acceleration : bool, optional
                If true, shows the trajectory acceleration values.
            show_jerk : bool, optional
                If true, shows the trajectory jerk values.
        """
        import matplotlib.pyplot as plt

        t_vec, q, qd, qdd, qddd = self.generate(dt)
        vertical_line_scale_factor = 1.2

        for dim in range(self.num_dims):
            if joint_names is not None:
                title = joint_names[dim]
            else:
                title = f"Dimension {dim + 1}"
            plt.figure(title)
            plt.cla()

            # Positions, velocities, and accelerations
            legend = []
            min_pos = min_vel = min_accel = min_jerk = np.inf
            max_pos = max_vel = max_accel = max_jerk = -np.inf
            if show_position:
                plt.plot(t_vec, q[dim, :])
                min_pos = np.min(q[dim, :])
                max_pos = np.max(q[dim, :])
                legend.append("Position")
            if show_velocity:
                plt.plot(t_vec, qd[dim, :])
                min_vel = np.min(qd[dim, :])
                max_vel = np.max(qd[dim, :])
                legend.append("Velocity")
            if show_acceleration:
                plt.plot(t_vec, qdd[dim, :])
                min_accel = np.min(qdd[dim, :])
                max_accel = np.max(qdd[dim, :])
                legend.append("Acceleration")
            if show_jerk:
                plt.plot(t_vec, qddd[dim, :])
                min_jerk = np.min(qddd[dim, :])
                max_jerk = np.max(qddd[dim, :])
                legend.append("Jerk")
            plt.legend(legend)

            # Times
            min_val = vertical_line_scale_factor * min(
                [min_pos, min_vel, min_accel, min_jerk]
            )
            max_val = vertical_line_scale_factor * max(
                [max_pos, max_vel, max_accel, max_jerk]
            )
            plt.vlines(
                [t[1] for t in self.segment_times],
                min_val,
                max_val,
                colors="k",
                linestyles=":",
                linewidth=1.0,
            )

        plt.show()


class CubicPolynomialTrajectory(PolynomialTrajectoryBase):
    """Describes a cubic (3rd-order) polynomial trajectory."""

    def __init__(self,
                 t: npt.NDArray[np.float32],
                 q: npt.NDArray[np.float32],
                 qd: Union[float, npt.NDArray[np.float32]] = 0.0):
        """
        Creates a cubic (3rd-order) polynomial trajectory.

        Parameters
        ----------
            t : array-like
                An array of length N corresponding to the time breakpoints.
            q : array-like
                A M-by-N array of M dimensions and N waypoints corresponding to the position breakpoints.
            qd : array-like, optional
                A M-by-N array of M dimensions and N waypoints corresponding to the velocity breakpoints.
                It can also be a scalar or a 1D array to copy to all dimensions/breakpoints.
                Defaults to zero endpoint velocities.
        """
        # Validate inputs.
        num_segments = len(t) - 1

        if len(q.shape) == 1:
            q = q[np.newaxis, :]
        if q.shape[1] != num_segments + 1:
            raise ValueError(
                "Position breakpoints must have the same number of columns as the time breakpoints."
            )
        self.num_dims = q.shape[0]

        if isinstance(qd, float):
            qd = qd * np.ones_like(q)
        if len(qd.shape) == 1:
            qd = qd[np.newaxis, :]

        # Fit cubic trajectory coefficients to each segment.
        self.segment_times = []
        self.coeffs = [[] for _ in range(self.num_dims)]
        for idx in range(num_segments):
            t_breakpoints = [t[idx], t[idx + 1]]
            self.segment_times.append(t_breakpoints)

            for dim in range(self.num_dims):
                # These are in the form:
                # q[0], q[T], qd[0], qd[T]
                endpoints = np.array(
                    [
                        q[dim, idx],
                        q[dim, idx + 1],
                        qd[dim, idx],
                        qd[dim, idx + 1],
                    ]
                )
                T = t_breakpoints[1] - t_breakpoints[0]
                M = np.array(
                    [
                        [0, 0, 0, 1],
                        [T**3, T**2, T, 1],
                        [0, 0, 1, 0],
                        [3 * T**2, 2 * T, 1, 0],
                    ]
                )
                coeffs = np.linalg.solve(M, endpoints)
                self.coeffs[dim].append(coeffs)


class QuinticPolynomialTrajectory(PolynomialTrajectoryBase):
    """Describes a quintic (5th-order) polynomial trajectory."""

    def __init__(self, t, q, qd=0.0, qdd=0.0):
        """
        Creates a quintic (5th-order) polynomial trajectory.

        Parameters
        ----------
            t : array-like
                An array of length N corresponding to the time breakpoints.
            q : array-like
                A M-by-N array of M dimensions and N waypoints corresponding to the position breakpoints.
            qd : array-like, optional
                A M-by-N array of M dimensions and N waypoints corresponding to the velocity breakpoints.
                It can also be a scalar or a 1D array to copy to all dimensions/breakpoints.
                Defaults to zero endpoint velocities.
            qdd : array-like, optional
                A M-by-N array of M dimensions and N waypoints corresponding to the acceleration breakpoints.
                It can also be a scalar or a 1D array to copy to all dimensions/breakpoints.
                Defaults to zero endpoint accelerations.
        """
        # Validate inputs.
        num_segments = len(t) - 1

        if len(q.shape) == 1:
            q = q[np.newaxis, :]
        if q.shape[1] != num_segments + 1:
            raise ValueError(
                "Position breakpoints must have the same number of columns as the time breakpoints."
            )
        self.num_dims = q.shape[0]

        if isinstance(qd, float):
            qd = qd * np.ones_like(q)
        if len(qd.shape) == 1:
            qd = qd[np.newaxis, :]

        if isinstance(qdd, float):
            qdd = qdd * np.ones_like(q)
        if len(qdd.shape) == 1:
            qdd = qdd[np.newaxis, :]

        # Fit quintic trajectory coefficients to each segment.
        self.segment_times = []
        self.coeffs = [[] for _ in range(self.num_dims)]
        for idx in range(num_segments):
            t_breakpoints = [t[idx], t[idx + 1]]
            self.segment_times.append(t_breakpoints)

            for dim in range(self.num_dims):
                # These are in the form:
                # q[0], q[T], qd[0], qd[T], qdd[0], qdd[T]
                endpoints = np.array(
                    [
                        q[dim, idx],
                        q[dim, idx + 1],
                        qd[dim, idx],
                        qd[dim, idx + 1],
                        qdd[dim, idx],
                        qdd[dim, idx + 1],
                    ]
                )
                T = t_breakpoints[1] - t_breakpoints[0]
                M = np.array(
                    [
                        [0, 0, 0, 0, 0, 1],
                        [T**5, T**4, T**3, T**2, T, 1],
                        [0, 0, 0, 0, 1, 0],
                        [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                        [0, 0, 0, 2, 0, 0],
                        [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
                    ]
                )
                coeffs = np.linalg.solve(M, endpoints)
                self.coeffs[dim].append(coeffs)
