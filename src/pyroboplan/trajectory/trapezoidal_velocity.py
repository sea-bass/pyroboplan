import numpy as np
import warnings


class TrapezoidalVelocityTrajectorySingleDofContainer:
    """
    Helper data structure containing data for a single-dof trapezoidal velocity profile trajectory.

    This class should not be used standalone.
    """

    def __init__(self):
        self.times = []
        self.positions = []
        self.velocities = []
        self.accelerations = []


class TrapezoidalVelocityTrajectory:
    """
    Describes a trapezoidal velocity profile trajectory.

    Some good resources:
      * Chapter 9 of this book: https://hades.mech.northwestern.edu/images/7/7f/MR.pdf
      * MATLAB documentation: https://www.mathworks.com/help/robotics/ref/trapveltraj.html
    """

    def __init__(self, q, qd_max, qdd_max, t_start=0.0):
        """
        Creates a trapezoidal velocity profile trajectory.

        Parameters
        ----------
            q : array-like
                A M-by-N array of M dimensions and N waypoints.
            qd_max : array-like or float
                The maximum velocity for each dimension (size N-by-1), or scalar for all dimensions.
            qdd_max : array-like or float
                The maximum acceleration for each dimension (size N-by-1), or scalar for all dimensions.
            t_start : float, optional
                The start time of the trajectory. Defaults to zero.
        """
        if len(q.shape) == 1:
            q = q[np.newaxis, :]
        self.waypoints = q
        self.num_dims = q.shape[0]

        if isinstance(qd_max, float) or len(qd_max) == 1:
            qd_max = qd_max * np.ones(self.num_dims)
        if isinstance(qdd_max, float) == 1:
            qdd_max = qdd_max * np.ones(self.num_dims)

        self.segment_times = [t_start]
        self.single_dof_trajectories = [
            TrapezoidalVelocityTrajectorySingleDofContainer()
            for _ in range(self.num_dims)
        ]
        for dim, traj in enumerate(self.single_dof_trajectories):
            traj.times.append(t_start)
            traj.positions.append(q[dim][0])
            traj.velocities.append(0.0)

        # Iterate through all the waypoints
        delta_q_vec = np.diff(q, axis=1)
        for idx in range(delta_q_vec.shape[1]):
            t_prev = self.segment_times[-1]
            delta_q = delta_q_vec[:, idx]
            delta_q_abs = np.abs(delta_q)

            # Find the limiting velocity and acceleration by normalizing.
            # This can lead to divisions by zero, which we can ignore here.
            with np.errstate(divide="ignore"):
                qd_max_norm = np.min(qd_max / delta_q_abs)
                qdd_max_norm = np.min(qdd_max / delta_q_abs)

            # If there is a zero displacement segment, skip it!
            if not np.isfinite(qd_max_norm) or not np.isfinite(qdd_max_norm):
                continue

            accel = qdd_max_norm * delta_q

            # Check if "bang-bang" control at constant acceleration is feasible.
            # This is determined by seeing if the position moved by ramping up
            # to max velocity and then immediately back down is greater than 1,
            # in normalized coordinates.
            #
            #      -------- qd_max
            #     /\
            #    /  \  <--- slope = qdd_max
            #   /    \
            #   <----> ---- 0.0
            #  t_segment
            #
            # Some key equations:
            #  qdd_max = 2 * qd_max / t_segment          (slope at constant acceleration)
            #  delta_q = 0.5 * t_segment * qd_max = 1.0  (area under the triangle)
            #
            # By using normalized positions and substituting in the equations above,
            # this is equivalent to checking that:
            #  qd_max ^ 2 >= qdd_max

            if qd_max_norm**2 >= qdd_max_norm:
                # This is possible without a zero-acceleration segment.
                # We must now find the qd_peak value, which could be less than
                # the maximum, to achieve a (normalized) delta_q value of 1.0.
                #
                #      -------- qd_peak
                #     /\
                #    /  \  <--- slope = qdd_peak
                #   /    \
                #   <----> ---- 0.0
                #  t_segment
                #
                # The equations are the same as above:
                #  qdd_max = 2 * qd_peak / t_segment          (slope at constant acceleration)
                #  delta_q = 0.5 * t_segment * qd_peak = 1.0  (area under the triangle)

                qd_peak_norm = np.sqrt(qdd_max_norm)
                t_segment = 2.0 * qd_peak_norm / qdd_max_norm

                # Unpack into the separate dimensions by unnormalizing speed and acceleration.
                qd_peak = qd_peak_norm * delta_q
                for dim in range(self.num_dims):
                    q_prev = self.single_dof_trajectories[dim].positions[-1]

                    times = [t_prev + t_segment / 2.0, t_prev + t_segment]
                    positions = [
                        q_prev + delta_q[dim] / 2.0,
                        q_prev + delta_q[dim],
                    ]
                    velocities = [qd_peak[dim], 0.0]
                    accelerations = [accel[dim], -accel[dim]]

                    self.single_dof_trajectories[dim].times.extend(times)
                    self.single_dof_trajectories[dim].positions.extend(positions)
                    self.single_dof_trajectories[dim].velocities.extend(velocities)
                    self.single_dof_trajectories[dim].accelerations.extend(
                        accelerations
                    )
            else:
                # This requires a zero-acceleration segment as follows.
                #
                #      ---------     --- qd_max
                #     /         \
                #    /           \   <--- slope = qdd_max
                #   /             \
                #      |<----->|     --- 0
                #     t_zero_accel
                #  |<------------->|
                #      t_segment

                t_const_accel = 2.0 * qd_max_norm / qdd_max_norm
                delta_q_norm_const_accel = 0.5 * t_const_accel * qd_max_norm
                delta_q_norm_zero_accel = 1.0 - delta_q_norm_const_accel
                t_zero_accel = delta_q_norm_zero_accel / qd_max_norm
                t_segment = t_const_accel + t_zero_accel

                # Unpack into the separate dimensions by unnormalizing speed and acceleration.
                qd_peak = qd_max_norm * delta_q
                delta_q_const_accel = delta_q_norm_const_accel * delta_q
                delta_q_zero_accel = delta_q_norm_zero_accel * delta_q
                for dim in range(self.num_dims):
                    q_prev = self.single_dof_trajectories[dim].positions[-1]

                    times = [
                        t_prev + t_const_accel / 2.0,
                        t_prev + t_const_accel / 2.0 + t_zero_accel,
                        t_prev + t_const_accel + t_zero_accel,
                    ]
                    positions = [
                        q_prev + delta_q_const_accel[dim] / 2.0,
                        q_prev
                        + delta_q_const_accel[dim] / 2.0
                        + delta_q_zero_accel[dim],
                        q_prev + delta_q[dim],
                    ]
                    velocities = [qd_peak[dim], qd_peak[dim], 0.0]
                    accelerations = [accel[dim], 0.0, -accel[dim]]

                    self.single_dof_trajectories[dim].times.extend(times)
                    self.single_dof_trajectories[dim].positions.extend(positions)
                    self.single_dof_trajectories[dim].velocities.extend(velocities)
                    self.single_dof_trajectories[dim].accelerations.extend(
                        accelerations
                    )

            self.segment_times.append(t_prev + t_segment)

    def evaluate(self, t):
        """
        Evaluates the trajectory at a specific time.

        Parameters
        ----------
            t : float
                The time, in seconds, at which to evaluate the trajectory.

        Returns
        -------
            tuple(array-like or float, array-like or float, array-like or float)
                The tuple of (q, qd, qdd) trajectory values at the specified time.
                If the trajectory is single-DOF, these will be scalars, otherwise they are arrays.
        """
        q = np.zeros(self.num_dims)
        qd = np.zeros(self.num_dims)
        qdd = np.zeros(self.num_dims)

        # Handle edge cases.
        if t < self.segment_times[0]:
            warnings.warn("Cannot evaluate trajectory before its start time.")
            return None
        elif t > self.segment_times[-1]:
            warnings.warn("Cannot evaluate trajectory after its end time.")
            return None

        for dim in range(self.num_dims):
            segment_idx = 0
            traj = self.single_dof_trajectories[dim]
            evaluated_segment = False
            while not evaluated_segment:
                t_next = traj.times[segment_idx + 1]
                t_prev = traj.times[segment_idx]
                t_segment = t_next - t_prev
                if t <= t_next:
                    q_prev = traj.positions[segment_idx]
                    t_prev = traj.times[segment_idx]
                    v_next = traj.velocities[segment_idx + 1]
                    v_prev = traj.velocities[segment_idx]

                    dt = t - t_prev
                    dv = (v_next - v_prev) * dt / t_segment
                    q[dim] = q_prev + (0.5 * dv + v_prev) * dt
                    qd[dim] = v_prev + dv
                    qdd[dim] = traj.accelerations[segment_idx]
                    evaluated_segment = True
                else:
                    segment_idx += 1

        # If the trajectory is single-DOF, return the values as scalars.
        if len(q) == 1:
            q = q[0]
            qd = qd[0]
            qdd = qdd[0]
        return (q, qd, qdd)

    def generate(self, dt):
        """
        Generates a full trajectory at a specified sample time.

        Parameters
        ----------
            dt : float
                The sample time, in seconds.

        Returns
        -------
            tuple(array-like, array-like, array-like, array-like)
                The tuple of (t, q, qd, qdd) trajectory values generated at the sample time.
                The time vector is 1D, and the others are 2D (time and dimension).
        """

        # Initialize the data structure
        t_start = self.segment_times[0]
        t_final = self.segment_times[-1]
        t_vec = np.arange(t_start, t_final, dt)
        if t_vec[-1] != t_final:
            t_vec = np.append(t_vec, t_final)
        num_pts = len(t_vec)

        q = np.zeros([self.num_dims, num_pts])
        qd = np.zeros([self.num_dims, num_pts])
        qdd = np.zeros([self.num_dims, num_pts])

        for dim in range(self.num_dims):
            traj = self.single_dof_trajectories[dim]
            segment_idx = 0
            t_idx = 0
            while t_idx < num_pts:
                t_cur = t_vec[t_idx]
                if segment_idx < len(traj.times) - 1:
                    t_next = traj.times[segment_idx + 1]
                else:
                    t_next = t_final
                    segment_idx -= 1
                t_prev = traj.times[segment_idx]

                if t_cur <= t_next:
                    q_prev = traj.positions[segment_idx]
                    v_next = traj.velocities[segment_idx + 1]
                    v_prev = traj.velocities[segment_idx]

                    dt = t_cur - t_prev
                    dv = (v_next - v_prev) * dt / (t_next - t_prev)
                    q[dim, t_idx] = q_prev + (0.5 * dv + v_prev) * dt
                    qd[dim, t_idx] = v_prev + dv
                    qdd[dim, t_idx] = traj.accelerations[segment_idx]
                    t_idx += 1
                else:
                    segment_idx += 1

        return t_vec, q, qd, qdd

    def visualize(self, dt=0.01, joint_names=None):
        """
        Visualizes a generated trajectory with one figure window per dimension.

        Parameters
        ----------
            dt : float, optional
                The sample time at which to generate the trajectory.
                This is needed to produce the position curve.
            joint_names : list[str], optional
                The joint names to use for the figure titles.
                If unset, uses the text "Dimension 1", "Dimension 2", etc.
        """
        import matplotlib.pyplot as plt

        vertical_line_scale_factor = 1.2
        t_vec, q, _, _ = self.generate(dt)
        for dim, traj in enumerate(self.single_dof_trajectories):
            if joint_names is not None:
                title = joint_names[dim]
            else:
                title = f"Dimension {dim + 1}"
            plt.figure(title)

            # Positions, velocities, and accelerations
            plt.plot(t_vec, q[dim, :])
            plt.plot(traj.times, traj.velocities)
            plt.stairs(traj.accelerations, edges=traj.times)
            plt.legend(["Position", "Velocity", "Acceleration"])

            # Times
            min_val = vertical_line_scale_factor * min(
                np.min(traj.positions),
                np.min(traj.velocities),
                np.min(traj.accelerations),
            )
            max_val = vertical_line_scale_factor * max(
                np.max(traj.positions),
                np.max(traj.velocities),
                np.max(traj.accelerations),
            )
            plt.vlines(
                self.segment_times,
                min_val,
                max_val,
                colors="k",
                linestyles=":",
                linewidth=1.0,
            )

        plt.show()
