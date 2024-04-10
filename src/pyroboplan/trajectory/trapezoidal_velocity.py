import numpy as np


class TrapezoidalVelocityTrajectorySingleDof:
    """Describes a segment of a trapezoidal velocity profile trajectory."""

    def __init__(self):
        """Creates an instance of a trapezoidal velocity profile trajectory segment."""
        self.times = [0.0]
        self.positions = []
        self.velocities = [0.0]
        self.accelerations = []


class TrapezoidalVelocityTrajectory:
    """
    Describes a trapezoidal velocity profile trajectory.

    Some good resources:
      * MATLAB documentation: https://www.mathworks.com/help/robotics/ref/trapveltraj.html
    """

    def __init__(self, q, qd_max, qdd_max):
        """
        Generates a trapezoidal velocity profile trajectory.

        Parameters
        ----------
            q : array-like
                A N-by-M array of N waypoints and M dimensions.
            qd_max : array-like or float
                The maximum velocity for each dimension (size N-by-1), or scalar for all dimensions.
            qdd_max : array-like or float
                The maximum acceleration for each dimension (size N-by-1), or scalar for all dimensions.
        """

        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        self.num_dims = q.shape[0]

        if len(qd_max) == 1:
            qd_max = qd_max * np.ones(self.num_dims)
        if len(qdd_max) == 1:
            qdd_max = qdd_max * np.ones(self.num_dims)

        self.segment_times = [0.0]
        self.single_dof_trajectories = [
            TrapezoidalVelocityTrajectorySingleDof() for dim in range(self.num_dims)
        ]
        for dim, traj in enumerate(self.single_dof_trajectories):
            traj.positions.append(q[dim][0])

        # Iterate through all the waypoints
        delta_q = np.diff(q, axis=1)
        for idx in range(delta_q.shape[1]):
            # First calculate the time bottleneck of all the joints by computing
            # the time needed for "bang-bang" control at maximum acceleration.
            #
            #      -------- qd_peak
            #     /\
            #    /  \  <--- slope = qdd_max
            #   /    \
            #   <----> ---- 0.0
            #  t_segment

            delta_q_cur = delta_q[:, idx]

            # delta_q = 0.5 * t_segment * qd_peak
            # qdd_max = 2 * qd_peak / t_segment
            #
            # From 2:
            #  qd_peak = qdd_max * t_segment / 2
            # To 1:
            #  delta_q = 0.5 * t_segment * qdd_max * t_semgnet / 2
            #  delta_q = 1/4 * t_segment**2 * qdd_max
            #  t_segment = sqrt( 4 * delta)

            t_min = 2.0 * np.sqrt(np.abs(delta_q_cur) / qdd_max)

            # If the peak velocity exceeded the max allowable velocity, this
            # needs to be done with a 3-segment trapezoidal velocity segment.
            #
            #      ---------     --- qd_max
            #     /         \
            #    /           \   <--- slope = qdd_max
            #   /             \
            #      |<----->|     --- 0
            #     t_zero_accel
            #  |<------------->|
            #      t_segment

            q_peak = qdd_max * t_min / 2.0
            for dim in range(self.num_dims):
                if np.abs(q_peak[dim]) > qd_max[dim]:
                    # qdd_max = 2 * qd_max / t =>
                    #   t = 2 * qd_max / qdd_max
                    t_max_accel = 2.0 * qd_max[dim] / qdd_max[dim]
                    delta_q_max_accel = 0.5 * qd_max[dim] * t_max_accel
                    delta_q_zero_accel = np.abs(delta_q_cur[dim]) - delta_q_max_accel
                    t_zero_accel = delta_q_zero_accel / qd_max[dim]
                    t_min[dim] = t_zero_accel + t_max_accel

            t_segment = max(t_min)

            # Now using that time, fit trajectory segments to each of these by
            # always using maximum acceleration.
            t_prev = self.segment_times[-1]
            for dim in range(self.num_dims):
                q_prev = self.single_dof_trajectories[dim].positions[-1]

                # Compute the acceleration direction
                dir = np.sign(delta_q_cur[dim])
                accel = dir * qdd_max[dim]

                qd_peak_limit = 0.5 * t_segment * accel
                if np.abs(qd_peak_limit) <= qd_max[dim]:
                    # This is a 2-segment trajectory.
                    qd_peak = 2.0 * delta_q_cur[dim] / t_segment
                    times = [t_prev + t_segment / 2.0, t_prev + t_segment]
                    positions = [q_prev + delta_q_cur[dim] / 2.0, q_prev + delta_q_cur[dim]]
                    velocities = [qd_peak, 0.0]
                    accelerations = [accel, -accel]
                else:
                    # This is a 3-segment trajectory.

                    # t_segment = t_max_accel + t_zero_accel            ... 1
                    # delta_q = delta_q_max_accel + delta_q_zero_accel  ... 2
                    # delta_q_max_accel = 0.5 * t_max_accel * qd_peak   ... 3
                    # delta_q_zero_accel = t_zero_accel * qd_peak       ... 4
                    # qdd_max = 2 * qd_peak / t_max_accel               ... 5

                    # From 5:
                    #  qd_peak = qdd_max * t_max_accel / 2
                    #  t_max_accel = 2 * qd_peak / qdd_max              ... A

                    # To 3:
                    #   delta_q_max_accel = 0.5 * t_max_accel * qdd_max * t_max_accel / 2
                    #   delta_q_max_accel = 0.25 * t_max_accel^2 * qdd_max

                    # From 5:
                    #   qd_peak = qdd_max * (t_segment - t_zero_accel) / 2
                    #   t_segment - t_zero_accel = 2 * qd_peak / qdd_max
                    #   t_zero_accel = t_segment - 2 * qd_peak / qdd_max   ... (B)

                    # delta_q = 0.5 * t_max_accel * qd_peak  +  t_zero_accel * qd_peak
                    # delta_q =  0.5 * (2*qd_peak/qdd_max) * qd_peak
                    #          + (t_segment - 2 * qd_peak / qdd_max) * qd_peak

                    # delta_q = qd_peak**2 /qdd_max
                    #          + t_segment * qd_peak - 2*qd_peak**2/qdd_max

                    # delta_q = t_segment * qd_peak - qd_peak**2 / qdd_max

                    # (1/qdd_max) qd_peak ** 2 + (-t_segment) * qd_peak + delta_q = 0
                    qd_peak_solutions = np.roots(
                        [1.0 / qdd_max[dim], -t_segment, np.abs(delta_q_cur[dim])]
                    )
                    qd_peak = dir * min(qd_peak_solutions)  # Why is it the min?

                    t_max_accel = 2 * np.abs(qd_peak / accel)
                    t_zero_accel = t_segment - t_max_accel
                    delta_q_max_accel = 0.25 * t_max_accel ** 2 * accel
                    delta_q_zero_accel = delta_q_cur[dim] - delta_q_max_accel

                    times = [
                        t_prev + t_max_accel / 2.0,
                        t_prev + t_max_accel / 2.0 + t_zero_accel,
                        t_prev + t_max_accel + t_zero_accel,
                    ]
                    positions = [
                        q_prev + delta_q_max_accel / 2.0,
                        q_prev + delta_q_max_accel / 2.0 + delta_q_zero_accel,
                        q_prev + delta_q_cur[dim],
                    ]
                    velocities = [qd_peak, qd_peak, 0.0]
                    accelerations = [accel, 0.0, -accel]

                # Add all segments to this dimension's trajectory.
                self.single_dof_trajectories[dim].times.extend(times)
                self.single_dof_trajectories[dim].positions.extend(positions)
                self.single_dof_trajectories[dim].velocities.extend(velocities)
                self.single_dof_trajectories[dim].accelerations.extend(
                    accelerations
                )

            self.segment_times.append(t_prev + t_segment)


    def evaluate(self, t):
        """ Evaluates the trajectory at a specific time. """
        pass

    
    def generate(self, dt):
        """ Generates a full trajectory at a specified sample time. """
        
        # Initialize the data structure
        t_final = self.segment_times[-1]
        t_vec = np.arange(0.0, t_final, dt)
        if t_vec[-1] != t_final:
            t_vec = np.append(t_vec, t_final)
        num_pts = len(t_vec)
        
        q = np.zeros([self.num_dims, num_pts])
        for dim in range(self.num_dims):
            q_cur = np.zeros(num_pts)
            traj = self.single_dof_trajectories[dim]
            t_segments = np.diff(traj.times)
            segment_idx = 0
            t_idx = 0
            while t_idx < num_pts:
                if t_vec[t_idx] <= traj.times[segment_idx + 1]:
                    dt = t_vec[t_idx] - traj.times[segment_idx]
                    dv = (traj.velocities[segment_idx+1] - traj.velocities[segment_idx]) * dt / t_segments[segment_idx]
                    init_vel = traj.velocities[segment_idx]
                    q_cur[t_idx] = traj.positions[segment_idx] + (0.5 * dv + init_vel) * dt
                    t_idx += 1
                else:
                    segment_idx += 1

            q[dim, :] = q_cur
        return t_vec, q

    def visualize(self, dt):
        import matplotlib.pyplot as plt

        t_vec, q = self.generate(dt)
        for dim, traj in enumerate(self.single_dof_trajectories):

            plt.figure(f"Dimension {dim + 1}")

            # Positions (TODO)
            plt.plot(traj.times, traj.positions, "rx")
            
            plt.plot(t_vec, q[dim,:])

            # Velocities
            plt.plot(traj.times, traj.velocities)

            # Accelerations
            plt.stairs(traj.accelerations, edges=traj.times)

            # Times
            plt.vlines(
                self.segment_times,
                min(1.2 * np.min(traj.velocities), 1.2 * np.min(traj.accelerations)),
                max(1.2 * np.max(traj.velocities), 1.2 * np.max(traj.accelerations)),
                colors="k",
                linestyles=":",
                linewidth=1.0,
            )
        plt.show()
