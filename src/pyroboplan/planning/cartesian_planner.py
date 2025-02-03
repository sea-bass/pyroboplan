"""Utilities for Cartesian (or task space) motion planning."""

import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation, Slerp

from pyroboplan.trajectory.trapezoidal_velocity import TrapezoidalVelocityTrajectory


class CartesianPlannerOptions:
    """Options for Cartesian planning."""

    def __init__(
        self,
        use_trapezoidal_scaling=True,
        max_linear_velocity=1.0,
        max_linear_acceleration=1.0,
        max_angular_velocity=1.0,
        max_angular_acceleration=1.0,
    ):
        """
        Initializes a set of Cartesian planning options.

        Parameters
        ----------
            use_trapezoidal_scaling : bool
                If True, uses trapezoidal velocity time scaling.
                Otherwise, uses linear time scaling.
            max_linear_velocity : float
                The maximum linear velocity magnitude, in meters per second.
            max_linear_acceleration : float
                The maximum linear acceleration magnitude, in meters per second squared.
            max_angular_velocity : float
                The maximum angular velocity magnitude, in radians per second.
            max_angular_acceleration : float
                The maximum angular acceleration magnitude, in radians per second squared.
        """
        self.use_trapezoidal_scaling = use_trapezoidal_scaling

        if max_linear_velocity <= 0:
            raise ValueError("Max linear velocity must be positive.")
        self.max_linear_velocity = max_linear_velocity
        if max_linear_acceleration <= 0:
            raise ValueError("Max linear acceleration must be positive.")
        self.max_linear_acceleration = max_linear_acceleration
        if max_angular_velocity <= 0:
            raise ValueError("Max angular velocity must be positive.")
        self.max_angular_velocity = max_angular_velocity
        if max_angular_acceleration <= 0:
            raise ValueError("Max angular acceleration must be positive.")
        self.max_angular_acceleration = max_angular_acceleration


class CartesianPlanner:
    """
    Cartesian (or task space) motion planner.

    This tool will plan a Cartesian trajectory through a set of poses,
    and then use an inverse kinematics solver to incrementally solve for valid
    robot joint configurations to achieve this task-space trajectory.

    Some good resources:
      * Slides: https://www.diag.uniroma1.it/~deluca/rob1_en/14_TrajectoryPlanningCartesian.pdf
      * PickNik blog: https://picknik.ai/cartesian%20planners/moveit/motion%20planning/2021/01/07/guide-to-cartesian-planners-in-moveit.html
    """

    def __init__(
        self, model, target_frame, tforms, ik_solver, options=CartesianPlannerOptions()
    ):
        """
        Initializes a Cartesian motion planner instance.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for motion planning.
            target_frame : str
                The name of the target frame in the model defining the Cartesian motion.
            tforms : list[`pinocchio.SE3`]
                A list of pose waypoints, in SE3, for the target frame to move through.
            ik_solver : any
                Any valid inverse kinematics solver from this library.
            options : `pyroboplan.planning.cartesian_planner.CartesianPlannerOptions`, optional
                A set of Cartesian planning options.
                If not set, a default set of options will be used.
        """
        self.model = model
        self.target_frame = target_frame
        self.tforms = tforms
        self.ik_solver = ik_solver
        self.options = options

        # Generate the segments
        cur_time = 0.0
        self.segments = []
        for idx in range(1, len(tforms)):
            tform_start = tforms[idx - 1]
            tform_end = tforms[idx]

            # Figure out a time scaling.
            tform_diff = tform_end.actInv(tform_start)
            linear_distance = np.linalg.norm(tform_diff.translation)
            angular_distance = Rotation.from_matrix(tform_diff.rotation).magnitude()

            if options.use_trapezoidal_scaling:
                # Use trapezoidal scaling limited by both the linear and angular velocities and accelerations.
                if linear_distance > 0:
                    normalized_linear_accel = (
                        options.max_linear_acceleration / linear_distance
                    )
                    normalized_linear_vel = (
                        options.max_linear_velocity / linear_distance
                    )
                else:
                    normalized_linear_accel = normalized_linear_vel = np.inf

                if angular_distance > 0:
                    normalized_angular_accel = (
                        options.max_angular_acceleration / angular_distance
                    )
                    normalized_angular_vel = (
                        options.max_angular_velocity / angular_distance
                    )
                else:
                    normalized_angular_accel = normalized_angular_vel = np.inf

                max_normalized_accel = min(
                    normalized_linear_accel, normalized_angular_accel
                )
                max_normalized_vel = min(normalized_linear_vel, normalized_angular_vel)

                traj = TrapezoidalVelocityTrajectory(
                    np.array([0.0, 1.0]), max_normalized_vel, max_normalized_accel
                )
                final_time = cur_time + traj.segment_times[-1]
                self.segments.append((traj, final_time))
            else:
                # Use constant velocity at the specified max
                constant_vel_time = max(
                    linear_distance / options.max_linear_velocity,
                    angular_distance / options.max_angular_velocity,
                )
                final_time = cur_time + constant_vel_time
                self.segments.append((None, final_time))

            cur_time = final_time

    def generate(self, q_init, dt):
        """
        Generates a set of joint trajectories from the Cartesian plan at a specified sample time.

        Parameters
        ----------
            q_init : array-like
                The starting joint configuration for the robot.
                This is used as the initial condition for solving inverse kinematics.
            dt : float
                The sample time to use for trajectory generation, in seconds.

        Returns
        -------
            tuple(bool, array-like, array-like)
                The tuple of (success, t_vec, q_vec) corresponding to a success metric,
                the vector of times, and vector of joint positions, respectively.
        """
        self.generated_tforms = []

        t_start = 0
        t_final = self.segments[-1][1]
        t_vec = np.arange(t_start, t_final, dt)
        if t_vec[-1] != t_final:
            t_vec = np.append(t_vec, t_final)
        num_pts = len(t_vec)

        q = np.zeros([self.model.nq, num_pts])
        q[:, 0] = q_init

        q_cur = q_init
        idx = 0
        segment_idx = 0
        t_prev = 0
        t_final_cur = 0

        while idx < num_pts:
            t = t_vec[idx]

            traj_cur, t_final_cur = self.segments[segment_idx]
            tform_prev = self.tforms[segment_idx]
            tform_next = self.tforms[segment_idx + 1]
            slerp = Slerp(
                [0.0, 1.0],
                Rotation.concatenate(
                    [
                        Rotation.from_matrix(tform_prev.rotation),
                        Rotation.from_matrix(tform_next.rotation),
                    ]
                ),
            )

            if t <= t_final_cur:
                # Find the normalized interpolation time.
                dt = t - t_prev
                if self.options.use_trapezoidal_scaling:
                    dt = min(dt, traj_cur.segment_times[-1])  # Avoids numerical issues?
                    alpha, _, _ = traj_cur.evaluate(dt)
                else:
                    alpha = dt / (t_final_cur - t_prev)

                # Interpolate position and rotation separately.
                p = (
                    alpha * tform_next.translation
                    + (1.0 - alpha) * tform_prev.translation
                )
                R = slerp(alpha).as_matrix()
                tform = pinocchio.SE3(R, p)
                self.generated_tforms.append(tform)

                # Solve IK.
                q_sol = self.ik_solver.solve(self.target_frame, tform, init_state=q_cur)
                if q_sol is not None:
                    q[:, idx] = q_sol
                else:
                    print(f"Failed to solve on point {idx + 1} out of {num_pts}")
                    return False, t_vec, q

                idx += 1
            else:
                segment_idx += 1
                t_prev = t_final_cur

        return True, t_vec, q
