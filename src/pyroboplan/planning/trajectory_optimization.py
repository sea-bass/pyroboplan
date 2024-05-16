""" Utilities for trajectory optimization based planning. """

import numpy as np
from pydrake.solvers import MathematicalProgram, Solve
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory


class CubicTrajectoryOptimizationOptions:
    """Options for cubic polynomial trajectory optimization."""

    def __init__(
        self,
    ):
        """
        Initializes a set of options
        """
        pass


class CubicTrajectoryOptimization:
    """Trajectory Optimization based planner"""

    def __init__(
        self, model, collision_model, options=CubicTrajectoryOptimizationOptions()
    ):
        """
        Creates an instance of an RRT planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `RRTPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model
        self.options = options
        self.latest_trajectory = None

    def plan(self, q_start, q_goal):
        """
        Plans a path from a start to a goal configuration.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_goal : array-like
                The goal robot configuration.
        """
        # Options TODO elevate
        self.num_waypoints = 3
        self.min_segment_time = 0.01
        self.max_segment_time = 100.0
        self.samples_per_segment = 11

        min_vel = -1.0 * np.ones_like(q_start)
        max_vel = 1.0 * np.ones_like(q_start)
        min_accel = -1.5 * np.ones_like(q_start)
        max_accel = 1.5 * np.ones_like(q_start)
        min_jerk = -2.0 * np.ones_like(q_start)
        max_jerk = 2.0 * np.ones_like(q_start)

        # Initialize the basic program and its variables
        self.num_dofs = len(q_start)
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.num_waypoints, self.num_dofs)
        x_d = prog.NewContinuousVariables(self.num_waypoints, self.num_dofs)
        xc = prog.NewContinuousVariables(self.num_waypoints - 1, self.num_dofs)
        xc_d = prog.NewContinuousVariables(self.num_waypoints - 1, self.num_dofs)
        h = prog.NewContinuousVariables(self.num_waypoints - 1)

        # Initial and final conditions
        prog.AddBoundingBoxConstraint(q_start, q_start, x[0, :])
        prog.AddBoundingBoxConstraint(q_goal, q_goal, x[self.num_waypoints - 1, :])
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[0, :])
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[self.num_waypoints - 1, :])

        data = self.model.createData()
        collision_data = self.collision_model.createData()

        import pinocchio
        from pydrake.autodiffutils import AutoDiffXd, ExtractGradient, ExtractValue
        from pydrake.math import ComputeNumericalGradient
        
        def check_collisions_helper(q):
            distances = []
            pinocchio.computeDistances(self.model, data, self.collision_model, collision_data, q)
            for p in range(len(self.collision_model.collisionPairs)):
                dist_result = collision_data.distanceResults[p]
                distances.append(dist_result.min_distance)
            return np.array([min(distances)])

        def check_collisions(q_val):
            if q_val.dtype != float: 
                q_val_num = ExtractValue(q_val)
                min_distance = check_collisions_helper(q_val_num)
                gradient = ComputeNumericalGradient(check_collisions_helper, q_val_num)
                return np.array([
                    AutoDiffXd(
                        min_distance[0],
                        gradient[0],
                    )
                ])

        for n in range(self.num_dofs):
            # Collocation point constraints
            for k in range(self.num_waypoints - 1):
                prog.AddConstraint(
                    xc[k, n]
                    == 0.5 * (x[k, n] + x[k + 1, n])
                    + (h[k] / 8.0) * (x_d[k, n] - x_d[k + 1, n])
                )
                prog.AddConstraint(
                    xc_d[k, n]
                    == -(1.5 / h[k]) * (x[k, n] - x[k + 1, n])
                    - 0.25 * (x_d[k, n] + x_d[k + 1, n])
                )

                # Sample and evaluate the trajectory to constrain
                for step in np.linspace(0, 1, self.samples_per_segment):
                    # Position limits
                    prog.AddConstraint(
                        x[k, n]
                        + x_d[k, n] * h[k]
                        + 0.5
                        * (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
                        * (step * h[k]) ** 2
                        / h[k]
                        + (2.0 / 3.0)
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k]) ** 3
                        / h[k] ** 2
                        <= self.model.upperPositionLimit[n]
                    )
                    prog.AddConstraint(
                        x[k, n]
                        + x_d[k, n] * h[k]
                        + 0.5
                        * (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
                        * (step * h[k]) ** 2
                        / h[k]
                        + (2.0 / 3.0)
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k]) ** 3
                        / h[k] ** 2
                        >= self.model.lowerPositionLimit[n]
                    )

                    # Velocity limits
                    prog.AddConstraint(
                        x_d[k, n]
                        + (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
                        * (step * h[k])
                        / h[k]
                        + 2.0
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k]) ** 2
                        / h[k] ** 2
                        <= max_vel[n]
                    )
                    prog.AddConstraint(
                        x_d[k, n]
                        + (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
                        * (step * h[k])
                        / h[k]
                        + 2.0
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k]) ** 2
                        / h[k] ** 2
                        >= min_vel[n]
                    )
                    # Acceleration limits
                    prog.AddConstraint(
                        (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k]
                        + 4.0
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k])
                        / h[k] ** 2
                        <= max_accel[n]
                    )
                    prog.AddConstraint(
                        (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k]
                        + 4.0
                        * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                        * (step * h[k])
                        / h[k] ** 2
                        >= min_accel[n]
                    )
                    # Jerk limits
                    prog.AddConstraint(
                        8.0 * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]) / h[k] ** 2
                        <= max_jerk[n]
                    )
                    prog.AddConstraint(
                        8.0 * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]) / h[k] ** 2
                        >= min_jerk[n]
                    )

            # Acceleration continuity between segments.
            for k in range(self.num_waypoints - 2):
                prog.AddConstraint(
                    (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k]
                    + 4.0
                    * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
                    * (1.0 * h[k])
                    / h[k] ** 2
                    == (-3.0 * x_d[k + 1, n] + 4.0 * xc_d[k + 1, n] - x_d[k + 2, n])
                    / h[k + 1]
                    + 4.0
                    * (x_d[k + 1, n] - 2.0 * xc_d[k + 1, n] + x_d[k + 2, n])
                    * (0.0 * h[k + 1])
                    / h[k + 1] ** 2
                )

            # Velocities and positions at waypoints
            prog.AddBoundingBoxConstraint(
                self.model.lowerPositionLimit[n],
                self.model.upperPositionLimit[n],
                x[:, n],
            )
            prog.AddBoundingBoxConstraint(
                self.model.lowerPositionLimit[n],
                self.model.upperPositionLimit[n],
                xc[:, n],
            )
            prog.AddBoundingBoxConstraint(min_vel[n], max_vel[n], x_d[:, n])
            prog.AddBoundingBoxConstraint(min_vel[n], max_vel[n], xc_d[:, n])

        for k in range(self.num_waypoints):
            prog.AddConstraint(check_collisions, [0.001], [np.inf], x[k, :])

        # Cost and bounds on trajectory segment times
        prog.AddBoundingBoxConstraint(self.min_segment_time, self.max_segment_time, h)
        prog.AddQuadraticCost(
            Q=np.eye(self.num_waypoints - 1),
            b=np.zeros(self.num_waypoints - 1),
            c=0.0,
            vars=h,
        )

        # Set initial condition assuming linear trajectory
        x_init = []
        position_breaks = []
        for k in range(self.num_waypoints):
            qval = q_start + (q_goal - q_start) * k / (self.num_waypoints + 1)
            position_breaks.append(qval)  # x
            x_init.extend(list(qval))
        for k in range(self.num_waypoints):
            x_init.extend(list(np.zeros_like(q_start)))  # x_d
        for k in range(self.num_waypoints - 1):
            x_init.extend(
                list(0.5 * (position_breaks[k] + position_breaks[k + 1]))
            )  # xc
        for k in range(self.num_waypoints - 1):
            x_init.extend(list(np.zeros_like(q_start)))  # xc_d
        x_init.extend(np.ones(self.num_waypoints - 1))  # h

        # Solve the program
        result = Solve(prog, x_init)
        if not result.is_success():
            print("Trajectory optimization failed.")
            return None

        # Unpack the values
        h_opt = result.GetSolution(h)
        x_opt = result.GetSolution(x)
        x_d_opt = result.GetSolution(x_d)
        xc_opt = result.GetSolution(xc)
        xc_d_opt = result.GetSolution(xc_d)
        print(f"h* = {h_opt}")
        print(f"x* = {x_opt}")
        print(f"xc* = {xc_opt}")
        print(f"x_d* = {x_d_opt}")
        print(f"xc_d* = {xc_d_opt}")

        t_vec = [0] + list(np.cumsum(h_opt))
        self.latest_trajectory = CubicPolynomialTrajectory(
            np.array(t_vec),
            np.array(x_opt.T),
            np.array(x_d_opt.T),
        )
        return self.latest_trajectory

    def visualize(self, frame_name, dt=0.001, joint_names=None):
        """TODO Visualize the trajectory"""
        if self.latest_trajectory is None:
            print("No trajectory available to visualize.")
            return

        self.latest_trajectory.visualize(dt=dt, joint_names=joint_names)
