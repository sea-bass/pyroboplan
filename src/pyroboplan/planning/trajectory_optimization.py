""" Utilities for trajectory optimization based planning. """

import numpy as np
from pydrake.solvers import MathematicalProgram, Solve
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory


class CubicTrajectoryOptimizationOptions:
    """Options for cubic polynomial trajectory optimization."""

    def __init__(
        self,
        num_waypoints=6,
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=100.0,
        min_vel=-np.inf,
        max_vel=np.inf,
        min_accel=-np.inf,
        max_accel=np.inf,
        min_jerk=-np.inf,
        max_jerk=np.inf,
    ):
        """
        Initializes a set of options for cubic polynomial trajectory optimization.

        Parameters
        ----------
            num_waypoints : int
                The number of waypoints in the trajectory. Must be greater than or equal to 2.
            samples_per_segment : int
                The number of samples to take along each trajectory segment for setting kinematic constraints.
            min_segment_time : float
                The minimum duration of a trajectory segment, in seconds.
            max_segment_time : float
                The maximum duration of a trajectory segment, in seconds.
            min_vel : float, or array-like
                The minimum velocity along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_vel : float or array-like
                The maximum velocity along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            min_accel : float, or array-like
                The minimum acceleration along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_accel : float or array-like
                The maximum acceleration along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            min_jerk : float, or array-like
                The minimum jerk along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
            max_jerk : float or array-like
                The maximum jerk along the trajectory.
                If scalar, applies to all degrees of freedom; otherwise allows for different limits per degree of freedom.
        """
        if num_waypoints < 2:
            raise ValueError(
                "The number of waypoints must be greater than or equal to 2."
            )
        if min_segment_time <= 0:
            raise ValueError("The minimum segment time must be positive.")

        self.num_waypoints = num_waypoints
        self.samples_per_segment = samples_per_segment
        self.min_segment_time = min_segment_time
        self.max_segment_time = max_segment_time
        self.min_vel = min_vel
        self.max_vel = max_vel
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.min_jerk = min_jerk
        self.max_jerk = max_jerk


class CubicTrajectoryOptimization:
    """
    Trajectory optimization based planner.

    This uses the direct collocation approach to optimize over waypoint and collocation point placements that describe a multi-segment cubic polynomial trajectory.

    Some good resources include:
        * Matthew Kelly's tutorials: https://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/index.html
        * Russ Tedrake's manipulation course book: https://underactuated.mit.edu/trajopt.html
    """

    def __init__(self, model, options=CubicTrajectoryOptimizationOptions()):
        """
        Creates an instance of an RRT planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            options : `CubicTrajectoryOptimizationOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.options = options

    def _process_limits(self, limits, num_dofs, name):
        """
        Helper function to process kinematics limits options.

          * If the input limits are scalar, reshape them to the number of degrees of freedom.
          * Whether the input limits are scalar, a list, or a numpy array, always return a numpy array.
          * If the input limits are not scalar, but the wrong size, this will raise an Exception.

        Parameters
        ----------
            limits : None, float, or array-like
                The input limits in various formats.
            num_dofs : int
                The number of degrees of freedom.
            name : str
                The name of the input limits, to generate a descriptive error.

        Return
        ------
            numpy.ndarray
                The processed limits, for compatibility with trajectory optimization.
        """
        limits = np.array(limits)
        if len(limits.shape) == 0 or limits.shape[0] == 1:
            limits = limits * np.ones(num_dofs)
        elif limits.shape != (num_dofs,):
            raise ValueError(f"{name} vector must have shape ({num_dofs},)")
        return limits

    def _eval_position(self, x, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory position.

        Parameters
        ----------
            x : pydrake.autodiffutils.AutoDiffXd
                The waypoint position points.
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The position along the specified segment evaluated at the specified step.
        """
        return (
            x[k, n]
            + x_d[k, n] * (step * h[k])
            + 0.5
            * (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
            * (step * h[k]) ** 2
            / h[k]
            + (2.0 / 3.0)
            * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
            * (step * h[k]) ** 3
            / h[k] ** 2
        )

    def _eval_velocity(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory velocity.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The velocity along the specified segment evaluated at the specified step.
        """
        return (
            x_d[k, n]
            + (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n])
            * (step * h[k])
            / h[k]
            + 2.0
            * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n])
            * (step * h[k]) ** 2
            / h[k] ** 2
        )

    def _eval_acceleration(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory acceleration.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The acceleration along the specified segment evaluated at the specified step.
        """
        return (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k] + 4.0 * (
            x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]
        ) * (step * h[k]) / h[k] ** 2

    def _eval_jerk(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory jerk.

        Parameters
        ----------
            x_d : pydrake.autodiffutils.AutoDiffXd
                The waypoint velocity points.
            xc_d : pydrake.autodiffutils.AutoDiffXd
                The collocation velocity points.
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            k : int
                The trajectory segment index.
            n : int
                The degree of freedom index.
            step : float
                The normalized distance, between 0 and 1, along that segment.

        Return
        ------
            pydrake.autodiffutils.AutodiffXd
                The jerk along the specified segment evaluated at the specified step.
        """
        return 8.0 * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]) / h[k] ** 2

    def plan(self, q_path):
        """
        Plans a trajectory from a start to a goal configuration, or along an entire trajectory.

        If the input list has 2 elements, then this is assumed to be the start and goal configurations.
        The intermediate waypoints will be determined automatically.

        If the input list has more than 2 elements, then these are the actual waypoints that must be achieved.

        Parameters
        ----------
            q_path : list[array-like]
                A list of joint configurations describing the desired motion.

        Return
        ------
            Optional[pyroboplan.trajectory.polynomial.CubicPolynomialTrajectory]
                The resulting trajectory, or None if optimization failed
        """
        if len(q_path) == 0:
            print("Cannot optimize over an empty path.")
            return None
        num_waypoints = self.options.num_waypoints
        num_dofs = len(q_path[0])

        if len(q_path) == num_waypoints:
            fully_specified_path = True
        elif len(q_path) == 2:
            fully_specified_path = False
        else:
            raise ValueError("Path must either be length 2 or equal to num_waypoints.")

        # Preprocess the kinematic limits
        min_vel = self._process_limits(self.options.min_vel, num_dofs, "min_vel")
        max_vel = self._process_limits(self.options.max_vel, num_dofs, "max_vel")
        min_accel = self._process_limits(self.options.min_accel, num_dofs, "min_accel")
        max_accel = self._process_limits(self.options.max_accel, num_dofs, "max_accel")
        min_jerk = self._process_limits(self.options.min_jerk, num_dofs, "min_jerk")
        max_jerk = self._process_limits(self.options.max_jerk, num_dofs, "max_jerk")

        # Initialize the basic program and its variables
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(num_waypoints, num_dofs)
        x_d = prog.NewContinuousVariables(num_waypoints, num_dofs)
        xc = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        xc_d = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        h = prog.NewContinuousVariables(num_waypoints - 1)

        # Initial and final conditions
        if fully_specified_path:
            for idx in range(num_waypoints):
                prog.AddBoundingBoxConstraint(q_path[idx], q_path[idx], x[idx, :])
        else:
            prog.AddBoundingBoxConstraint(q_path[0], q_path[0], x[0, :])
            prog.AddBoundingBoxConstraint(q_path[-1], q_path[-1], x[-1, :])
        # Initial and final velocities should always be zero.
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[0, :])
        prog.AddBoundingBoxConstraint(0.0, 0.0, x_d[num_waypoints - 1, :])

        for n in range(num_dofs):
            # Collocation point constraints
            for k in range(num_waypoints - 1):
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

                # Sample points along each segment to evaluate the constraints.
                for step in np.linspace(0.0, 1.0, self.options.samples_per_segment):
                    # Position limits
                    pos = self._eval_position(x, x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(pos <= self.model.upperPositionLimit[n])
                    prog.AddConstraint(pos >= self.model.lowerPositionLimit[n])

                    # Velocity limits
                    vel = self._eval_velocity(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(vel >= min_vel[n])
                    prog.AddConstraint(vel <= max_vel[n])

                    # Acceleration limits
                    accel = self._eval_acceleration(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(accel >= min_accel[n])
                    prog.AddConstraint(accel <= max_accel[n])

                    # Jerk limits
                    jerk = self._eval_jerk(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(jerk >= min_jerk[n])
                    prog.AddConstraint(jerk <= max_jerk[n])

            # Acceleration continuity between segments.
            for k in range(num_waypoints - 2):
                prog.AddConstraint(
                    self._eval_acceleration(x_d, xc_d, h, k, n, 1.0)
                    == self._eval_acceleration(x_d, xc_d, h, k + 1, n, 0.0)
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

        # Cost and bounds on trajectory segment times
        prog.AddBoundingBoxConstraint(
            self.options.min_segment_time, self.options.max_segment_time, h
        )
        prog.AddQuadraticCost(
            Q=np.eye(num_waypoints - 1),
            b=np.zeros(num_waypoints - 1),
            c=0.0,
            vars=h,
        )

        # Set initial conditions to help search
        if fully_specified_path:
            # Set initial guess assuming collocation points are exactly between the waypoints.
            prog.SetInitialGuess(x, np.array(q_path))
            init_collocation_points = []
            for k in range(num_waypoints - 1):
                init_collocation_points.append(0.5 * (q_path[k] + q_path[k + 1]))
            prog.SetInitialGuess(xc, np.array(init_collocation_points))
        else:
            # Set initial guess assuming linear trajectory from start to end.
            init_points = np.linspace(q_path[0], q_path[-1], 2 * num_waypoints - 1)
            prog.SetInitialGuess(x, init_points[::2])
            prog.SetInitialGuess(xc, init_points[1::2])

        h_init = 0.5 * (self.options.max_segment_time - self.options.min_segment_time)
        prog.SetInitialGuess(h, h_init * np.ones(num_waypoints - 1))
        prog.SetInitialGuess(x_d, np.zeros((num_waypoints, num_dofs)))
        prog.SetInitialGuess(xc_d, np.zeros((num_waypoints - 1, num_dofs)))

        # Solve the program
        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed.")
            return None

        # Unpack the values
        h_opt = result.GetSolution(h)
        x_opt = result.GetSolution(x)
        x_d_opt = result.GetSolution(x_d)

        # Generate the cubic trajectory and return it.
        t_opt = [0] + list(np.cumsum(h_opt))
        self.latest_trajectory = CubicPolynomialTrajectory(
            np.array(t_opt),
            np.array(x_opt.T),
            np.array(x_d_opt.T),
        )
        return self.latest_trajectory
