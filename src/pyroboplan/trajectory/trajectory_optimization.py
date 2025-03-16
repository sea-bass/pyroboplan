"""Utilities for trajectory optimization based planning."""

from functools import partial
import numpy as np
import warnings

import pinocchio
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient, ExtractValue
from pydrake.solvers import (
    MathematicalProgram,
    QuadraticConstraint,
    Solve,
    SolverOptions,
    SnoptSolver,
)

from pyroboplan.core.utils import (
    calculate_collision_vector_and_jacobians,
    get_collision_pair_indices_from_bodies,
)
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory


class CubicTrajectoryOptimizationOptions:
    """Options for cubic polynomial trajectory optimization."""

    def __init__(
        self,
        num_waypoints=3,
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=10.0,
        min_vel=-np.inf,
        max_vel=np.inf,
        min_accel=-np.inf,
        max_accel=np.inf,
        min_jerk=-np.inf,
        max_jerk=np.inf,
        max_planning_time=30.0,
        check_collisions=False,
        collision_link_list=[],
        min_collision_dist=0.0,
        collision_influence_dist=0.1,
        collision_avoidance_cost_weight=0.0,
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
            max_planning_time : float
                The maximum planning time, in seconds.
            check_collisions : bool
                If true, adds collision constraints to trajectory optimization.
            collision_link_list : list[str]
                A list of collision objects over which to check minimum distance constraints.
            min_collision_dist : float
                The minimum allowable collision distance, in meters.
            collision_influence_dist : float
                The distance for collision/distance checks, in meters, above which to ignore results.
            collision_avoidance_cost_weight : float
                Weight for a cost function that penalizes the minimum distance from collision.
                If this is set to zero, the cost function is not added to the mathematical program.
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

        self.max_planning_time = max_planning_time

        self.check_collisions = check_collisions
        self.collision_link_list = collision_link_list
        self.min_collision_dist = min_collision_dist
        self.collision_influence_dist = collision_influence_dist
        self.collision_avoidance_cost_weight = collision_avoidance_cost_weight


class CubicTrajectoryOptimization:
    """
    Trajectory optimization based planner.

    This uses the direct collocation approach to optimize over waypoint and collocation point placements that describe a multi-segment cubic polynomial trajectory.

    Some good resources include:
        * Matthew Kelly's tutorials: https://www.matthewpeterkelly.com/tutorials/trajectoryOptimization/index.html
        * Russ Tedrake's manipulation course book: https://underactuated.mit.edu/trajopt.html
    """

    def __init__(
        self, model, collision_model, options=CubicTrajectoryOptimizationOptions()
    ):
        """
        Creates an instance of a cubic trajectory optimization planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.GeometryModel`
                The model to use for collision checking.
            options : `CubicTrajectoryOptimizationOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
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

        This directly relates to equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

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

        This is the first derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

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
            pydrake.autodiffutils.AutoDiffXd
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

        This is the second derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

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
            pydrake.autodiffutils.AutoDiffXd
                The acceleration along the specified segment evaluated at the specified step.
        """
        return (-3.0 * x_d[k, n] + 4.0 * xc_d[k, n] - x_d[k + 1, n]) / h[k] + 4.0 * (
            x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]
        ) * (step * h[k]) / h[k] ** 2

    def _eval_jerk(self, x_d, xc_d, h, k, n, step):
        """
        Helper function to symbolically evaluate a trajectory jerk.

        This is the third derivative of equation 4.13 in https://epubs.siam.org/doi/10.1137/16M1062569.

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
            pydrake.autodiffutils.AutoDiffXd
                The jerk along the specified segment evaluated at the specified step.
        """
        return 8.0 * (x_d[k, n] - 2.0 * xc_d[k, n] + x_d[k + 1, n]) / h[k] ** 2

    def _total_time_cost(self, h, weight=1.0):
        """
        Helper function that sets a quadratic cost on the total trajectory time.

        Parameters
        ----------
            h : pydrake.autodiffutils.AutoDiffXd
                The time segment durations.
            weight : float, optional
                A weight to scale the cost function.

        Return
        ------
            pydrake.autodiffutils.AutoDiffXd
                Quadratic cost on the total trajectory time.
        """
        total_time = 0.0
        for step in h:
            total_time += step
        return weight * total_time**2

    def _collision_constraint(
        self,
        vars,
        num_waypoints,
        num_dofs,
        samples_per_segment,
        collision_pairs,
        min_allowable_dist,
        influence_dist,
    ):
        """
        Helper function to evaluate collision constraint and its gradients.

        These calculations are based off the following resources:
          * https://typeset.io/pdf/a-collision-free-mpc-for-whole-body-dynamic-locomotion-and-1l6itpfk.pdf
          * https://laas.hal.science/hal-04425002

        Parameters
        ----------
            vars : pydrake.autodiffutils.AutoDiffXd
                A flattened list of joint positions, joint velocities, collocation positions,
                and waypoint times. Needs to be in one variable to add this function as a constraint.
            num_waypoints : int
                The number of waypoints.
            num_dofs : int
                The number of degrees of freedom.
            samples_per_segment : int
                The number of samples per waypoint at which to evaluate collisions.
            collision_pairs : list[int]
                A list containing the indices of collision model pairs to check.
            min_allowable_dist : float
                The minimum allowable collision distance, in meters, for the constraint to be satisfied.
                Positive is out of collision and negative is in collision.
            influence_dist : float
                The distance over which the collision distance has an effect on the constraint gradient.
                This value should be larger than `min_allowable_dist`.

        Return
        ------
            pydrake.autodiffutils.AutoDiffXd
                The constraint value representing the minimum collision distance for every collision object and waypoint,
                as well as its corresponding gradient.
        """
        all_vars = ExtractValue(vars)
        all_gradients = ExtractGradient(vars)
        x_all = all_vars[: num_waypoints * num_dofs].reshape((num_waypoints, num_dofs))
        x_d_all = all_vars[
            num_waypoints * num_dofs : 2 * num_waypoints * num_dofs
        ].reshape((num_waypoints, num_dofs))
        xc_d_all = all_vars[
            2
            * num_waypoints
            * num_dofs : (2 * num_waypoints + num_waypoints - 1)
            * num_dofs
        ].reshape((num_waypoints - 1, num_dofs))
        h_all = all_vars[-(num_waypoints - 1) :]

        num_collision_pairs = len(collision_pairs)
        total_num_points = (num_waypoints - 1) * samples_per_segment

        min_distance_smoothed_squared = np.zeros(total_num_points * num_collision_pairs)

        gradient = np.zeros(
            (total_num_points * num_collision_pairs, all_gradients.shape[0])
        )
        point_idx = 0
        for k in range(num_waypoints - 1):
            for step in np.linspace(0.0, 1.0, samples_per_segment):
                q = np.array(
                    [
                        self._eval_position(x_all, x_d_all, xc_d_all, h_all, k, n, step)
                        for n in range(num_dofs)
                    ]
                )

                # Compute collision and distance checks
                pinocchio.framesForwardKinematics(self.model, self.data, q)
                pinocchio.computeDistances(
                    self.model, self.data, self.collision_model, self.collision_data, q
                )

                # Get the minimum distance in the allowable range
                for link_idx, pair in enumerate(collision_pairs):
                    dist = self.collision_data.distanceResults[pair].min_distance

                    # Find the collision Jacobian for the closest point pair, and calculate gradients.
                    if dist <= influence_dist:
                        distance_vec, Jcoll1, Jcoll2 = (
                            calculate_collision_vector_and_jacobians(
                                self.model,
                                self.collision_model,
                                self.data,
                                self.collision_data,
                                pair,
                                q,
                            )
                        )
                        grad = distance_vec.T @ (Jcoll2 - Jcoll1)
                    else:
                        grad = np.zeros(num_dofs)

                    # Minimum distances are smoothed using a squared hinge loss to have better gradients.
                    min_distance_smoothed = np.clip(
                        1.0
                        + (dist - min_allowable_dist)
                        / (min_allowable_dist - influence_dist),
                        0.0,
                        np.inf,
                    )
                    grad_idx = point_idx * num_collision_pairs + link_idx
                    min_distance_smoothed_squared[grad_idx] = min_distance_smoothed**2
                    grad = (
                        2.0
                        * grad
                        * min_distance_smoothed
                        / (min_allowable_dist - influence_dist)
                    )

                    # Assign the gradients for x, x_d, xc_d, and h at this index.
                    start = k * num_dofs
                    end = start + num_dofs
                    gradient[grad_idx, start:end] += grad

                    start = (num_waypoints * num_dofs) + k * num_dofs
                    end = start + num_dofs
                    gradient[grad_idx, start:end] += grad

                    start = (num_waypoints * 2 * num_dofs) + k * num_dofs
                    end = start + num_dofs
                    gradient[grad_idx, start:end] += grad

                    h_idx = ((num_waypoints * 2 + num_waypoints - 1) * num_dofs) + k
                    gradient[grad_idx, h_idx] += np.sum(grad)

                point_idx += 1

        return InitializeAutoDiff(
            min_distance_smoothed_squared, gradient @ all_gradients
        )

    def plan(self, q_path, init_path=None):
        """
        Plans a trajectory from a start to a goal configuration, or along an entire trajectory.

        If the input list has 2 elements, then this is assumed to be the start and goal configurations.
        The intermediate waypoints will be determined automatically.

        If the input list has more than 2 elements, then these are the actual waypoints that must be achieved.

        Parameters
        ----------
            q_path : list[array-like]
                A list of joint configurations describing the desired motion.
            init_path : list[array-like], optional
                If set, defines the initial guess for the path waypoints.

        Return
        ------
            Optional[pyroboplan.trajectory.polynomial.CubicPolynomialTrajectory]
                The resulting trajectory, or None if optimization failed
        """
        if len(q_path) == 0:
            warnings.warn("Cannot optimize over an empty path.")
            return None
        num_waypoints = self.options.num_waypoints
        num_dofs = len(q_path[0])

        if len(q_path) == num_waypoints:
            fully_specified_path = True
        elif len(q_path) == 2:
            fully_specified_path = False
        else:
            raise ValueError("Path must either be length 2 or equal to num_waypoints.")

        # Preprocess the kinematic limits.
        min_vel = self._process_limits(self.options.min_vel, num_dofs, "min_vel")
        max_vel = self._process_limits(self.options.max_vel, num_dofs, "max_vel")
        min_accel = self._process_limits(self.options.min_accel, num_dofs, "min_accel")
        max_accel = self._process_limits(self.options.max_accel, num_dofs, "max_accel")
        min_jerk = self._process_limits(self.options.min_jerk, num_dofs, "min_jerk")
        max_jerk = self._process_limits(self.options.max_jerk, num_dofs, "max_jerk")

        # Initialize the basic program and its variables.
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(num_waypoints, num_dofs)
        x_d = prog.NewContinuousVariables(num_waypoints, num_dofs)
        xc = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        xc_d = prog.NewContinuousVariables(num_waypoints - 1, num_dofs)
        h = prog.NewContinuousVariables(num_waypoints - 1)

        # Initial, final, and intermediate waypoint conditions.
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
            # Collocation point constraints.
            # Specifically, this constrains the position and velocities of the collocation points to be
            # expressed in terms of the waypoint positions and velocities, assuming cubic splines.
            # This is described in the "Direct Collocation" section here:
            # https://underactuated.mit.edu/trajopt.html
            for k in range(num_waypoints - 1):
                prog.AddQuadraticConstraint(
                    0.5 * (x[k, n] + x[k + 1, n])
                    + (h[k] / 8.0) * (x_d[k, n] - x_d[k + 1, n])
                    - xc[k, n],
                    0.0,
                    0.0,
                    QuadraticConstraint.HessianType.kIndefinite,
                )
                prog.AddConstraint(
                    xc_d[k, n]
                    == -(1.5 / h[k]) * (x[k, n] - x[k + 1, n])
                    - 0.25 * (x_d[k, n] + x_d[k + 1, n])
                )

                # Sample points along each segment to evaluate the position and velocity constraints, since
                # they are cubic and quadratic (respectively) and could overshoot the waypoints and collocation points.
                for step in np.linspace(0.0, 1.0, self.options.samples_per_segment):
                    pos = self._eval_position(x, x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(pos <= self.model.upperPositionLimit[n])
                    prog.AddConstraint(pos >= self.model.lowerPositionLimit[n])

                    vel = self._eval_velocity(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(vel >= min_vel[n])
                    prog.AddConstraint(vel <= max_vel[n])

                # The acceleration and jerk are linear and piecewise constant (respectively),
                # so these can be constrained simply by checking each endpoint.
                for step in [0.0, 1.0]:
                    accel = self._eval_acceleration(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(accel >= min_accel[n])
                    prog.AddConstraint(accel <= max_accel[n])

                    jerk = self._eval_jerk(x_d, xc_d, h, k, n, step)
                    prog.AddConstraint(jerk >= min_jerk[n])
                    prog.AddConstraint(jerk <= max_jerk[n])

            # Enforce acceleration continuity between segments.
            # That is, the final acceleration of the kth waypoint must equal the initial acceleration of the (k+1)th waypoint.
            for k in range(num_waypoints - 2):
                prog.AddConstraint(
                    self._eval_acceleration(x_d, xc_d, h, k, n, 1.0)
                    == self._eval_acceleration(x_d, xc_d, h, k + 1, n, 0.0)
                )

        # Cost and bounds on trajectory times.
        prog.AddBoundingBoxConstraint(
            self.options.min_segment_time, self.options.max_segment_time, h
        )
        prog.AddCost(self._total_time_cost, h)

        # Collision checking by sampling points along the entire trajectory.
        link_list = self.options.collision_link_list
        if self.options.check_collisions and len(link_list) > 0:
            all_pairs = get_collision_pair_indices_from_bodies(
                self.model, self.collision_model, link_list
            )
            num_pairs = len(all_pairs)

            total_points = (num_waypoints - 1) * self.options.samples_per_segment
            # The minimum distance values are smoothed in the constraint such that
            # a value of 1.0 or lower satisfies the constraint.
            min_dist_val = -np.inf * np.ones(total_points * num_pairs)
            max_dist_val = 1.0 * np.ones(total_points * num_pairs)
            collision_expr = partial(
                self._collision_constraint,
                num_waypoints=num_waypoints,
                num_dofs=num_dofs,
                samples_per_segment=self.options.samples_per_segment,
                collision_pairs=all_pairs,
                min_allowable_dist=self.options.min_collision_dist,
                influence_dist=self.options.collision_influence_dist,
            )
            vars = np.concat([x.flatten(), x_d.flatten(), xc_d.flatten(), h.flatten()])
            prog.AddConstraint(collision_expr, min_dist_val, max_dist_val, vars)

            # Optionally add a cost for collision avoidance.
            # Since the smoothed minimum distance is 1.0 at the constraint limit and
            # increases further into collision, we want to minimize this value.
            coll_cost_weight = self.options.collision_avoidance_cost_weight
            if coll_cost_weight > 0.0:
                prog.AddCost(
                    lambda var: -coll_cost_weight * max(collision_expr(var))[0],
                    vars,
                )

        # Set initial conditions to help search.
        if init_path or fully_specified_path:
            # Set initial guess assuming collocation points are exactly between the waypoints.
            if init_path:
                q_path = init_path
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

        h_init = 0.5 * (self.options.min_segment_time + self.options.max_segment_time)
        prog.SetInitialGuess(h, h_init * np.ones(num_waypoints - 1))
        prog.SetInitialGuess(x_d, np.zeros((num_waypoints, num_dofs)))
        prog.SetInitialGuess(xc_d, np.zeros((num_waypoints - 1, num_dofs)))

        # Solve the program.
        solver_options = SolverOptions()
        solver_options.SetOption(
            SnoptSolver.id(), "Time limit", self.options.max_planning_time
        )
        solver_options.SetOption(SnoptSolver.id(), "Timing level", 3)
        result = Solve(prog, solver_options=solver_options)
        if not result.is_success():
            print("Trajectory optimization failed.")
            return None

        # Unpack the values.
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
