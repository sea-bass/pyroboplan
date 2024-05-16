import numpy as np
from pydrake.solvers import MathematicalProgram, Solve
from pyroboplan.trajectory.polynomial import CubicPolynomialTrajectory


q_start = np.array([0.0, 0.0, 0.0])
q_goal = np.array([-1.0, 0.5, 2.0])

N = 10  # Number of segments
M = len(q_start)  # Number of degrees of freedom
prog = MathematicalProgram()

x = prog.NewContinuousVariables(N, M)
x_d = prog.NewContinuousVariables(N, M)
xc = prog.NewContinuousVariables(N - 1, M)
xc_d = prog.NewContinuousVariables(N - 1, M)
h = prog.NewContinuousVariables(N - 1)


max_vel = 1.0 * np.ones_like(q_start)
min_vel = -1.0 * np.ones_like(q_start)
max_accel = 1.5 * np.ones_like(q_start)
min_accel = -1.5 * np.ones_like(q_start)
min_jerk = -1.25 * np.ones_like(q_start)
max_jerk = 1.25 * np.ones_like(q_start)

for m in range(M):

    # Initial and final conditions
    prog.AddConstraint(x[0, m] == q_start[m])
    prog.AddConstraint(x[N - 1, m] == q_goal[m])
    prog.AddConstraint(x_d[0, m] == 0.0)
    prog.AddConstraint(x_d[N - 1, m] == 0.0)

    # Collocation point constraints
    for k in range(N - 1):
        prog.AddConstraint(
            xc[k, m]
            == 0.5 * (x[k, m] + x[k + 1, m])
            + (h[k] / 8.0) * (x_d[k, m] - x_d[k + 1, m])
        )
        prog.AddConstraint(
            xc_d[k, m]
            == -(1.5 / h[k]) * (x[k, m] - x[k + 1, m])
            - 0.25 * (x_d[k, m] + x_d[k + 1, m])
        )

        # Sample and evaluate the trajectory to constrain
        for step in np.linspace(0, 1, 11):
            # Velocity limits
            prog.AddConstraint(
                x_d[k, m]
                + (-3.0 * x_d[k, m] + 4.0 * xc_d[k, m] - x_d[k + 1, m])
                * (step * h[k])
                / h[k]
                + 2.0
                * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m])
                * (step * h[k]) ** 2
                / h[k] ** 2
                <= max_vel[m]
            )
            prog.AddConstraint(
                x_d[k, m]
                + (-3.0 * x_d[k, m] + 4.0 * xc_d[k, m] - x_d[k + 1, m])
                * (step * h[k])
                / h[k]
                + 2.0
                * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m])
                * (step * h[k]) ** 2
                / h[k] ** 2
                >= min_vel[m]
            )
            # Acceleration limits
            prog.AddConstraint(
                (-3.0 * x_d[k, m] + 4.0 * xc_d[k, m] - x_d[k + 1, m]) / h[k]
                + 4.0
                * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m])
                * (step * h[k])
                / h[k] ** 2
                <= max_accel[m]
            )
            prog.AddConstraint(
                (-3.0 * x_d[k, m] + 4.0 * xc_d[k, m] - x_d[k + 1, m]) / h[k]
                + 4.0
                * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m])
                * (step * h[k])
                / h[k] ** 2
                >= min_accel[m]
            )
            # Jerk limits
            prog.AddConstraint(
                8.0 * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m]) / h[k] ** 2
                <= max_jerk[m]
            )
            prog.AddConstraint(
                8.0 * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m]) / h[k] ** 2
                >= min_jerk[m]
            )

    # Acceleration continuity between segments.
    for k in range(N - 2):
        prog.AddConstraint(
            (-3.0 * x_d[k, m] + 4.0 * xc_d[k, m] - x_d[k + 1, m]) / h[k]
            + 4.0
            * (x_d[k, m] - 2.0 * xc_d[k, m] + x_d[k + 1, m])
            * (1.0 * h[k])
            / h[k] ** 2
            == (-3.0 * x_d[k + 1, m] + 4.0 * xc_d[k + 1, m] - x_d[k + 2, m]) / h[k + 1]
            + 4.0
            * (x_d[k + 1, m] - 2.0 * xc_d[k + 1, m] + x_d[k + 2, m])
            * (0.0 * h[k + 1])
            / h[k + 1] ** 2
        )

    # Max value constraints
    prog.AddBoundingBoxConstraint(min_vel[m], max_vel[m], x_d[:, m])
    prog.AddBoundingBoxConstraint(min_vel[m], max_vel[m], xc_d[:, m])

# Cost on trajectory segment times
prog.AddBoundingBoxConstraint(0.01, 100.0, h)
prog.AddQuadraticCost(Q=np.eye(N - 1), b=np.zeros(N - 1), c=0.0, vars=h)

result = Solve(prog)

h_opt = result.GetSolution(h)
x_opt = result.GetSolution(x)
x_d_opt = result.GetSolution(x_d)
xc_opt = result.GetSolution(xc)
xc_d_opt = result.GetSolution(xc_d)

print("Success? ", result.is_success())
print("x* = ", x_opt)
print("x_d* = ", x_d_opt)
print("xc* = ", xc_opt)
print("xc_d* = ", xc_d_opt)
print("h* = ", h_opt)
print("optimal cost = ", result.get_optimal_cost())
print("solver is: ", result.get_solver_id().name())

# Generate trajectory
dt = 0.001
t_waypt = [0] + list(np.cumsum(h_opt))
traj = CubicPolynomialTrajectory(
    np.array(t_waypt),
    np.array(x_opt.T),
    np.array(x_d_opt.T),
)
t_vec, q_vec, qd_vec, qdd_vec = traj.generate(dt)
traj.visualize(dt=dt)
