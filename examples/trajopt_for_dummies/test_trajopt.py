import numpy as np
from pydrake.solvers import MathematicalProgram, Solve


N = 3
prog = MathematicalProgram()

x = prog.NewContinuousVariables(N)
x_d = prog.NewContinuousVariables(N)
xc = prog.NewContinuousVariables(N - 1)
xc_d = prog.NewContinuousVariables(N - 1)
h = prog.NewContinuousVariables(N - 1)

# Initial and final conditions
prog.AddConstraint(x[0] == 0.0)
prog.AddConstraint(x[N - 1] == 2.0)
prog.AddConstraint(x_d[0] == 0.0)
prog.AddConstraint(x_d[N - 1] == 0.0)

max_vel = 1.0
min_vel = -1.0
max_accel = 1.5
min_accel = -1.5
min_jerk = -1.25
max_jerk = 1.25

# Collocation point constraints
for k in range(N - 1):
    prog.AddConstraint(
        xc[k] == 0.5 * (x[k] + x[k + 1]) + (h[k] / 8.0) * (x_d[k] - x_d[k + 1])
    )
    prog.AddConstraint(
        xc_d[k] == -(1.5 / h[k]) * (x[k] - x[k + 1]) - 0.25 * (x_d[k] + x_d[k + 1])
    )

    # Sample and evaluate the trajectory to constrain
    for step in np.linspace(0, 1, 11):
        # Velocity limits
        prog.AddConstraint(
            x_d[k]
            + (-3.0 * x_d[k] + 4.0 * xc_d[k] - x_d[k + 1]) * (step * h[k]) / h[k]
            + 2.0
            * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1])
            * (step * h[k]) ** 2
            / h[k] ** 2
            <= max_vel
        )
        prog.AddConstraint(
            x_d[k]
            + (-3.0 * x_d[k] + 4.0 * xc_d[k] - x_d[k + 1]) * (step * h[k]) / h[k]
            + 2.0
            * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1])
            * (step * h[k]) ** 2
            / h[k] ** 2
            >= min_vel
        )
        # Acceleration limits
        prog.AddConstraint(
            (-3.0 * x_d[k] + 4.0 * xc_d[k] - x_d[k + 1]) / h[k]
            + 4.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) * (step * h[k]) / h[k] ** 2
            <= max_accel
        )
        prog.AddConstraint(
            (-3.0 * x_d[k] + 4.0 * xc_d[k] - x_d[k + 1]) / h[k]
            + 4.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) * (step * h[k]) / h[k] ** 2
            >= min_accel
        )
        # Jerk limits
        prog.AddConstraint(
            8.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) / h[k] ** 2 <= max_jerk
        )
        prog.AddConstraint(
            8.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) / h[k] ** 2 >= min_jerk
        )


# Acceleration and jerk continuity between segments.
for k in range(N - 2):
    prog.AddConstraint(
        (-3.0 * x_d[k] + 4.0 * xc_d[k] - x_d[k + 1]) / h[k]
        + 4.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) * (1.0 * h[k]) / h[k] ** 2
        == (-3.0 * x_d[k + 1] + 4.0 * xc_d[k + 1] - x_d[k + 2]) / h[k + 1]
        + 4.0
        * (x_d[k + 1] - 2.0 * xc_d[k + 1] + x_d[k + 2])
        * (0.0 * h[k + 1])
        / h[k + 1] ** 2
    )
    prog.AddConstraint(
        8.0 * (x_d[k] - 2.0 * xc_d[k] + x_d[k + 1]) / h[k] ** 2
        == 8.0 * (x_d[k + 1] - 2.0 * xc_d[k + 1] + x_d[k + 2]) / h[k] ** 2
    )

# Max value constraints
prog.AddBoundingBoxConstraint(min_vel, max_vel, x_d)
prog.AddBoundingBoxConstraint(min_vel, max_vel, xc_d)
prog.AddBoundingBoxConstraint(0.01, np.inf, h)

# Cost on trajectory segment times
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

t_waypt = [0] + list(np.cumsum(h_opt))
t_coll = [t_waypt[idx] + 0.5 * h_opt[idx] for idx in range(N - 1)]

# Generate trajectory
dt = 0.001
t_final = t_waypt[-1]
t_vec = np.arange(0.0, t_final, dt)
if t_vec[-1] != t_final:
    t_vec = np.append(t_vec, t_final)

x_vec = np.zeros_like(t_vec)
x_d_vec = np.zeros_like(t_vec)
x_dd_vec = np.zeros_like(t_vec)
x_ddd_vec = np.zeros_like(t_vec)
idx = 0
segment_idx = 0
while idx < len(x_vec):
    t = t_vec[idx]
    t_start = t_waypt[segment_idx]
    t_end = t_waypt[segment_idx + 1]
    if t <= t_end:
        tau = t - t_start
        h_cur = h_opt[segment_idx]
        x_vec[idx] = (
            x_opt[segment_idx]
            + x_d_opt[segment_idx] * tau
            + 0.5
            * (
                -3.0 * x_d_opt[segment_idx]
                + 4.0 * xc_d_opt[segment_idx]
                - x_d_opt[segment_idx + 1]
            )
            * tau**2
            / h_cur
            + (2.0 / 3.0)
            * (
                x_d_opt[segment_idx]
                - 2.0 * xc_d_opt[segment_idx]
                + x_d_opt[segment_idx + 1]
            )
            * tau**3
            / h_cur**2
        )
        x_d_vec[idx] = (
            x_d_opt[segment_idx]
            + (
                -3.0 * x_d_opt[segment_idx]
                + 4.0 * xc_d_opt[segment_idx]
                - x_d_opt[segment_idx + 1]
            )
            * tau
            / h_cur
            + 2.0
            * (
                x_d_opt[segment_idx]
                - 2.0 * xc_d_opt[segment_idx]
                + x_d_opt[segment_idx + 1]
            )
            * tau**2
            / h_cur**2
        )
        x_dd_vec[idx] = (
            -3.0 * x_d_opt[segment_idx]
            + 4.0 * xc_d_opt[segment_idx]
            - x_d_opt[segment_idx + 1]
        ) / h_cur + 4.0 * (
            x_d_opt[segment_idx]
            - 2.0 * xc_d_opt[segment_idx]
            + x_d_opt[segment_idx + 1]
        ) * tau / h_cur**2
        x_ddd_vec[idx] = (
            8.0
            * (
                x_d_opt[segment_idx]
                - 2.0 * xc_d_opt[segment_idx]
                + x_d_opt[segment_idx + 1]
            )
            / h_cur**2
        )
        idx += 1
    else:
        segment_idx += 1

# Plot the trajectory
import matplotlib.pyplot as plt

plt.figure()
plt.plot(t_vec, x_vec, "k-")
plt.plot(t_vec, x_d_vec, "r-")
plt.plot(t_vec, x_dd_vec, "b-")
plt.plot(t_vec, x_ddd_vec, "g-")
plt.plot(t_waypt, x_opt, "ko")
plt.plot(t_coll, xc_opt, "ko")
plt.plot(t_waypt, x_d_opt, "ro")
plt.plot(t_coll, xc_d_opt, "ro")
plt.legend(["Position", "Velocity", "Acceleration", "Jerk"])
plt.show()
