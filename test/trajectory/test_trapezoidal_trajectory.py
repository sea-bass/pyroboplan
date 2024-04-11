import numpy as np

from pyroboplan.trajectory.trapezoidal_velocity import TrapezoidalVelocityTrajectory


q = np.array(
    [
        [1.0, 2.0, 2.0, 2.4, 5.0],
        [1.0, 0.0, -1.0, 0.5, 2.0],
        [0.0, 1.0, 0.0, -1.0, 0.0],
    ]
)
qd_max = np.array(
    [
        1.5,
        0.5,
        0.7,
    ]
)
qdd_max = np.array(
    [
        1.0,
        1.5,
        0.9,
    ]
)

t = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)
print(t.single_dof_trajectories[0].times)
print(t.single_dof_trajectories[0].positions)
# print(t.velocities)
# print(t.accelerations)
t.visualize(dt=0.01)
