"""
This example shows PyRoboPlan capabilities for trajectory generation giving
various time scaling approaches, such as polynomial and trapezoidal velocity.
"""

import matplotlib.pyplot as plt
import numpy as np
from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
)
from pyroboplan.trajectory.polynomial import QuinticPolynomialTrajectory
from pyroboplan.trajectory.trapezoidal_velocity import TrapezoidalVelocityTrajectory
from pyroboplan.visualization.meshcat_utils import visualize_frames

if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the start, middle, and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)
    q_mid = 0.5 * (q_start + q_end)

    # Generate a trajectory
    dt = 0.025
    q = np.vstack([q_start, q_mid, q_end]).T

    mode = "trapezoidal"
    if mode == "trapezoidal":
        qd_max = 0.75
        qdd_max = 0.5
        traj = TrapezoidalVelocityTrajectory(q, qd_max, qdd_max)
        t_vec, q_vec, qd_vec, qdd_vec = traj.generate(dt)
    elif mode == "quintic":
        t_vec = [0.0, 3.0, 6.0]
        qd = 0.0
        qdd = 0.0
        traj = QuinticPolynomialTrajectory(t_vec, q, qd, qdd)
        t_vec, q_vec, qd_vec, qdd_vec, qddd_vec = traj.generate(dt)

    # Display the trajectory and points along the path.
    # If using a polynomial trajectory, you can also add show_jerk=True.
    plt.ion()
    traj.visualize(
        dt=dt,
        joint_names=model.names[1:],
        show_position=True,
        show_velocity=True,
        show_acceleration=True,
    )
    time.sleep(0.5)

    tforms = extract_cartesian_poses(model, "panda_hand", [q_start, q_mid, q_end])
    viz.display(q_start)
    visualize_frames(viz, "waypoints", tforms)
    time.sleep(0.5)

    # Animate the generated trajectory
    input("Press 'Enter' to animate the path.")
    for idx in range(q_vec.shape[1]):
        viz.display(q_vec[:, idx])
        time.sleep(dt)
