"""
This example shows PyRoboPlan capabilities for path planning using
trajectory optimization.
"""

import matplotlib.pyplot as plt
from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
from pyroboplan.visualization.meshcat_utils import visualize_frames


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    q_start = get_random_collision_free_state(model, collision_model)
    viz.display(q_start)
    time.sleep(1.0)

    # Configure trajectory optimization
    dt = 0.025
    options = CubicTrajectoryOptimizationOptions(
        num_waypoints=5,
        samples_per_segment=11,
        min_segment_time=0.01,
        max_segment_time=10.0,
        min_vel=-1.5,
        max_vel=1.5,
        min_accel=-0.75,
        max_accel=0.75,
    )

    # Perform trajectory optimization
    multi_point = True
    if multi_point:
        # Multi point means we set all the waypoints and optimize how to move between them.
        q_path = [q_start] + [
            get_random_collision_free_state(model, collision_model)
            for _ in range(options.num_waypoints - 1)
        ]
    else:
        # Single point means we set just the start and goal.
        # All other intermediate waypoints are optimized automatically.
        q_path = [q_start, get_random_collision_free_state(model, collision_model)]

    planner = CubicTrajectoryOptimization(model, options)
    traj = planner.plan(q_path)

    if traj is not None:
        print("Trajectory optimization successful")
        traj_gen = traj.generate(dt)
        q_vec = traj_gen[1]

        # Display the trajectory and points along the path.
        plt.ion()
        traj.visualize(
            dt=dt,
            joint_names=model.names[1:],
            show_position=True,
            show_velocity=True,
            show_acceleration=True,
            show_jerk=True,
        )
        time.sleep(0.5)

        tforms = extract_cartesian_poses(model, "panda_hand", q_vec.T)
        viz.display(q_start)
        visualize_frames(viz, "waypoints", tforms, line_length=0.075, line_width=2)
        time.sleep(1.0)

        # Animate the generated trajectory
        input("Press 'Enter' to animate the path.")
        for idx in range(q_vec.shape[1]):
            viz.display(q_vec[:, idx])
            time.sleep(dt)
