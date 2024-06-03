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
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
from pyroboplan.visualization.meshcat_utils import visualize_frames


if __name__ == "__main__":
    # Create models and data.
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)
    data = model.createData()

    # Initialize visualizer.
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Configure trajectory optimization.
    dt = 0.025
    options = CubicTrajectoryOptimizationOptions(
        max_planning_time=15.0,
        num_waypoints=7,
        samples_per_segment=11,
        min_segment_time=0.1,
        max_segment_time=10.0,
        min_vel=-1.5,
        max_vel=1.5,
        min_accel=-0.75,
        max_accel=0.75,
        check_collisions=True,
        min_collision_dist=0.001,
        collision_influence_dist=0.01,
        collision_link_list=[
            "obstacle_box_1",
            "obstacle_box_2",
            "obstacle_sphere_1",
            "obstacle_sphere_2",
            "ground_plane",
        ],
    )

    def random_valid_state():
        return get_random_collision_free_state(
            model, collision_model, distance_padding=options.min_collision_dist
        )

    while True:
        print("")
        q_start = get_random_collision_free_state(model, collision_model)
        viz.display(q_start)
        viz.viewer["waypoints"].delete()
        time.sleep(1.0)

        # Perform trajectory optimization.
        multi_point = False
        if multi_point:
            # Multi point means we set all the waypoints and optimize how to move between them.
            q_path = [q_start] + [
                random_valid_state() for _ in range(options.num_waypoints - 1)
            ]
        else:
            # Single point means we set just the start and goal.
            # All other intermediate waypoints are optimized automatically.
            q_path = [q_start, random_valid_state()]

        planner = CubicTrajectoryOptimization(model, collision_model, options)

        max_retries = 5
        for idx in range(max_retries):
            print(f"Optimizing trajectory, try {idx+1}/{max_retries}...")
            if idx == 0:
                init_path = None
            else:
                print("Restarting with random path")
                init_path = (
                    [q_start]
                    + [random_valid_state() for _ in range(options.num_waypoints - 2)]
                    + [q_path[-1]]
                )
            traj = planner.plan(q_path, init_path=init_path)

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
                tforms = extract_cartesian_poses(model, "panda_hand", q_vec.T)
                visualize_frames(
                    viz, "waypoints", tforms, line_length=0.075, line_width=2
                )

                # Animate the generated trajectory.
                input("Press 'Enter' to animate the path.")
                for idx in range(q_vec.shape[1]):
                    viz.display(q_vec[:, idx])
                    time.sleep(dt)

                break
