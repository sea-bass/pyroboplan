"""
This example shows PyRoboPlan capabilities for optimizing an existing
motion plan using trajectory optimization.
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
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
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

    distance_padding = 0.005
    max_rrt_tries = 3

    def random_valid_state():
        return get_random_collision_free_state(
            model, collision_model, distance_padding=distance_padding
        )

    while True:
        q_start = random_valid_state()
        q_goal = random_valid_state()
        viz.viewer["optimized_trajectory"].delete()
        viz.display(q_start)
        time.sleep(1.0)

        # Search for a path
        options = RRTPlannerOptions(
            max_step_size=0.05,
            max_connection_dist=0.25,
            rrt_connect=False,
            bidirectional_rrt=True,
            rrt_star=True,
            max_rewire_dist=2.0,
            max_planning_time=10.0,
            fast_return=True,
            goal_biasing_probability=0.15,
            collision_distance_padding=distance_padding,
        )
        print("")
        for idx in range(max_rrt_tries):
            print(f"Planning a path, try {idx+1}/{max_rrt_tries}...")
            planner = RRTPlanner(model, collision_model, options=options)
            q_path = planner.plan(q_start, q_goal)
            if len(q_path) > 0:
                print(f"Got a path with {len(q_path)} waypoints")
                planner.visualize(viz, "panda_hand", show_tree=True)
                break

        if len(q_path) == 0:
            print("Failed to plan.")
            continue

        # Perform trajectory optimization.
        dt = 0.025
        options = CubicTrajectoryOptimizationOptions(
            max_planning_time=15.0,
            num_waypoints=len(q_path),
            samples_per_segment=11,
            min_segment_time=0.1,
            max_segment_time=10.0,
            min_vel=-1.5,
            max_vel=1.5,
            min_accel=-0.75,
            max_accel=0.75,
            check_collisions=True,
            min_collision_dist=distance_padding,
            collision_influence_dist=0.025,
            collision_link_list=[
                "obstacle_box_1",
                "obstacle_box_2",
                "obstacle_sphere_1",
                "obstacle_sphere_2",
                "ground_plane",
            ],
        )
        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(model, collision_model, options)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

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
            visualize_frames(
                viz, "optimized_trajectory", tforms, line_length=0.075, line_width=2
            )
            time.sleep(1.0)

            # Animate the generated trajectory.
            input("Press 'Enter' to animate the path.")
            for idx in range(q_vec.shape[1]):
                viz.display(q_vec[:, idx])
                time.sleep(dt)
