"""
This example shows PyRoboPlan capabilities for optimizing an existing
motion plan using trajectory optimization.
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
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.trajectory.trajectory_optimization import (
    CubicTrajectoryOptimization,
    CubicTrajectoryOptimizationOptions,
)
from pyroboplan.visualization.meshcat_utils import visualize_frames


if __name__ == "__main__":
    # Create models and data.
    # NOTE: We are using sphere collisions since they behave better with optimization.
    model, collision_model, visual_model = load_models(use_sphere_collisions=True)
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)
    data = model.createData()

    # Color the collision spheres to see them more easily.
    for cobj in collision_model.geometryObjects:
        if "panda" in cobj.name:
            cobj.meshColor = np.array([0.7, 0.0, 0.5, 0.3])

    # Initialize visualizer.
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayCollisions(True)  # Enable to show collision spheres

    distance_padding = 0.001

    def random_valid_state():
        return get_random_collision_free_state(
            model, collision_model, distance_padding=0.01
        )

    while True:
        q_start = random_valid_state()
        q_goal = random_valid_state()
        viz.viewer["rrt_start"].delete()
        viz.viewer["rrt_goal"].delete()
        viz.viewer["planned_path"].delete()
        viz.viewer["optimized_trajectory"].delete()
        viz.display(q_start)
        time.sleep(0.5)

        # Search for a path
        options = RRTPlannerOptions(
            max_step_size=0.05,
            max_connection_dist=5.0,
            rrt_connect=False,
            bidirectional_rrt=True,
            rrt_star=True,
            max_rewire_dist=5.0,
            max_planning_time=10.0,
            fast_return=True,
            goal_biasing_probability=0.15,
            collision_distance_padding=0.01,
        )
        print("")
        print(f"Planning a path...")
        planner = RRTPlanner(model, collision_model, options=options)
        q_path = planner.plan(q_start, q_goal)
        if len(q_path) > 0:
            print(f"Got a path with {len(q_path)} waypoints")
            planner.visualize(viz, "panda_hand", show_tree=True)
        else:
            print("Failed to plan.")
            continue

        # Perform trajectory optimization.
        dt = 0.025
        options = CubicTrajectoryOptimizationOptions(
            num_waypoints=len(q_path),
            samples_per_segment=7,
            min_segment_time=0.5,
            max_segment_time=10.0,
            min_vel=-1.5,
            max_vel=1.5,
            min_accel=-0.75,
            max_accel=0.75,
            min_jerk=-1.0,
            max_jerk=1.0,
            max_planning_time=30.0,
            check_collisions=True,
            min_collision_dist=distance_padding,
            collision_influence_dist=0.05,
            collision_avoidance_cost_weight=0.0,
            collision_link_list=[
                "obstacle_box_1",
                "obstacle_box_2",
                "obstacle_sphere_1",
                "obstacle_sphere_2",
                "ground_plane",
                "panda_hand",
            ],
        )
        print("Optimizing the path...")
        optimizer = CubicTrajectoryOptimization(model, collision_model, options)
        traj = optimizer.plan([q_path[0], q_path[-1]], init_path=q_path)

        if traj is None:
            print("Retrying with all the RRT waypoints...")
            traj = optimizer.plan(q_path, init_path=q_path)

        if traj is not None:
            print("Trajectory optimization successful")
            traj_gen = traj.generate(dt)
            q_vec = traj_gen[1]

            show_trajectory_plots = False
            if show_trajectory_plots:
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

        else:
            input("Press 'Enter' to try another path.")
