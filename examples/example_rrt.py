"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm.
"""

from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import get_random_collision_free_state
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the start and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)
    viz.display(q_start)
    time.sleep(1.0)

    # Search for a path
    options = RRTPlannerOptions(
        max_angle_step=0.05,
        max_connection_dist=0.25,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=False,
        max_rewire_dist=3.0,
        max_planning_time=10.0,
        fast_return=True,
        goal_biasing_probability=0.15,
    )

    planner = RRTPlanner(model, collision_model, options=options)
    path = planner.plan(q_start, q_end)
    planner.visualize(viz, "panda_hand", show_tree=True)

    do_shortcutting = True
    if do_shortcutting:
        from pyroboplan.core.utils import extract_cartesian_poses
        from pyroboplan.planning.path_shortcutting import shortcut_path
        from pyroboplan.visualization.meshcat_utils import visualize_frames

        path = shortcut_path(model, collision_model, path)

        # TODO: Factor to reusable function
        q_path = []
        for idx in range(1, len(path)):
            q_start = path[idx - 1]
            q_goal = path[idx]
            q_path = q_path + discretize_joint_space_path(
                q_start, q_goal, options.max_angle_step
            )

        target_tforms = extract_cartesian_poses(model, "panda_hand", q_path)
        visualize_frames(
            viz, "shortened_path", target_tforms, line_length=0.05, line_width=1.5
        )

    # Animate the path
    if path:
        input("Press 'Enter' to animate the path.")
        for idx in range(1, len(path)):
            segment_start = path[idx - 1]
            segment_end = path[idx]
            q_path = discretize_joint_space_path(
                segment_start, segment_end, options.max_angle_step
            )
            for q in q_path:
                viz.display(q)
                time.sleep(0.05)
