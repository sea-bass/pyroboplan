"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm on a 2-DOF manipulator.
"""

from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import get_random_collision_free_state
from pyroboplan.models.two_dof import load_models, add_object_collisions
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path

if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_object_collisions(model, collision_model, visual_model)
    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the start and end configurations
    q_start = get_random_collision_free_state(
        model, collision_model, distance_padding=0.1
    )
    q_end = get_random_collision_free_state(
        model, collision_model, distance_padding=0.1
    )

    # Configure the RRT planner
    options = RRTPlannerOptions(
        max_step_size=0.05,
        max_connection_dist=0.2,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=False,
        max_rewire_dist=1.0,
        max_planning_time=10.0,
        rng_seed=None,
        fast_return=True,
        goal_biasing_probability=0.1,
        collision_distance_padding=0.0,
    )

    planner = RRTPlanner(model, collision_model, options=options)

    while True:
        viz.display(q_start)
        time.sleep(0.5)

        # Search for a path
        path = planner.plan(q_start, q_end)
        planner.visualize(viz, "ee", show_tree=True)

        # Animate the path
        if path:
            discretized_path = discretize_joint_space_path(path, options.max_step_size)

            input("Press 'Enter' to animate the path.")
            for q in discretized_path:
                viz.display(q)
                time.sleep(0.05)

            input("Press 'Enter' to plan another path, or ctrl-c to quit.")
            print()
            q_start = q_end
            q_end = get_random_collision_free_state(model, collision_model)
