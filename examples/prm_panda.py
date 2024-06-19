"""
This example shows PyRoboPlan capabilities for path planning using a
Probabilistic Roadmap (PRM) algorithm, on a 7-DOF Panda robot.
"""

import os
import time

from pinocchio.visualize import MeshcatVisualizer

from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.prm import PRMPlanner, PRMPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames


def run_prm_search(q_start, q_end, planner, options, ee_name, max_retries=5):
    for i in range(max_retries):
        print(f"Searching for path attempt: {i+1}...")
        path = planner.plan(q_start, q_end)
        if path:
            print("Found path!")
            break
        else:
            print("Failed to find a path, growing the PRM...")
            planner.construct_roadmap()

    # Animate the path
    if path:
        planner.visualize(viz, ee_name, show_path=True, show_graph=True)

        # Optionally shortcut the path
        do_shortcutting = False
        if do_shortcutting:
            path = shortcut_path(model, collision_model, path)

        discretized_path = discretize_joint_space_path(path, options.max_step_size)

        if do_shortcutting:
            target_tforms = extract_cartesian_poses(model, ee_name, discretized_path)
            visualize_frames(
                viz, "shortened_path", target_tforms, line_length=0.05, line_width=1.5
            )

        input("Press 'Enter' to animate the path.")
        q_path = discretize_joint_space_path(path, options.max_step_size)
        for q in q_path:
            viz.display(q)
            time.sleep(0.05)


if __name__ == "__main__":
    # We are going to construct a roadmap using the Panda model as an example.
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)
    ee_name = "panda_hand"
    data = model.createData()
    collision_data = collision_model.createData()

    # The PRMPlanner can write/read saved maps from disk to avoid having to construct
    # the PRM from scratch, set this to load it if it exists, and to save it post-
    # initialization.
    prm_file = ""
    load_prm = False
    if os.path.isfile(prm_file):
        print(f"Loading pre-existing PRM from: {prm_file}")
        load_prm = True

    # Configure, initialize, and construct a PRM from scratch.
    # If using PRM*, recommend upping the initial connection radius to 2*Pi.
    options = PRMPlannerOptions(
        max_step_size=0.05,
        max_neighbor_radius=3.14,
        max_neighbor_connections=10,
        max_construction_nodes=2500,
        construction_timeout=20.0,
        rng_seed=None,
        prm_star=False,
        prm_file=prm_file if load_prm else None,
    )
    planner = PRMPlanner(model, collision_model, options=options)

    if not load_prm:
        print(
            f"Initializing the PRM, this will take up to {options.construction_timeout} seconds..."
        )
        planner.construct_roadmap()

        # We can save the graph to file for use in future PRMs.
        if prm_file:
            print(f"Saving generated PRM to {prm_file}")
            planner.graph.save_to_file(prm_file)

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Define the start and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)
    viz.display(q_start)

    # Uncomment to optionally visualize the resulting PRM.
    # This can look messy when the graph has many nodes.
    # print("Visualizing the PRM...")
    # planner.visualize(viz, ee_name, show_path=False, show_graph=True)

    while True:
        run_prm_search(q_start, q_end, planner, options, ee_name, max_retries=5)

        # Reusing planners for repeated queries in static environments is one of the
        # key advantages of PRMs.
        input("Press 'Enter' to plan another path, or ctrl-c to quit.")
        print()
        planner.reset()
        q_start = q_end
        q_end = get_random_collision_free_state(model, collision_model)
