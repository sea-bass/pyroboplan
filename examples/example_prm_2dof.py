"""
This example shows PyRoboPlan capabilities for path planning using a
Probabilistic Roadmap (PRM) algorithm. We rely on a 2-DOF manipulator
to shrink the configuration space so that we can construct a covering
roadmap in reasonable time.
"""

import time

from pinocchio.visualize import MeshcatVisualizer

from pyroboplan.core.utils import get_random_collision_free_state
from pyroboplan.planning.prm import PRMPlanner, PRMPlannerOptions
from pyroboplan.planning.utils import (
    discretized_joint_space_generator,
    discretize_joint_space_path,
)


def load_2dof_model():

    from pyroboplan.models.two_dof import (
        load_models,
    )

    model, collision_model, visual_model = load_models()

    return model, collision_model, visual_model, "ee"


def load_panda_model():

    from pyroboplan.models.panda import (
        load_models,
        add_self_collisions,
        add_object_collisions,
    )

    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

    return model, collision_model, visual_model, "panda_hand"


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
            planner.visualize(viz, ee_name, show_path=False, show_graph=True)

    # Animate the path
    if path:
        planner.visualize(viz, ee_name, show_path=True, show_graph=False)
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


if __name__ == "__main__":
    # Default to the simple 2 DOF example model. If switching to the Panda
    # or another high DOF manipulator, we recommend increasing the max_neighbor_radius
    # and max_neighbor_connections to accelerate planning.
    model, collision_model, visual_model, ee_name = load_2dof_model()
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

    # Configure, initialize, and construct a PRM from scratch
    options = PRMPlannerOptions(
        max_angle_step=0.05,
        max_neighbor_radius=0.25,
        max_neighbor_connections=5,
        max_construction_nodes=2500,
        construction_timeout=10.0,
        prm_file=None,
    )
    planner = PRMPlanner(model, collision_model, options=options)

    # We're going to "prime" construction of the PRM by adding nodes
    # from the discretized joint state to start. This is not as effective for
    # high dof manipulators, and primarily serves and an example for
    # parameterizing the sampling strategy when constructing PRMs.
    print("Initializing the PRM, this will take a few seconds...")
    generator = discretized_joint_space_generator(
        model, step_size=0.2, generate_random=True
    )
    planner.construct_roadmap(generator)

    # Visualize the resulting PRM
    print("Visualizing the PRM...")
    planner.visualize(viz, ee_name, show_path=False, show_graph=True)

    # We can save the graph to file for use in future PRMs.
    # planner.graph.save_to_file("a.graph")

    while True:
        run_prm_search(q_start, q_end, planner, options, ee_name, max_retries=5)

        # Reusing planners for repeated queries in static environments is one of the
        # key advantages of PRMs.
        input("Press 'Enter' to plan another trajectory, or ctrl-c to quit.")
        print()
        planner.reset()
        q_start = q_end
        q_end = get_random_collision_free_state(model, collision_model)
