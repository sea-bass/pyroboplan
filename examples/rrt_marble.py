"""
RRT path planning example with a point robot in a 2D planar labyrinth.
"""

from pinocchio.visualize import MeshcatVisualizer
import time

from pyroboplan.core.utils import (
    get_random_collision_free_state,
    extract_cartesian_poses,
)
from pyroboplan.models.marble import load_models, add_object_collisions
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.visualization.meshcat_utils import visualize_frames

if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_object_collisions(model, collision_model, visual_model)
    data = model.createData()
    collision_data = collision_model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    # optional pseudo orthographic view
    # viz.viewer["/Cameras/default/rotated/<object>"].set_property("zoom", 10)

    # Define the start and end configurations
    q_start = get_random_collision_free_state(
        model, collision_model, distance_padding=0.1
    )
    q_end = get_random_collision_free_state(
        model, collision_model, distance_padding=0.1
    )

    # Configure the RRT planner
    options = RRTPlannerOptions(
        max_step_size=0.02,
        max_connection_dist=2.0,
        rrt_connect=True,
        bidirectional_rrt=True,
        rrt_star=False,
        max_rewire_dist=1.0,
        max_planning_time=1.0,
        rng_seed=None,
        fast_return=True,
        goal_biasing_probability=0.1,
        collision_distance_padding=0.0,
    )

    planner = RRTPlanner(model, collision_model, options=options)

    while True:
        viz.display(q_start)

        path = planner.plan(q_start, q_end)
        planner.visualize(viz, "body", show_tree=True, show_path=False)
        if path:
            do_shortcutting = True
            if do_shortcutting:
                path = shortcut_path(
                    model,
                    collision_model,
                    path,
                    max_iters=200,
                    max_step_size=options.max_step_size,
                )
            path = discretize_joint_space_path(path, options.max_step_size)
            target_tforms = extract_cartesian_poses(model, "body", path)
            viz.viewer["shortened_path"].delete()
            visualize_frames(
                viz,
                "shortened_path",
                target_tforms,
                line_length=0.05,
                line_width=2.5,
            )

            for q in path:
                viz.display(q)
                time.sleep(0.01)

            input("Press 'Enter' to plan another path, or ctrl-c to quit.")
            q_start = q_end
        q_end = get_random_collision_free_state(model, collision_model)
