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
    add_self_collisions(collision_model)
    add_object_collisions(collision_model, visual_model)

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
    options = RRTPlannerOptions()
    options.max_angle_step = 0.05
    options.max_connection_dist = 0.25
    options.goal_biasing_probability = 0.15
    options.max_planning_time = 10.0
    options.rrt_connect = False
    options.bidirectional_rrt = False
    options.rrt_star = False
    options.max_rewire_dist = 3.0

    planner = RRTPlanner(model, collision_model)
    path = planner.plan(q_start, q_end, options=options)
    planner.visualize(viz, "panda_hand", show_tree=True)

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
