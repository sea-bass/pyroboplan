"""
description
"""


from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_pointcloud_collisions,
)
from pyroboplan.planning.path_shortcutting import shortcut_path
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
from pyroboplan.planning.utils import discretize_joint_space_path
from pyroboplan.visualization.meshcat_utils import visualize_frames
from pyroboplan.core.utils import (
    extract_cartesian_poses,
    get_random_collision_free_state,
)
import sys
import time

import hppfcl as fcl
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_pointcloud_collisions(model, collision_model, visual_model)


    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    # time.sleep(5.0)

    # Define the start and end configurations
    q_start = get_random_collision_free_state(model, collision_model)
    q_end = get_random_collision_free_state(model, collision_model)

    while True:
        viz.display(q_start)
        time.sleep(0.5)
