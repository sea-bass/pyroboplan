"""
description
"""

from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_octree_collisions,
)

from pinocchio.visualize import MeshcatVisualizer
from rrt_panda import rrt_panda

if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()

    add_self_collisions(model, collision_model)
    add_octree_collisions(model, collision_model, visual_model)
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    rrt_panda(viz, model, collision_model)
