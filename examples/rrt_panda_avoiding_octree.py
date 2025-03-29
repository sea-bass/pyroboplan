"""
This example shows PyRoboPlan capabilities for path planning using
Rapidly-Exploring Random Tree (RRT) algorithm on a 7-DOF Panda robot in a point cloud environment
"""

from pyroboplan.models.panda import (
    load_models,
    load_point_cloud,
    add_self_collisions,
    add_octree_collisions,
)

from pinocchio.visualize import MeshcatVisualizer
from rrt_panda import rrt_panda

if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    # Provide a path to your point cloud if None example point cloud will be used
    octree = load_point_cloud(pointcloud_path=None)

    add_octree_collisions(model, collision_model, visual_model, octree)

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    rrt_panda(viz, model, collision_model)
