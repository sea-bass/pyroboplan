""" Utilities for manipulation-specific Rapidly-Expanding Random Trees (RRTs). """

import numpy as np
import time

from ..core.utils import (
    check_collisions_at_state,
    check_collisions_along_path,
    configuration_distance,
    extract_cartesian_poses,
    get_random_state,
)
from ..visualization.meshcat_utils import visualize_frames, visualize_path

from .graph import Node, Graph
from .utils import discretize_joint_space_path


class RRTPlannerOptions:
    """Options for Rapidly-expanding Random Tree (RRT) planning."""

    max_angle_step = 0.05
    """ Maximum angle step, in radians, for collision checking along path segments. """

    max_connection_dist = 0.2
    """ Maximum angular distance, in radians, for connecting nodes. """

    max_planning_time = 10.0
    """ Maximum planning time, in seconds. """

    goal_biasing_probability = 0.0
    """ Probability of sampling the goal configuration itself, which can help planning converge. """


class RRTPlanner:
    """Rapidly-expanding Random Tree (RRT) planner.

    This is a sampling-based motion planner that finds collision-free paths from a start to a goal configuration.

    Some good resources:
      * https://msl.cs.illinois.edu/~lavalle/papers/Lav98c.pdf
    """

    def __init__(self, model, collision_model):
        """
        Creates an instance of an RRT planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
        """
        self.model = model
        self.collision_model = collision_model

        self.latest_path = None
        self.graph = Graph()

    def plan(self, q_start, q_goal, options=RRTPlannerOptions()):
        """
        Plans a path from a start to a goal configuration.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_start : array-like
                The goal robot configuration.
            options : `RRTPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        t_start = time.time()
        self.options = options
        self.graph = Graph()
        start_node = Node(q_start, parent=None)
        self.graph.add_node(start_node)

        goal_found = False
        latest_node = start_node

        # Check start and end pose collisions.
        if check_collisions_at_state(self.model, self.collision_model, q_start):
            print("Start configuration in collision.")
            return None
        if check_collisions_at_state(self.model, self.collision_model, q_goal):
            print("Goal configuration in collision.")
            return None

        # Check direct connection to goal.
        path_to_goal = discretize_joint_space_path(
            start_node.q, q_goal, self.options.max_angle_step
        )
        if not check_collisions_along_path(
            self.model, self.collision_model, path_to_goal
        ):
            goal_node = Node(q_goal, parent=start_node)
            self.graph.add_node(latest_node)
            self.graph.add_edge(start_node, goal_node)
            goal_found = True

        while not goal_found:
            # Check for timeouts
            if time.time() - t_start > options.max_planning_time:
                print(f"Planning timed out after {options.max_planning_time} seconds.")
                break

            # Sample a new configuration.
            if np.random.random() < self.options.goal_biasing_probability:
                q_sample = q_goal
            else:
                q_sample = get_random_state(self.model)
            nearest_node = self.graph.get_nearest_node(q_sample)

            # Clip the distance between nearest and sampled nodes to max connection distance.
            distance = configuration_distance(q_sample, nearest_node.q)
            if distance > self.options.max_connection_dist:
                scale = self.options.max_connection_dist / distance
                q_sample = nearest_node.q + scale * (q_sample - nearest_node.q)

            # Add the node only if it is collision free.
            if check_collisions_at_state(self.model, self.collision_model, q_sample):
                continue

            path_to_node = discretize_joint_space_path(
                nearest_node.q, q_sample, self.options.max_angle_step
            )
            if not check_collisions_along_path(
                self.model, self.collision_model, path_to_node
            ):
                latest_node = Node(q_sample, parent=nearest_node)
                self.graph.add_node(latest_node)
                self.graph.add_edge(nearest_node, latest_node)

                # Check if latest node connects directly to goal.
                path_to_goal = discretize_joint_space_path(
                    latest_node.q, q_goal, self.options.max_angle_step
                )
                if not check_collisions_along_path(
                    self.model, self.collision_model, path_to_goal
                ):
                    goal_node = Node(q_goal, parent=latest_node)
                    self.graph.add_node(goal_node)
                    self.graph.add_edge(latest_node, goal_node)
                    goal_found = True

        # Back out the path by traversing the parents from the goal.
        self.latest_path = []
        if goal_found:
            cur_node = goal_node
            path_extracted = False
            while not path_extracted:
                if cur_node is None:
                    path_extracted = True
                else:
                    self.latest_path.append(cur_node.q)
                    cur_node = cur_node.parent
            self.latest_path.reverse()

        return self.latest_path

    def visualize(
        self,
        visualizer,
        frame_name,
        path_name="planned_path",
        tree_name="rrt",
        show_path=True,
        show_tree=False,
    ):
        """
        Visualizes the RRT path.

        Parameters
        ----------
            visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`, optional
                The visualizer to use for this solver.
            frame_name : str
                The name of the frame to use when visualizing paths in Cartesian space.
            path_name : str, optional
                The name of the MeshCat component for the path.
            tree_name : str, optional
                The name of the MeshCat component for the tree.
            show_path : bool, optional
                If true, shows the final path from start to goal.
            show_tree : bool, optional
                If true, shows the entire sampled tree.
        """
        if show_path:
            q_path = []
            for idx in range(1, len(self.latest_path)):
                q_start = self.latest_path[idx - 1]
                q_goal = self.latest_path[idx]
                q_path = q_path + discretize_joint_space_path(
                    q_start, q_goal, self.options.max_angle_step
                )

            target_tforms = extract_cartesian_poses(self.model, frame_name, q_path)
            visualize_frames(
                visualizer, path_name, target_tforms, line_length=0.05, line_width=1.5
            )

        if show_tree:
            for idx, edge in enumerate(self.graph.edges):
                q_path = discretize_joint_space_path(
                    edge.nodeA.q, edge.nodeB.q, self.options.max_angle_step
                )
                path_tforms = extract_cartesian_poses(self.model, frame_name, q_path)
                visualize_path(
                    visualizer,
                    f"{tree_name}/edge{idx}",
                    path_tforms,
                    line_width=0.5,
                    line_color=[0.9, 0.0, 0.9],
                )
