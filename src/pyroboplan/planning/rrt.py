""" Utilities for manipulation-specific Rapidly-Expanding Random Trees (RRTs). """

import numpy as np

from ..core.utils import extract_cartesian_poses, get_random_state
from ..visualization.meshcat_utils import visualize_frames

from .graph import Node, Graph
from .utils import discretize_joint_space_path, check_collisions_along_path


class RRTPlannerOptions:
    max_angle_step = 0.05
    """ Maximum angle step, in radians, for collision checking. """

    max_connection_dist = 0.25
    """ Maximum angular distance, in radians, for connecting nodes. """


class RRTPlanner:
    def __init__(self, model, collision_model, options=RRTPlannerOptions()):
        """
        TODO
        """
        self.model = model
        self.collision_model = collision_model
        self.options = options

    def plan(self, q_start, q_goal):
        """
        TODO
        """
        self.graph = Graph()
        start_node = Node(q_start, parent=None)
        self.graph.add_node(start_node)

        goal_found = False
        latest_node = start_node

        # Check direct connection to goal
        path_to_goal = discretize_joint_space_path(
            start_node.q, q_goal, self.options.max_angle_step
        )
        if not check_collisions_along_path(
            self.model, self.collision_model, path_to_goal
        ):
            goal_node = Node(q_goal, parent=start_node)
            self.graph.add_node(latest_node)
            self.graph.add_edge(start_node)
            goal_found = True

        while not goal_found:
            # Now sample a new configuration and try connect
            q_sample = get_random_state(self.model)
            nearest_node = self.graph.get_nearest_node(q_sample)

            # Sample to max connection distance
            q_diff = q_sample - nearest_node.q
            dist_norm = np.linalg.norm(q_diff)
            if dist_norm > self.options.max_connection_dist:
                q_sample = (
                    nearest_node.q
                    + (self.options.max_connection_dist / dist_norm) * q_diff
                )

            path_to_node = discretize_joint_space_path(
                nearest_node.q, q_sample, self.options.max_angle_step
            )
            if not check_collisions_along_path(
                self.model, self.collision_model, path_to_node
            ):
                latest_node = Node(
                    q_sample, parent=nearest_node
                )  # TODO: Should be closer
                self.graph.add_node(latest_node)
                self.graph.add_edge(nearest_node, latest_node)

                # Check if that latest node connects directly to goal
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

        # Back out the path
        cur_node = goal_node
        self.latest_path = []
        path_extracted = False
        while not path_extracted:
            if cur_node is None:
                path_extracted = True
            else:
                self.latest_path.append(cur_node.q)
                cur_node = cur_node.parent
        self.latest_path.reverse()
        return self.latest_path

    def visualize(self, visualizer, show_path=True, show_tree=False):
        """
        Visualizes the RRT path.
        """
        target_frame = "panda_hand"
        path_name = "planned_path"
        tree_name = "rrt"

        if show_path:
            q_path = []
            for idx in range(1, len(self.latest_path)):
                q_start = self.latest_path[idx - 1]
                q_goal = self.latest_path[idx]
                q_path = q_path + discretize_joint_space_path(
                    q_start, q_goal, self.options.max_angle_step
                )

            target_tforms = extract_cartesian_poses(self.model, target_frame, q_path)
            visualize_frames(
                visualizer, path_name, target_tforms, line_length=0.05, line_width=1
            )

        if show_tree:
            for idx, edge in enumerate(self.graph.edges):
                q_path = discretize_joint_space_path(
                    edge.nodeA.q, edge.nodeB.q, self.options.max_angle_step
                )
                target_tforms = extract_cartesian_poses(
                    self.model, target_frame, q_path
                )
                visualize_frames(
                    visualizer,
                    f"{tree_name}/edge{idx}",
                    target_tforms,
                    line_length=0.02,
                    line_width=0.5,
                    line_color=[0.0, 0.0, 0.6],
                )
