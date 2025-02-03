"""Utilities for manipulation-specific Probabilistic Roadmaps (PRMs)."""

import numpy as np
import time

from pyroboplan.planning.graph_search import astar

from ..core.utils import (
    check_collisions_at_state,
    extract_cartesian_poses,
    get_random_state,
)
from ..visualization.meshcat_utils import visualize_frames, visualize_paths

from .graph import Node, Graph
from .utils import discretize_joint_space_path, has_collision_free_path


def random_model_state_generator(model):
    while True:
        yield get_random_state(model)


class PRMPlannerOptions:
    """Options for Probabilistic Roadmap (PRM) planning."""

    def __init__(
        self,
        max_step_size=0.05,
        max_neighbor_radius=0.5,
        max_neighbor_connections=15,
        max_construction_nodes=5000,
        construction_timeout=10.0,
        rng_seed=None,
        prm_star=False,
        prm_file=None,
    ):
        """
        Initializes a set of PRM planner options.

        Parameters
        ----------
            max_step_size : float
                Maximum joint configuration step size for collision checking along path segments.
            max_neighbor_radius : float
                The maximum allowable connectable distance between nodes.
            max_neighbor_connections : float
                The maximum number of neighbors to check when adding a node to the roadmap.
            max_construction_nodes : int
                The maximum number of samples to generate in the configuration space when growing the graph.
            construction_timeout : float
                Maximum time allotted to sample the configuration space per call.
            rng_seed : int, optional
                Sets the seed for random number generation. Use to generate deterministic results.
            prm_star : str
                If True, use the PRM* approach to dynamically select the radius and max number of neighbors
                during construction of the roadmap.
            prm_file : str, optional
                Full file path of a persisted PRM graph to use in the planner.
                If this is not specified, the PRM will be constructed from scratch.
        """
        self.max_step_size = max_step_size
        self.max_neighbor_radius = max_neighbor_radius
        self.max_neighbor_connections = max_neighbor_connections
        self.rng_seed = rng_seed
        self.construction_timeout = construction_timeout
        self.max_construction_nodes = max_construction_nodes
        self.prm_star = prm_star
        self.prm_file = prm_file


class PRMPlanner:
    """Probabilistic Roadmap (PRM) planner.

    This is a sampling-based motion planner that constructs a graph in the free configuration
    space and then searches for a path using standard graph search functions.

    Graphs can be persisted to disk for use in future applications.

    Some helpful resources:
        * The original publication:
          https://www.kavrakilab.org/publications/kavraki-svestka1996probabilistic-roadmaps-for.pdf
        * Modifications of PRM including PRM*:
          https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf
        * A nice guide to higher dimensional motion planning:
          https://motion.cs.illinois.edu/RoboticSystems/MotionPlanningHigherDimensions.html

    """

    def __init__(self, model, collision_model, options=PRMPlannerOptions()):
        """
        Creates an instance of a PRM planner.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this solver.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `PRMPlannerOptions`, optional
                The options to use for planning. If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.options = options
        self.latest_path = None

        if not self.options.prm_file:
            self.graph = Graph()
        else:
            self.graph = Graph.load_from_file(self.options.prm_file)

    def construct_roadmap(self, sample_generator=None):
        """
        Grows the graph by sampling nodes using the provided generator, then connecting
        them to the Graph. The caller can optionally override the default generator, if desired.

        Parameters
        ----------
            sample_generator : Generator[array-like, None, None]
                The sample function to use in construction of the roadmap.
                Defaults to randomly sampling the robot's configuration space.
        """
        # Default to randomly sampling the model if no sample function is provided.
        np.random.seed(self.options.rng_seed)
        if not sample_generator:
            sample_generator = random_model_state_generator(self.model)

        t_start = time.time()
        added_nodes = 0
        while added_nodes < self.options.max_construction_nodes:
            if time.time() - t_start > self.options.construction_timeout:
                print(
                    f"Roadmap construction timed out after {self.options.construction_timeout} seconds."
                )
                break

            # At each iteration we naively sample a valid random state and attempt to connect it to the roadmap.
            q_sample = next(sample_generator)
            if check_collisions_at_state(
                self.model,
                self.collision_model,
                q_sample,
                self.data,
                self.collision_data,
            ):
                continue

            radius = self.options.max_neighbor_radius
            max_neighbors = self.options.max_neighbor_connections

            # If using PRM* we dynamically scale the radius and max number of connections
            # each iteration. The scaling is a function of log(num_nodes). For more info refer to section
            # 3.3 of https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/optreadings/rrtstar.pdf.
            if self.options.prm_star:
                num_nodes = len(self.graph.nodes)
                dimension = len(q_sample)
                if num_nodes > 0:
                    radius = radius * (np.log(num_nodes) / num_nodes) ** (1 / dimension)
                    max_neighbors = int(max_neighbors * np.log(num_nodes))

            # It's a valid configuration so add it to the roadmap
            new_node = Node(q_sample)
            self.graph.add_node(new_node)
            self.connect_node(new_node, radius, max_neighbors)
            added_nodes += 1

    def connect_node(self, new_node, radius, k):
        """
        Identifies all neighbors and makes connections to the added node.

        Parameters
        ----------
            parent_node : `pyroboplan.planning.graph.Node`
                The node to add.
            radius : float
                Only consider connections within the provided radius.
            k : int
                Only consider up to a maximum of k neighbors to make connections.

        Returns
        -------
            bool :
                True if the node was connected to the graph. False otherwise.
        """
        # Add the node and find all neighbors.
        neighbors = self.graph.get_nearest_neighbors(new_node.q, radius)

        # Attempt to connect at most `max_neighbor_connections` neighbors.
        success = False
        for node, _ in neighbors[0:k]:
            # If the nodes are connectable then add an edge.
            if has_collision_free_path(
                node.q,
                new_node.q,
                self.options.max_step_size,
                self.model,
                self.collision_model,
            ):
                self.graph.add_edge(node, new_node)
                success |= True

        return success

    def reset(self):
        """
        Resets the PRM's transient data between queries.
        """
        self.latest_path = None
        for node in self.graph.nodes:
            node.parent = None
            node.cost = None

    def connect_planning_nodes(self, start_node, goal_node):
        """
        Ensures the start and goal configurations can be connected to the PRM.

        Parameters
        ----------
            start_node : `pyroboplan.planning.graph.Node`
                The start node to connect.
            goal_node : `pyroboplan.planning.graph.Node`
                The goal node to connect.

        Returns
        -------
            bool :
                True if the nodes were able to be connected, False otherwise.
        """
        success = True

        # If we cannot connect the start and goal nodes then there is no recourse.
        if not self.connect_node(
            start_node,
            self.options.max_neighbor_radius,
            self.options.max_neighbor_connections,
        ):
            print("Failed to connect the start configuration to the PRM.")
            success = False
        if not self.connect_node(
            goal_node,
            self.options.max_neighbor_radius,
            self.options.max_neighbor_connections,
        ):
            print("Failed to connect the goal configuration to the PRM.")
            success = False

        return success

    def plan(self, q_start, q_goal):
        """
        Plans a path from a start to a goal configuration using the constructed graph.

        Parameters
        ----------
            q_start : array-like
                The starting robot configuration.
            q_goal : array-like
                The goal robot configuration.

        Returns
        -------
            list[array-like] :
                A path from the start to the goal state, if one exists. Otherwise None.
        """

        # Check start and end pose collisions.
        if check_collisions_at_state(
            self.model, self.collision_model, q_start, self.data, self.collision_data
        ):
            print("Start configuration in collision.")
            return None
        if check_collisions_at_state(
            self.model, self.collision_model, q_goal, self.data, self.collision_data
        ):
            print("Goal configuration in collision.")
            return None

        # Ensure the start and goal nodes are in the graph.
        start_node = Node(q_start)
        self.graph.add_node(start_node)
        goal_node = Node(q_goal)
        self.graph.add_node(goal_node)

        # Ensure the start and goal nodes are connected before attempting to plan
        path = None
        if self.connect_planning_nodes(start_node, goal_node):

            # Use a graph search to determine if there is a path between the start and goal poses.
            node_path = astar(self.graph, start_node, goal_node)

            # Reconstruct the path if it exists
            path = [node.q for node in node_path] if node_path else None
            self.latest_path = path

        # Always remove the start and end nodes from the PRM.
        self.graph.remove_node(start_node)
        self.graph.remove_node(goal_node)

        return path

    def visualize(
        self,
        visualizer,
        frame_name,
        path_name="planned_path",
        graph_name="prm",
        show_path=True,
        show_graph=False,
    ):
        """
        Visualizes the PRM and the latest path found within it.

        Parameters
        ----------
            visualizer : `pinocchio.visualize.meshcat_visualizer.MeshcatVisualizer`, optional
                The visualizer to use for this solver.
            frame_name : str
                The name of the frame to use when visualizing paths in Cartesian space.
            path_name : str, optional
                The name of the MeshCat component for the path.
            graph_name : str, optional
                The name of the MeshCat component for the tree.
            show_path : bool, optional
                If true, shows the final path from start to goal.
            show_graph : bool, optional
                If true, shows the entire PRM graph.
        """
        if show_graph:
            path_tforms = []
            for edge in self.graph.edges:
                q_path = discretize_joint_space_path(
                    [edge.nodeA.q, edge.nodeB.q], self.options.max_step_size
                )
                path_tforms.append(
                    extract_cartesian_poses(self.model, frame_name, q_path)
                )
            visualize_paths(
                visualizer,
                f"{graph_name}/edges",
                path_tforms,
                line_width=0.5,
                line_color=[0.9, 0.0, 0.9],
            )

        visualizer.viewer[path_name].delete()
        if show_path and self.latest_path:
            q_path = discretize_joint_space_path(
                self.latest_path, self.options.max_step_size
            )

            target_tforms = extract_cartesian_poses(self.model, frame_name, q_path)
            visualize_frames(
                visualizer, path_name, target_tforms, line_length=0.05, line_width=2.0
            )
