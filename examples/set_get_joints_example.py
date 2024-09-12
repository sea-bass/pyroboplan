"""
This example shows PyRoboPlan capabilities for getting and setting joint states easily with named configurations.
"""

from pinocchio.visualize import MeshcatVisualizer

from pyroboplan.core.utils import (
    get_random_collision_free_state,
)
from pyroboplan.models.two_dof import load_models, add_object_collisions
from pyroboplan.visualization.set_get_joints import (
    NamedJointConfigurationsOptions,
    NamedJointConfigurations,
)

if __name__ == "__main__":
    # Show an example saved joints states with the 2DoF robot
    model, collision_model, visual_model = load_models()
    add_object_collisions(model, collision_model, visual_model)
    ee_name = "ee"
    data = model.createData()
    collision_data = collision_model.createData()

    # Define the initial configuration
    q = get_random_collision_free_state(model, collision_model, distance_padding=0.1)

    # Initialize
    named_joints_configuration = NamedJointConfigurations(model, collision_model)
    named_joints_configuration["home"] = q

    # Initialize visualizer
    visualizer = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    visualizer.initViewer(open=True)
    visualizer.loadViewerModel()
    visualizer.display(q)

    while True:
        # get user input
        user_input = input(
            "Press 'Enter' to show another random state, enter a new name to save the current state as a named joint configuration, enter a used name to visualize that saved state, type 'h' or 'help' to see the existing saved joint configurations, or ctrl-c to quit.\n"
        )
        print()

        if user_input:
            # if the input isn't empty
            user_input = user_input.lower()
            if user_input == "h" or user_input == "help":
                print("Stored states:")
                print(named_joints_configuration)
            if user_input in named_joints_configuration:
                named_joints_configuration.visualize_state(visualizer, user_input)
            else:
                named_joints_configuration[user_input] = q
        else:
            # if the input is empty, make a new state
            q = get_random_collision_free_state(
                model, collision_model, distance_padding=0.1
            )
            visualizer.display(q)
