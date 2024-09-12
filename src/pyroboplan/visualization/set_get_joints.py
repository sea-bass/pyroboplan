from pyroboplan.core.utils import (
    check_collisions_at_state,
    check_valid_pose,
    check_within_limits,
)


class NamedJointConfigurationsOptions:
    """Options for named joint configurations"""

    def __init__(self, allow_collisions=False):
        """
        Initializes a set of named joint configuration options.

        Parameters
        ----------
            allow_collisions : bool
                Whether to allow in-collision states to be stored.
        """
        self.allow_collisions = allow_collisions


class NamedJointConfigurations:
    """Named joint configurations helper

    This is a simple helper to get, set, and visualize joint states for a given model.
    Each model should have its own NamedJointConfigurations, and validation is performed to
    ensure joint states are dimensionally valid and, optionally, within joint limits.

    """

    def __init__(
        self, model, collision_model, options=NamedJointConfigurationsOptions()
    ):
        """
        Creates an instance of a named joint configurations object.

        Parameters
        ----------
            model : `pinocchio.Model`
                The model to use for this named joint configurations.
            collision_model : `pinocchio.Model`
                The model to use for collision checking.
            options : `NamedJointConfigurationsOptions`, optional
                The options to use for this named joint configurations object.
                  If not specified, default options are used.
        """
        self.model = model
        self.collision_model = collision_model
        self.data = self.model.createData()
        self.collision_data = self.collision_model.createData()
        self.options = options
        self.configuration_dict = {}

    def __setitem__(self, key, state):
        """
        Set the configuration at key to state
        """
        # first, check if the state is valid
        if not check_valid_pose(self.model, state):
            if not check_within_limits(self.model, state):
                raise ValueError("Tried to add state outside of model's joint limits")
            raise ValueError(
                f"Tried to add state of dimension {len(state)} to model with number of joints {self.model.nq}"
            )

        # then, check collisions if necessary
        if not self.options.allow_collisions:
            if check_collisions_at_state(self.model, self.collision_model, state):
                raise ValueError("Tried to add state in collision")

        # finally, add to the dict; if the key is of the wrong type then the error message from this should be sufficiently explanatory
        self.configuration_dict[key] = state

    def __getitem__(self, key):
        """
        Get the configuration stored at key
        """
        # just return the dict at the key. Errors from accessing the dict incorrectly (invalid key type, no entry) should be sufficiently explanatory
        return self.configuration_dict[key]

    def __contains__(self, key):
        """
        See whether this object contains a configuration at key
        """
        return key in self.configuration_dict

    def __str__(self):
        """
        Print the states in this object
        """
        return str([k for k in self.configuration_dict])

    def visualize_state(self, visualizer, name):
        """
        Visualize the joint state stored in key `name`
        """
        visualizer.display(self[name])
