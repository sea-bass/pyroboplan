from pyroboplan.core.utils import (
    check_collisions_at_state,
    check_valid_pose,
    check_within_limits,
)


class NamedJointConfigurationsOptions:
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
    def __init__(
        self, model, collision_model, options=NamedJointConfigurationsOptions()
    ):
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
        return str([k for k in self.configuration_dict])

    def visualize_state(self, visualizer, name):
        visualizer.display(self[name])
