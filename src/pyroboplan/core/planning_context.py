import pinocchio


class PlanningContext:
    """
    Defines a planning context, which contains Pinocchio models and necessary utilities for motion planning.
    """

    def __init__(self, model, visual_model=None, collision_model=None):
        """
        Creates a planning context instance given a set of Pinocchio models.
        """
        self.model = model
        self.visual_model = visual_model
        self.collision_model = collision_model

        self.data = model.createData()
        self.collision_data = collision_model.createData()
