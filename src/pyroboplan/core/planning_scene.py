class PlanningScene:

    def __init__(self, model, visual_model=None, collision_model=None):
        """
        Creates a planning scene instance given a Pinocchio model.
        """
        self.model = model
        self.visual_model = visual_model
        self.collision_model = collision_model
