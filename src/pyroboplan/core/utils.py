import numpy as np


def get_random_state(model):
    return np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)


def check_within_limits(model, q):
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )
