import numpy as np


def zero_nullspace_component(model):
    return np.zeros_like(model.lowerPositionLimit)


def joint_limit_nullspace_component(model, q, gain=1.0, padding=0.0):
    upper_limits = model.upperPositionLimit - padding
    lower_limits = model.lowerPositionLimit + padding

    grad = zero_nullspace_component(model)
    for idx in range(len(grad)):
        if q[idx] > upper_limits[idx]:
            grad[idx] = -gain * (q[idx] - upper_limits[idx])
        elif q[idx] < lower_limits[idx]:
            grad[idx] = -gain * (q[idx] - lower_limits[idx])
    return grad


def joint_center_nullspace_component(model, q, gain=1.0):
    joint_center_positions = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
    return gain * (joint_center_positions - q)
