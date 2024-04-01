import numpy as np
import pinocchio


def get_random_state(model):
    return np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)


def get_random_transform(model, target_frame):
    q_target = get_random_state(model)
    data = model.createData()
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_target)
    return data.oMf[target_frame_id]


def check_within_limits(model, q):
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )
