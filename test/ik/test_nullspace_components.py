import numpy as np
import pytest

from pyroboplan.ik.nullspace_components import (
    zero_nullspace_component,
    joint_limit_nullspace_component,
    joint_center_nullspace_component,
    collision_avoidance_nullspace_component,
)
from pyroboplan.models.panda import load_models, add_self_collisions
from pyroboplan.core.utils import get_random_state


def test_zero_nullspace_component():
    model, _, _ = load_models()
    q = get_random_state(model)
    component = zero_nullspace_component(model, q)
    assert component.shape == (9,)
    assert np.all(component == 0.0)


def test_joint_limit_nullspace_component():
    model, _, _ = load_models()

    # Nominal case
    q = get_random_state(model)
    component = joint_limit_nullspace_component(model, q)
    assert component.shape == (9,)
    assert np.all(component == 0.0)

    # Off limits case
    q = get_random_state(model)
    q[1] = model.lowerPositionLimit[1] - 1.0
    q[5] = model.upperPositionLimit[5] + 1.0
    component = joint_limit_nullspace_component(model, q)
    assert component.shape == (9,)
    assert component[1] == pytest.approx(1.0)
    assert component[5] == pytest.approx(-1.0)

    # Off limits case with nondefault parameters
    q = get_random_state(model)
    q[1] = model.lowerPositionLimit[1] - 1.0
    q[5] = model.upperPositionLimit[5] + 1.0
    component = joint_limit_nullspace_component(model, q, gain=0.5, padding=0.1)
    assert component.shape == (9,)
    assert component[1] == pytest.approx(0.55)
    assert component[5] == pytest.approx(-0.55)


def test_joint_center_nullspace_component():
    model, _, _ = load_models()

    # Exactly centered case
    q = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
    component = joint_center_nullspace_component(model, q)
    assert component.shape == (9,)
    assert np.all(component == pytest.approx(0.0))

    # Not centered and nondefault parameters
    q = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
    q[2] += 0.25
    q[6] -= 0.5
    component = joint_center_nullspace_component(model, q, gain=0.5)
    assert component.shape == (9,)
    assert component[2] == pytest.approx(-0.125)
    assert component[6] == pytest.approx(0.25)


def test_collision_avoidance_nullpace_component():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)
    data = model.createData()
    collision_data = collision_model.createData()

    # Collision free case
    q = 0.5 * (model.lowerPositionLimit + model.upperPositionLimit)
    component = collision_avoidance_nullspace_component(
        model,
        data,
        collision_model,
        collision_data,
        q,
    )
    assert component.shape == (9,)
    assert np.all(component == 0.0)

    # Collision case should have nonzero components
    q = np.array(
        [
            1.98103111,
            0.27084068,
            -1.60819541,
            -3.07282607,
            -0.56607161,
            2.2466373,
            2.80244523,
            0.01154539,
            0.00453211,
        ]
    )
    component = collision_avoidance_nullspace_component(
        model,
        data,
        collision_model,
        collision_data,
        q,
        gain=1.0,
        dist_padding=0.05,
    )
    assert component.shape == (9,)
    assert not np.all(component == 0.0)
