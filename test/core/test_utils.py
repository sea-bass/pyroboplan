import numpy as np
import pinocchio
import pytest

from pyroboplan.core.utils import (
    check_collisions_at_state,
    check_collisions_along_path,
    check_within_limits,
    configuration_distance,
    extract_cartesian_pose,
    extract_cartesian_poses,
    get_collision_geometry_ids,
    get_path_length,
    get_random_state,
    get_random_transform,
    get_random_collision_free_state,
    get_random_collision_free_transform,
    set_collisions,
)
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
    add_object_collisions,
)


# Use a fixed seed for random number generation in tests.
np.random.seed(1234)


def test_configuration_distance():
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    q_end = np.array([0.3, 0.0, -0.4, 0.0, 0.0])
    assert configuration_distance(q_start, q_end) == pytest.approx(0.5)


def test_get_path_length():
    q_path = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.3, 0.0, -0.4, 0.0, 0.0]),
        np.array([0.3, 1.0, -0.4, 0.0, 0.0]),
        np.array([0.3, 1.0, -0.4, 0.0, 10.0]),
    ]
    assert get_path_length(q_path) == pytest.approx(11.5)


def test_get_random_state():
    model, _, _ = load_models()
    for _ in range(10):
        state = get_random_state(model)
        assert np.all(state >= model.lowerPositionLimit)
        assert np.all(state <= model.upperPositionLimit)


def test_get_random_state_with_scalar_padding():
    model, _, _ = load_models()
    padding = 0.01
    for _ in range(10):
        state = get_random_state(model, padding=padding)
        assert np.all(state >= model.lowerPositionLimit + padding)
        assert np.all(state <= model.upperPositionLimit - padding)


def test_get_random_state_with_vector_padding():
    model, _, _ = load_models()
    padding = [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.01, 0.01]
    for _ in range(10):
        state = get_random_state(model, padding=padding)
        assert np.all(state >= model.lowerPositionLimit + padding)
        assert np.all(state <= model.upperPositionLimit - padding)


def test_get_random_transform():
    model, _, _ = load_models()
    tform = get_random_transform(model, "panda_hand")
    assert isinstance(tform, pinocchio.SE3)


def test_get_random_collision_free_state():
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

    state = get_random_collision_free_state(model, collision_model)
    assert np.all(state >= model.lowerPositionLimit)
    assert np.all(state <= model.upperPositionLimit)


def test_get_random_collision_free_ransform():
    model, collision_model, visual_model = load_models()
    add_self_collisions(model, collision_model)
    add_object_collisions(model, collision_model, visual_model)

    tform = get_random_collision_free_transform(model, collision_model, "panda_hand")
    assert isinstance(tform, pinocchio.SE3)


def test_within_limits():
    model, _, _ = load_models()

    assert check_within_limits(model, get_random_state(model))
    assert not check_within_limits(model, model.upperPositionLimit + 0.2)
    assert not check_within_limits(model, model.lowerPositionLimit - 0.2)


def test_extract_cartesian_pose():
    model, _, _ = load_models()

    q = get_random_state(model)
    tform = extract_cartesian_pose(model, "panda_hand", q)

    assert isinstance(tform, pinocchio.SE3)


def test_extract_cartesian_poses():
    model, _, _ = load_models()

    q_vec = [get_random_state(model) for _ in range(5)]
    tforms = extract_cartesian_poses(model, "panda_hand", q_vec)

    assert len(tforms) == 5
    for tform in tforms:
        assert isinstance(tform, pinocchio.SE3)


def test_check_collisions_at_state():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)

    # while True:
    #     q = get_random_state(model)
    #     if check_collisions_at_state(model, collision_model, q):
    #         print(f"Q collision: {q}")
    #     else:
    #         print(f"Q free {q}")

    collision_state = np.array(
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

    free_state = np.array(
        [
            0.5990507,
            -1.46131209,
            0.90076761,
            -2.70060109,
            0.77612686,
            1.65669724,
            -1.72831715,
            0.02858002,
            0.01910153,
        ]
    )

    assert check_collisions_at_state(model, collision_model, collision_state)
    assert not check_collisions_at_state(model, collision_model, free_state)


def test_check_collisions_along_path():
    model, collision_model, _ = load_models()
    add_self_collisions(model, collision_model)

    collision_state = np.array(
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

    free_state_1 = np.array(
        [
            0.5990507,
            -1.46131209,
            0.90076761,
            -2.70060109,
            0.77612686,
            1.65669724,
            -1.72831715,
            0.02858002,
            0.01910153,
        ]
    )

    free_state_2 = np.array(
        [
            1.61357261,
            0.81296483,
            -2.15473055,
            -1.38469138,
            -0.20183876,
            3.72713384,
            0.34544908,
            0.02216068,
            0.0398552,
        ]
    )

    assert check_collisions_along_path(
        model, collision_model, [free_state_1, collision_state, free_state_2]
    )
    assert not check_collisions_along_path(
        model, collision_model, [free_state_1, free_state_2, free_state_1]
    )


def test_get_collision_geometry_ids():
    model, collision_model, _ = load_models()

    # This is directly the name of a collision geometry.
    body = "panda_link1_0"
    ids = get_collision_geometry_ids(model, collision_model, body)
    assert ids == [collision_model.getGeometryId(body)]

    # This is the name of a frame in the main model.
    body = "panda_link3"
    ids = get_collision_geometry_ids(model, collision_model, body)
    assert ids == [collision_model.getGeometryId("panda_link3_0")]

    # This is not a valid body name.
    body = "panda_link1000"
    ids = get_collision_geometry_ids(model, collision_model, body)
    assert ids == []


def test_set_collisions():
    model, collision_model, _ = load_models()
    assert len(collision_model.collisionPairs) == 0

    # Enable the collisions
    set_collisions(model, collision_model, "panda_link4", "panda_link5", True)
    assert len(collision_model.collisionPairs) == 1
    assert collision_model.collisionPairs[0].first == 4
    assert collision_model.collisionPairs[0].second == 5

    # Disable the collisions
    set_collisions(model, collision_model, "panda_link4", "panda_link5", False)
    assert len(collision_model.collisionPairs) == 0
