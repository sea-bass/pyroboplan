import copy
from os.path import dirname, join, abspath
import numpy as np
import pinocchio

from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions


def load_panda_models():
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "..", "models")
    package_dir = join(pinocchio_model_dir, "panda_description")
    urdf_filename = join(package_dir, "urdf", "panda.urdf")
    return pinocchio.buildModelsFromUrdf(
        urdf_filename, package_dirs=pinocchio_model_dir
    )


def test_ik_solve_trivial_ik():
    model, _, _ = load_panda_models()
    data = model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform to the current joint state's FK
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q)
    target_tform = data.oMf[target_frame_id]

    # Solve
    ik = DifferentialIk(model, data=None, visualizer=None, verbose=False)
    options = DifferentialIkOptions()
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q,
        options=options,
        nullspace_components=[],
    )

    # The result should be very, very close to the initial state
    np.testing.assert_almost_equal(q, q_sol, decimal=6)


def test_ik_solve_ik():
    model, _, _ = load_panda_models()
    data = model.createData()
    target_frame = "panda_hand"

    # Initial joint states
    q_init = np.array([0.0, 1.57, 0.0, 0.0, 1.57, 1.57, 0.0, 0.0, 0.0])

    # Set the target transform 1 cm along z axis
    offset = 0.01
    target_frame_id = model.getFrameId(target_frame)
    pinocchio.framesForwardKinematics(model, data, q_init)
    target_tform = copy.deepcopy(data.oMf[target_frame_id])
    target_tform.translation[2] = target_tform.translation[2] + offset

    # Solve it
    ik = DifferentialIk(model, data=None, visualizer=None, verbose=False)
    options = DifferentialIkOptions()
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=q_init,
        options=options,
        nullspace_components=[],
    )
    assert q_sol is not None

    # Get the resulting FK
    pinocchio.framesForwardKinematics(model, data, q_sol)
    result_tform = data.oMf[target_frame_id]

    # Assert that they are very, very close, lines up with the max pos/rot error from
    # the options.
    error = target_tform.actInv(result_tform)
    np.testing.assert_almost_equal(np.identity(3), error.rotation, decimal=3)
    np.testing.assert_almost_equal(np.zeros(3), error.translation, decimal=3)


def test_ik_solve_impossible_ik():
    model, _, _ = load_panda_models()
    target_frame = "panda_hand"

    # Target is unreachable by the panda
    R = np.identity(3)
    T = np.array([10.0, 10.0, 10.0])
    target_tform = pinocchio.SE3(R, T)

    # Solve
    ik = DifferentialIk(model, data=None, visualizer=None, verbose=False)
    options = DifferentialIkOptions()
    q_sol = ik.solve(
        target_frame,
        target_tform,
        init_state=None,
        options=options,
        nullspace_components=[],
    )

    assert q_sol is None, "Solution should be impossible!"
