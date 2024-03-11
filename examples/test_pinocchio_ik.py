import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import copy
import numpy as np
from os.path import dirname, join, abspath
import time


def create_robot_models():
    # Get the URDF file for the robot model
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "models")
    urdf_filename = join(pinocchio_model_dir, "panda_description", "urdf", "panda.urdf")

    # Load the model from URDF
    model = pinocchio.buildModelFromUrdf(urdf_filename)

    # Load collision model
    collision_model = pinocchio.buildGeomFromUrdf(
        model,
        urdf_filename,
        pinocchio.GeometryType.COLLISION,
        package_dirs=pinocchio_model_dir,
    )
    collision_model.addAllCollisionPairs()

    # Load visual model
    visual_model = pinocchio.buildGeomFromUrdf(
        model,
        urdf_filename,
        pinocchio.GeometryType.VISUAL,
        package_dirs=pinocchio_model_dir,
    )

    return (model, collision_model, visual_model)


def get_random_state(model):
    return np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)


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


def check_within_limits(model, q):
    return np.all(q >= model.lowerPositionLimit) and np.all(
        q <= model.upperPositionLimit
    )


def visualize_frame(name, tform):
    import meshcat.geometry as mg

    FRAME_AXIS_POSITIONS = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])
        .astype(np.float32)
        .T
    )
    FRAME_AXIS_COLORS = (
        np.array(
            [[1, 0, 0], [1, 0.6, 0], [0, 1, 0], [0.6, 1, 0], [0, 0, 1], [0, 0.6, 1]]
        )
        .astype(np.float32)
        .T
    )
    viz.viewer[name].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=0.2 * FRAME_AXIS_POSITIONS,
                color=FRAME_AXIS_COLORS,
            ),
            mg.LineBasicMaterial(
                linewidth=4,
                vertexColors=True,
            ),
        )
    )
    viz.viewer[name].set_transform(tform.homogeneous)


def solve_ik(model, target_frame, target_tform=None, init_state=None):
    target_frame_id = model.getFrameId(target_frame)
    viz.displayFrames(True, frame_ids=[target_frame_id])

    # Create a random initial state and/or target pose, if not specified
    if init_state is None:
        init_state = get_random_state(model)
    if target_tform is None:
        q_target = get_random_state(model)
        pinocchio.framesForwardKinematics(model, data, q_target)
        target_tform = copy.deepcopy(data.oMf[target_frame_id])
    visualize_frame("ik_target_pose", target_tform)

    # IK Hyperparameters
    MAX_ITERS = 200
    MAX_RETRIES = 10
    MAX_TRANSLATION_ERROR = 1e-3
    MAX_ROTATION_ERROR = 1e-3
    DAMPING = 1e-3
    MIN_STEP = 0.1
    MAX_STEP = 0.5

    # Initialize IK
    solved = False
    n_tries = 0
    q_cur = init_state
    viz.display(q_cur)
    time.sleep(0.5)  # Needed to render

    while n_tries < MAX_RETRIES:
        n_iters = 0
        max_error = 0
        while n_iters < MAX_ITERS:
            # Compute forward kinematics at the current state
            pinocchio.framesForwardKinematics(model, data, q_cur)
            cur_tform = data.oMf[target_frame_id]

            # Check the error using actInv
            error = target_tform.actInv(cur_tform)
            error = -pinocchio.log(error).vector
            error_norm = np.linalg.norm(error)
            max_error = max(max_error, error_norm)
            # print(f"Iteration {n_iters}, Error norm: {error_norm}")
            if (
                np.linalg.norm(error[:3]) < MAX_TRANSLATION_ERROR
                and np.linalg.norm(error[3:]) < MAX_ROTATION_ERROR
            ):
                # Wrap to the range -/+ pi and check joint limits
                q_cur = (q_cur + np.pi) % (2 * np.pi) - np.pi
                if check_within_limits(model, q_cur):
                    print("Solved and within joint limits!")
                    solved = True
                else:
                    print("Solved, but outside joint limits.")
                break

            # Calculate the Jacobian
            J = pinocchio.computeFrameJacobian(
                model,
                data,
                q_cur,
                target_frame_id,
                pinocchio.ReferenceFrame.LOCAL,
            )

            # Solve for the gradient using damping and nullspace components,
            # as specified
            jjt = J.dot(J.T) + DAMPING**2 * np.eye(6)
            # nullspace_component = zero_nullspace_component(model)
            nullspace_component = joint_limit_nullspace_component(
                model, q_cur, padding=0.05, gain=1.0
            ) + joint_center_nullspace_component(model, q_cur, gain=0.1)

            # Gradient descent step
            alpha = MIN_STEP + (1.0 - error_norm / max_error) * (MAX_STEP - MIN_STEP)
            q_cur += alpha * (
                J.T @ (np.linalg.solve(jjt, error - J @ (nullspace_component)))
                + nullspace_component
            )

            n_iters += 1
            viz.display(q_cur)
            time.sleep(0.05)

        if solved:
            print(f"Solved in {n_tries+1} tries.")
            break
        else:
            q_cur = get_random_state(model)
            n_tries += 1
            print(f"Retry {n_tries}")

    if solved:
        return q_cur
    else:
        return None


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = create_robot_models()
    data = model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    # Solve IK
    NUM_SOLVES = 10
    for i in range(NUM_SOLVES):
        q_sol = solve_ik(model, "panda_hand")
        np.set_printoptions(precision=3)
        print(f"Solution configuration:\n{q_sol}\n")
