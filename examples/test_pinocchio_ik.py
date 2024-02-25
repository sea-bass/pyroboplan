import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import copy
import numpy as np
from os.path import dirname, join, abspath
import time


def create_robot_models():
    # Get the URDF file for the robot model
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "models")
    urdf_filename = join(
        pinocchio_model_dir, "ur_description", "urdf", "ur5_joint_limited_robot.urdf"
    )

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


def joint_limit_nullspace_component(model, q, gain=0.1, padding=0.0):
    grad = zero_nullspace_component(model)
    for idx in range(len(grad)):
        if q[idx] > model.upperPositionLimit[idx] - padding:
            grad[idx] = -gain * (q[idx] - model.upperPositionLimit[idx] + padding)
        elif q[idx] < model.lowerPositionLimit[idx] + padding:
            grad[idx] = -gain * (q[idx] - model.lowerPositionLimit[idx] - padding)
    return grad


def elbow_up_nullspace_component(model, q, gain=0.1):
    grad = zero_nullspace_component(model)
    joint_idx = model.getJointId("elbow_joint")
    if q[joint_idx] < 0:
        grad[joint_idx] = -gain * q[joint_idx]
    return grad


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
    MAX_ITERS = 500
    MAX_RETRIES = 10
    MAX_TRANSLATION_ERROR = 1e-4
    MAX_ROTATION_ERROR = 1e-4
    DAMPING = 1e-3
    ALPHA = 0.1

    # Initialize IK
    solved = False
    n_tries = 0
    q_cur = init_state
    while n_tries < MAX_RETRIES:
        n_iters = 0
        while n_iters < MAX_ITERS:
            # Compute forward kinematics at the current state
            pinocchio.framesForwardKinematics(model, data, q_cur)
            cur_tform = data.oMf[target_frame_id]

            # Check the error using actInv
            error = target_tform.actInv(cur_tform)
            error = pinocchio.log(error).vector
            # print(f"Iteration {n_iters}, Error: {error}")
            if (
                np.linalg.norm(error[:3]) < MAX_TRANSLATION_ERROR
                and np.linalg.norm(error[3:]) < MAX_ROTATION_ERROR
            ):
                # Wrap to the range -/+ pi and check joint limits
                q_cur = q_cur % np.pi
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
            nullspace_component = joint_limit_nullspace_component(
                model, q_cur
            ) + elbow_up_nullspace_component(model, q_cur)
            vel = (
                -J.T.dot(np.linalg.solve(jjt, error - J @ nullspace_component))
                - nullspace_component
            )
            q_cur = pinocchio.integrate(model, q_cur, vel * ALPHA)

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

    return q_cur


if __name__ == "__main__":
    # Create models and data
    model, collision_model, visual_model = create_robot_models()
    data = model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    time.sleep(0.5)  # Needed to render

    # Solve IK
    q_sol = solve_ik(model, "ee_link")
    np.set_printoptions(precision=3)
    print(f"\nSolution configuration:\n{q_sol}")
