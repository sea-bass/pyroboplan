import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import copy
import numpy as np
from os.path import dirname, join, abspath
import time

# This path refers to Pinocchio source code but you can define your own directory here.
pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "models")

# You should change here to set up your own URDF file or just pass it as an argument of this example.
urdf_filename = pinocchio_model_dir + '/ur_description/urdf/ur5_robot.urdf'

# Load the urdf model
model = pinocchio.buildModelFromUrdf(urdf_filename)

# Load collision model
mesh_path = pinocchio_model_dir
collision_model = pinocchio.buildGeomFromUrdf(
    model,
    urdf_filename,
    pinocchio.GeometryType.COLLISION,
    package_dirs=pinocchio_model_dir
)
collision_model.addAllCollisionPairs()
collision_data = pinocchio.GeometryData(collision_model)

# Load visual model
mesh_path = pinocchio_model_dir
visual_model = pinocchio.buildGeomFromUrdf(
    model,
    urdf_filename,
    pinocchio.GeometryType.VISUAL,
    package_dirs=pinocchio_model_dir
)

# Create data required by the algorithms
data = model.createData()

# Initialize visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
viz.initViewer(open=True)
viz.loadViewerModel()

def get_random_state(model):
    return np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)

def visualize_frame(name, tform):
    import meshcat.geometry as mg
    FRAME_AXIS_POSITIONS = np.array([
        [0, 0, 0], [1, 0, 0],
        [0, 0, 0], [0, 1, 0],
        [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
    FRAME_AXIS_COLORS = np.array([
        [1, 0, 0], [1, 0.6, 0],
        [0, 1, 0], [0.6, 1, 0],
        [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
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

def ik_loop():
    target_frame = "ee_link"
    frame_id = model.getFrameId(target_frame)
    viz.displayFrames(True, frame_ids=[frame_id])

    # Create target pose
    q_target = get_random_state(model)
    pinocchio.framesForwardKinematics(model, data, q_target)
    target_tform = copy.deepcopy(data.oMf[frame_id])
    visualize_frame("ik_target_pose", target_tform)

    # IK Hyperparameters
    MAX_ITERS = 500
    MAX_RETRIES = 10
    MAX_TRANSLATION_ERROR = 1e-4
    MAX_ROTATION_ERROR = 1e-4
    DAMPING = 1e-4
    DT = 0.1

    # Initialize IK
    solved = False
    n_tries = 0
    q_cur = get_random_state(model)
    while n_tries < MAX_RETRIES:
        n_iters = 0
        while n_iters < MAX_ITERS:
            # Compute forward kinematics at the current state
            pinocchio.framesForwardKinematics(model, data, q_cur)
            cur_tform = data.oMf[frame_id]

            # Check the error using actInv
            error = target_tform.actInv(cur_tform)
            error = pinocchio.log(error).vector
            # print(f"Iteration {n_iters}, Error: {error}")
            if np.linalg.norm(error[:3]) < MAX_TRANSLATION_ERROR and np.linalg.norm(error[3:]) < MAX_ROTATION_ERROR:           
                solved = True
                break

            # Calculate the Jacobian
            J = pinocchio.computeFrameJacobian(
                model, data, q_cur, frame_id,
                pinocchio.ReferenceFrame.LOCAL,
            )
            vel = - J.T.dot(
                np.linalg.solve(J.dot(J.T) + DAMPING**2 * np.eye(6), error)
            )
            q_cur = pinocchio.integrate(model, q_cur, vel * DT)

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

ik_loop()
