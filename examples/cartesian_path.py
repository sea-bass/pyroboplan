"""
This example shows PyRoboPlan capabilities for Cartesian planning to track
task-space, or operational space, motions with straight-line interpolation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from scipy.spatial.transform import Rotation
import time

from pyroboplan.core.utils import extract_cartesian_pose
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.models.panda import (
    load_models,
    add_self_collisions,
)
from pyroboplan.planning.cartesian_planner import (
    CartesianPlanner,
    CartesianPlannerOptions,
)
from pyroboplan.visualization.meshcat_utils import visualize_frames


# Create models and data
model, collision_model, visual_model = load_models()
add_self_collisions(model, collision_model)
data = model.createData()
collision_data = collision_model.createData()

# Define the Cartesian path from a start joint configuration
target_frame = "panda_hand"
q_start = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0, 0.0])
init = extract_cartesian_pose(model, target_frame, q_start, data=data)
rot = Rotation.from_euler("z", 60, degrees=True).as_matrix()
rot_neg = Rotation.from_euler("z", -60, degrees=True).as_matrix()
tforms = [
    init,
    init * pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),
    init * pinocchio.SE3(rot, np.array([0.0, 0.25, 0.2])),
    init * pinocchio.SE3(rot_neg, np.array([0.0, -0.25, 0.2])),
    init * pinocchio.SE3(np.eye(3), np.array([0.2, 0.0, 0.0])),
    init,
]

# Initialize visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.displayFrames(True, frame_ids=[model.getFrameId(target_frame)])

# Define the IK solve to use.
ik = DifferentialIk(
    model,
    data=data,
    collision_model=collision_model,
    options=DifferentialIkOptions(max_retries=5, rng_seed=None),
)

# Create the Cartesian Planner over the entire desired path.
options = CartesianPlannerOptions(
    use_trapezoidal_scaling=True,
    max_linear_velocity=0.1,
    max_linear_acceleration=0.5,
    max_angular_velocity=1.0,
    max_angular_acceleration=1.0,
)

TOPP_RA = False
dt = 0.025
curr_time = 0

planner = CartesianPlanner(model, target_frame, tforms, ik, options=options)
success, t_vec, q_vec = planner.generate(q_start, dt)

tforms_to_show = planner.generated_tforms[::5]
viz.display(q_start)
visualize_frames(viz, "cartesian_plan", tforms_to_show, line_length=0.05, line_width=1)
time.sleep(0.5)

plt.ion()
plt.figure()
plt.title("Pyroboplan - Joint Position Trajectories")

if TOPP_RA:
    import toppra as ta

    # replan path with TOPP-RA
    vlims = model.velocityLimit
    alims = 10 * np.ones_like(vlims)
    # Define the geometric path and two constraints.
    path = ta.SplineInterpolator(t_vec, q_vec.T)
    pc_vel = ta.constraint.JointVelocityConstraint(vlims)
    pc_acc = ta.constraint.JointAccelerationConstraint(alims)

    instance = ta.algorithm.TOPPRA(
        [pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel"
    )
    jnt_traj = instance.compute_trajectory()

    # Overwrite variables with TOPP-RA data
    t_vec = np.linspace(0, jnt_traj.duration, q_vec.shape[1])
    dt = jnt_traj.duration / q_vec.shape[1]
    q_vec = jnt_traj(t_vec).T
    plt.title("TOPP-RA - Joint Position Trajectories")

for idx in range(q_vec.shape[0]):
    plt.plot(t_vec, q_vec[idx, :])

time_line = plt.axvline(x=curr_time, color="k", linestyle="--")
plt.legend(model.names[1:])
plt.show()

if success:
    print("Cartesian planning successful!")
    input("Press 'Enter' to animate the path.")
    for idx in range(q_vec.shape[1]):
        viz.display(q_vec[:, idx])
        plt.pause(dt)
        curr_time += dt
        time_line.set_xdata([curr_time, curr_time])
else:
    print("Cartesian planning failed.")
