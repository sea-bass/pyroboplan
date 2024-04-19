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
from pyroboplan.planning.cartesian_planner import CartesianPlanner
from pyroboplan.visualization.meshcat_utils import visualize_frames


# Create models and data
model, collision_model, visual_model = load_models()
add_self_collisions(model, collision_model)
data = model.createData()
collision_data = collision_model.createData()

target_frame = "panda_hand"

# Initialize visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.displayFrames(True, frame_ids=[model.getFrameId(target_frame)])

# Panda home pose
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

ik = DifferentialIk(
    model,
    data=data,
    collision_model=collision_model,
)
options = DifferentialIkOptions()
options.max_retries = 5

planner = CartesianPlanner(model, "panda_hand", tforms, ik)

dt = 0.025
success, t_vec, q_vec = planner.generate(q_start, dt)

viz.display(q_start)
visualize_frames(
    viz, "cartesian_plan", planner.generated_tforms[::5], line_length=0.05, line_width=1
)
time.sleep(1.0)

import matplotlib.pyplot as plt

plt.ion()
plt.figure()
plt.title("Joint position trajectories")
for i in range(q_vec.shape[0]):
    plt.plot(t_vec, q_vec[i, :])
plt.show()

if success:
    print("Cartesian planning successful!")
    input("Press 'Enter' to animate the path.")
    for idx in range(q_vec.shape[1]):
        viz.display(q_vec[:, idx])
        time.sleep(dt)
else:
    print("Cartesian planning failed.")
