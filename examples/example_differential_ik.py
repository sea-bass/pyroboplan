import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from os.path import dirname, join, abspath

from pyroboplan.core.utils import get_random_state, get_random_transform
from pyroboplan.ik.differential_ik import DifferentialIk, DifferentialIkOptions
from pyroboplan.ik.nullspace_components import joint_limit_nullspace_component

if __name__ == "__main__":
    # Create models and data
    pinocchio_model_dir = join(dirname(str(abspath(__file__))), "..", "models")
    package_dir = join(pinocchio_model_dir, "panda_description")
    urdf_filename = join(package_dir, "urdf", "panda.urdf")

    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_filename, package_dirs=pinocchio_model_dir
    )
    data = model.createData()

    # Initialize visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    np.set_printoptions(precision=3)

    # Set up the IK solver
    ik = DifferentialIk(model, data, visualizer=viz, verbose=True)
    target_frame = "panda_hand"
    options = DifferentialIkOptions()
    nullspace_components = [
        lambda model, q: joint_limit_nullspace_component(model, q, gain=0.5)
    ]

    # Solve IK several times and print the results
    for _ in range(10):
        init_state = get_random_state(model)
        target_tform = get_random_transform(model, target_frame)
        q_sol = ik.solve(
            target_frame, target_tform, options=options, nullspace_components=[]
        )
        print(f"Solution configuration:\n{q_sol}\n")
