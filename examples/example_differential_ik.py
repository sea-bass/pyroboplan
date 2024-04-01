import pinocchio
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
from os.path import dirname, join, abspath

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
    options = DifferentialIkOptions()
    nullspace_components = [
        lambda model, q: joint_limit_nullspace_component(model, q, gain=0.5)
    ]

    # Solve IK several times and print the results
    for _ in range(10):
        q_sol = ik.solve("panda_hand", options=options, nullspace_components=[])
        print(f"Solution configuration:\n{q_sol}\n")
