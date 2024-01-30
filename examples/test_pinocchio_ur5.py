import pinocchio
from pinocchio.visualize import MeshcatVisualizer

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

print('model name: ' + model.name)

# Create data required by the algorithms
data = model.createData()

# Sample a random configuration
q = pinocchio.randomConfiguration(model)
print('q: %s' % q.T)

# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)

# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print(("{:<24} : {: .2f} {: .2f} {: .2f}"
          .format(name, *oMi.translation.T.flat)))


viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
viz.initViewer(open=True)
viz.loadViewerModel()
viz.displayFrames(True)

def random_loop():
    for i in range(100):
        q = pinocchio.randomConfiguration(model)

        pinocchio.framesForwardKinematics(model, data, q)
        tool_tform = data.oMf[model.getFrameId("tool0")]
        print(f"q: {q}\ntool_tform:\n{tool_tform}")

        # Compute collisions
        pinocchio.computeCollisions(
            model,
            data,
            collision_model,
            collision_data,
            q,
            False
        )
        # Print the status of collision for all collision pairs
        n_collision_pairs = len(collision_model.collisionPairs)
        for k in range(n_collision_pairs):
            cr = collision_data.collisionResults[k]
            cp = collision_model.collisionPairs[k]
            if cr.isCollision():
                print("collision between:", collision_model.geometryObjects[cp.first].name,
                    " and ", collision_model.geometryObjects[cp.second].name)

        print("\n")

        viz.display(q)
        viz.drawFrameVelocities(frame_id=model.getFrameId("tool0"))

        time.sleep(0.5)

def sim_loop():
    tau0 = np.zeros(model.nv)
    qs = [np.array([0.0, -np.pi/2 + 0.01, 0.0, 0.0, 0.0, 0.0])]
    vs = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]

    dt = 0.005
    total_time = 30

    nsteps = int(total_time / dt)
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]
        a1 = pinocchio.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pinocchio.integrate(model, q, dt * vnext)

        pinocchio.framesForwardKinematics(model, data, qnext)
        tool_tform = data.oMf[model.getFrameId("tool0")]
        print(f"q: {q}\ntool_tform:\n{tool_tform}")

        # Compute collisions
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, qnext, False
        )
        # Print the status of collision for all collision pairs
        for k in range(len(collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = collision_model.collisionPairs[k]
            if cr.isCollision():
                print("collision between:", collision_model.geometryObjects[cp.first].name,
                    " and ", collision_model.geometryObjects[cp.second].name)

        qs.append(qnext)
        vs.append(vnext)

        viz.display(qnext)

        time.sleep(dt)
    return qs, vs


sim_loop()
