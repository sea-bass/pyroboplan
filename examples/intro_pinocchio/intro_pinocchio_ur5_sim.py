"""
This example loads a model of a UR5 arm from a URDF file.
It then demonstrates basic capabilities in Pinocchio such as forward kinematics,
Jacobian computation, collision checking, and dynamics simulation.
"""

import meshcat.geometry as mg
import numpy as np
import os
import time

import pinocchio
from pinocchio.visualize import MeshcatVisualizer
from pyroboplan.models.utils import get_example_models_folder


def main():
    # Load the models from the URDF file.
    model_dir = get_example_models_folder()
    urdf_filename = os.path.join(
        model_dir, "ur5_description", "urdf", "ur5_gripper.urdf"
    )
    model, collision_model, visual_model = pinocchio.buildModelsFromUrdf(
        urdf_filename, package_dirs=model_dir
    )

    # Modify the collision model for display.
    srdf_filename = os.path.join(
        model_dir, "ur5_description", "srdf", "ur5_gripper.srdf"
    )
    collision_model.addAllCollisionPairs()
    pinocchio.removeCollisionPairs(model, collision_model, srdf_filename)
    for cobj in collision_model.geometryObjects:
        cobj.meshColor = np.array([0.7, 0.7, 0.7, 0.2])

    # Create data required by the algorithms
    data = model.createData()
    collision_data = collision_model.createData()

    # Sample a random configuration
    q = pinocchio.randomConfiguration(model)
    print("Joint configuration: %s" % q.T)

    # Perform the forward kinematics over the kinematic tree
    pinocchio.forwardKinematics(model, data, q)

    # Print out the placement of each joint of the kinematic tree
    for name, oMi in zip(model.names, data.oMi):
        print(
            ("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
        )

    # Initialize visualization with MeshCat.
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayCollisions(True)
    viz.displayVisuals(True)
    viz.displayFrames(True)

    # Set up initial conditions for the simulation.
    q = np.array([0.0, -np.pi / 2 + 0.01, 0.0, 0.0, 0.0, 0.0])
    v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    tau0 = np.zeros(model.nv)

    dt = 0.01
    total_time = 30

    nsteps = int(total_time / dt)
    for _ in range(nsteps):
        # Perform integration using the model dynamics.
        a1 = pinocchio.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qnext = pinocchio.integrate(model, q, dt * vnext)

        # Compute forward kinematics and the Jacobian for a specified frame.
        target_frame_id = model.getFrameId("tool0")

        pinocchio.framesForwardKinematics(model, data, qnext)
        tool_tform = data.oMf[target_frame_id]
        print(f"q: {q}\nFrame Transform:\n{tool_tform}")

        J = pinocchio.computeFrameJacobian(
            model,
            data,
            q,
            target_frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        print(f"Frame Jacobian:\n{J}")

        # Compute self-collisions and collision distances using HPP-FCL.
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, qnext, False
        )
        pinocchio.computeDistances(model, data, collision_model, collision_data, qnext)

        # Visualize the collisions and distance information.
        visualize_collisions(viz, collision_model, collision_data)

        viz.display(qnext)
        time.sleep(dt)
        q = qnext
        v = vnext


def visualize_collisions(viz, collision_model, collision_data):
    contacts = []
    distances = []
    for k in range(len(collision_model.collisionPairs)):
        cr = collision_data.collisionResults[k]
        cp = collision_model.collisionPairs[k]
        dr = collision_data.distanceResults[k]
        if cr.isCollision():
            print(
                "collision between:",
                collision_model.geometryObjects[cp.first].name,
                " and ",
                collision_model.geometryObjects[cp.second].name,
            )
            for contact in cr.getContacts():
                contacts.extend(
                    [
                        contact.pos,
                        contact.pos - contact.normal * contact.penetration_depth,
                    ]
                )
        else:
            distances.extend([dr.getNearestPoint1(), dr.getNearestPoint2()])

    if len(contacts) == 0:
        print("no collisions!")

    viz.viewer["collision_display"].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=np.array(contacts).T,
                color=np.array([[1.0, 0.0, 0.0] for _ in contacts]).T,
            ),
            mg.LineBasicMaterial(
                linewidth=3,
                vertexColors=True,
            ),
        )
    )
    viz.viewer["distance_display"].set_object(
        mg.LineSegments(
            mg.PointsGeometry(
                position=np.array(distances).T,
                color=np.array([[0.0, 0.6, 0.0] for _ in distances]).T,
            ),
            mg.LineBasicMaterial(
                linewidth=3,
                vertexColors=True,
            ),
        )
    )


if __name__ == "__main__":
    main()
