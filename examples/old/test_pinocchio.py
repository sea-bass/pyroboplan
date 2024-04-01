import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import hppfcl
import numpy as np
import time


def create_model():
    model = pin.Model()

    # Create a joint
    jid0 = model.addJoint(
        0,  # joint id
        pin.JointModelRY(),  # Revolute
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),  # Joint placement
        "joint0",  # name
    )
    model.appendBodyToJoint(
        jid0,  # joint id
        pin.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pin.SE3.Identity(),  # transform from joint frame?
    )

    jid1 = model.addJoint(
        jid0,  # joint id
        pin.JointModelRY(),  # Revolute
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 1.0])),  # Joint placement
        "joint1",  # name
    )
    model.appendBodyToJoint(
        jid1,  # joint id
        pin.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.375]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pin.SE3.Identity(),  # transform from joint frame?
    )

    # Add tip frame
    frame = pin.Frame(
        "tip",  # name
        jid1,  # Joint parent
        2,  # Previous frame
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 1.0])),  # Relative tform,
        pin.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame, False)

    print(model)

    # Forward kinematics
    data = model.createData()

    q = np.array([np.pi / 4, -np.pi / 3])
    pin.framesForwardKinematics(model, data, q)

    print("OMI")
    print(data.oMi.tolist())
    print("OMF")
    print(data.oMf.tolist())

    J = pin.computeJointJacobians(model, data, q)
    print(f"Joint Jacobians:\n{J}")

    JF = pin.computeFrameJacobian(
        model,
        data,
        q,
        model.getFrameId("tip"),
        pin.ReferenceFrame.LOCAL,
    )
    print(f"Frame Jacobian:\n{JF}")

    return model


def create_visual_model(model):
    visual_model = pin.GeometryModel()
    link0 = pin.GeometryObject(
        "link0",
        model.getJointId("joint0"),
        hppfcl.Box(0.05, 0.05, 1.0),
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.5])),
    )
    link0.meshColor = np.array([1.0, 0.0, 0.0, 1.0])
    visual_model.addGeometryObject(link0)
    link1 = pin.GeometryObject(
        "link1",
        model.getJointId("joint1"),
        hppfcl.Box(0.05, 0.05, 1.0),
        pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.5])),
    )
    link1.meshColor = np.array([0.0, 0.0, 1.0, 1.0])
    visual_model.addGeometryObject(link1)

    return visual_model


if __name__ == "__main__":
    model = create_model()
    visual_model = create_visual_model(model)

    viz = MeshcatVisualizer(model, pin.GeometryModel(), visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    for i in range(100):
        viz.display(2 * np.pi * np.random.random(2))
        time.sleep(0.5)
