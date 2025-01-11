"""
This example shows how to use Pinocchio to manually build a robot model.
It then demonstrates basic capabilities in Pinocchio such as forward kinematics,
Jacobian computation, and collision checking.
"""

import pinocchio
from pinocchio.visualize import MeshcatVisualizer

import coal
import numpy as np
import time


def create_model():
    model = pinocchio.Model()

    max_effort = np.array([100.0])
    max_velocity = np.array([10.0])

    #################
    # Create Link 0 #
    #################
    jid0 = model.addJoint(
        0,  # Parent joint ID -- the universe frame
        pinocchio.JointModelRZ(),  # Revolute about Z
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),  # Joint placement
        "joint0",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi]),  # Min position
        np.array([np.pi]),  # Max position
    )
    # Add a body to the joint
    body0 = model.appendBodyToJoint(
        jid0,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame0 = pinocchio.Frame(
        "joint0",  # name
        jid0,  # Joint parent
        model.getFrameId("universe"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame0, False)

    #################
    # Create Link 1 #
    #################
    jid1 = model.addJoint(
        jid0,  # Parent joint ID
        pinocchio.JointModelRY(),  # Revolute about Y
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),  # Joint placement
        "joint1",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi / 2.0]),  # Min position
        np.array([np.pi / 2.0]),  # Max position
    )
    # Add a body to the joint
    body1 = model.appendBodyToJoint(
        jid1,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.25])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame1 = pinocchio.Frame(
        "joint1",  # name
        jid1,  # Joint parent
        model.getFrameId("joint0"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame1, False)

    #################
    # Create Link 2 #
    #################
    jid2 = model.addJoint(
        jid1,  # Parent joint ID
        pinocchio.JointModelRY(),  # Revolute about Y
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.5])),  # Joint placement
        "joint2",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi / 2.0]),  # Min position
        np.array([np.pi / 2.0]),  # Max position
    )
    # Add a body to the joint
    body2 = model.appendBodyToJoint(
        jid2,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.25])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame2 = pinocchio.Frame(
        "joint2",  # name
        jid2,  # Joint parent
        model.getFrameId("joint1"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame2, False)

    #################
    # Create Link 3 #
    #################
    jid3 = model.addJoint(
        jid2,  # Parent joint ID
        pinocchio.JointModelRZ(),  # Revolute about Z
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.5])),  # Joint placement
        "joint3",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi]),  # Min position
        np.array([np.pi]),  # Max position
    )
    # Add a body to the joint
    body3 = model.appendBodyToJoint(
        jid3,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame3 = pinocchio.Frame(
        "joint3",  # name
        jid3,  # Joint parent
        model.getFrameId("joint2"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame3, False)

    #################
    # Create Link 4 #
    #################
    jid4 = model.addJoint(
        jid3,  # Parent joint ID
        pinocchio.JointModelRX(),  # Revolute about X
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),  # Joint placement
        "joint4",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi / 2.0]),  # Min position
        np.array([np.pi / 2.0]),  # Max position
    )
    # Add a body to the joint
    body4 = model.appendBodyToJoint(
        jid4,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame4 = pinocchio.Frame(
        "joint4",  # name
        jid4,  # Joint parent
        model.getFrameId("joint3"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame4, False)

    #################
    # Create Link 5 #
    #################
    jid5 = model.addJoint(
        jid4,  # Parent joint ID
        pinocchio.JointModelRZ(),  # Revolute about Z
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),  # Joint placement
        "joint5",  # name
        max_effort,
        max_velocity,
        np.array([-np.pi]),  # Min position
        np.array([np.pi]),  # Max position
    )
    # Add a body to the joint
    body5 = model.appendBodyToJoint(
        jid5,  # joint id
        pinocchio.Inertia(
            1.0,  # mass,
            np.array([0.0, 0.0, 0.5]),  # center of mass
            np.diag([0.1, 0.1, 0.1]),  # moments of inertia
        ),
        pinocchio.SE3(
            np.eye(3), np.array([0.0, 0.0, 0.1])
        ),  # transform from joint frame
    )
    # Add a frame to the joint
    frame5 = pinocchio.Frame(
        "joint5",  # name
        jid5,  # Joint parent
        model.getFrameId("joint4"),  # Parent frame
        pinocchio.SE3.Identity(),  # Relative tform
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(frame5, False)

    ##########################
    # Add end effector frame #
    ##########################
    hand_frame = pinocchio.Frame(
        "hand",  # name
        jid5,  # Joint parent
        model.getFrameId("joint5"),  # Parent frame
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),  # Relative tform,
        pinocchio.FrameType.OP_FRAME,  # Frame type
    )
    model.addFrame(hand_frame, False)

    return model


def create_visual_model(model, alpha=0.5):
    visual_model = pinocchio.GeometryModel()

    # Add the links corresponding to the robot joints.
    link0 = pinocchio.GeometryObject(
        "link0",
        model.getJointId("joint0"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),
        coal.Box(0.25, 0.25, 0.2),
    )
    link0.meshColor = np.array([1.0, 0.0, 0.0, alpha])
    visual_model.addGeometryObject(link0)

    link1 = pinocchio.GeometryObject(
        "link1",
        model.getJointId("joint1"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.25])),
        coal.Cylinder(0.075, 0.5),
    )
    link1.meshColor = np.array([0.0, 0.0, 1.0, alpha])
    visual_model.addGeometryObject(link1)

    link2 = pinocchio.GeometryObject(
        "link2",
        model.getJointId("joint2"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.25])),
        coal.Cylinder(0.075, 0.5),
    )
    link2.meshColor = np.array([0.0, 1.0, 0.0, alpha])
    visual_model.addGeometryObject(link2)

    link3 = pinocchio.GeometryObject(
        "link3",
        model.getJointId("joint3"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),
        coal.Sphere(0.1),
    )
    link3.meshColor = np.array([1.0, 0.0, 1.0, alpha])
    visual_model.addGeometryObject(link3)

    link4 = pinocchio.GeometryObject(
        "link4",
        model.getJointId("joint4"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),
        coal.Sphere(0.1),
    )
    link4.meshColor = np.array([0.0, 1.0, 1.0, alpha])
    visual_model.addGeometryObject(link4)

    link5 = pinocchio.GeometryObject(
        "link5",
        model.getJointId("joint5"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.1])),
        coal.Sphere(0.1),
    )
    link5.meshColor = np.array([1.0, 1.0, 0.0, alpha])
    visual_model.addGeometryObject(link5)

    # Add three separate collision bodies at the final joint for the end effector.
    ee_base_link = pinocchio.GeometryObject(
        "ee_base",
        model.getJointId("joint5"),
        pinocchio.SE3(np.eye(3), np.array([0.0, 0.0, 0.2])),
        coal.Box(0.2, 0.1, 0.05),
    )
    ee_base_link.meshColor = np.array([0.6, 0.6, 0.6, alpha])
    visual_model.addGeometryObject(ee_base_link)

    ee_left_link = pinocchio.GeometryObject(
        "ee_left",
        model.getJointId("joint5"),
        pinocchio.SE3(np.eye(3), np.array([0.1, 0.0, 0.275])),
        coal.Box(0.025, 0.1, 0.2),
    )
    ee_left_link.meshColor = np.array([0.6, 0.6, 0.6, alpha])
    visual_model.addGeometryObject(ee_left_link)

    ee_right_link = pinocchio.GeometryObject(
        "ee_right",
        model.getJointId("joint5"),
        pinocchio.SE3(np.eye(3), np.array([-0.1, 0.0, 0.275])),
        coal.Box(0.025, 0.1, 0.2),
    )
    ee_right_link.meshColor = np.array([0.6, 0.6, 0.6, alpha])
    visual_model.addGeometryObject(ee_right_link)

    return visual_model


def setup_collisions(collision_model):
    collision_pairs = [
        ("link0", "link2"),
        ("link0", "link3"),
        ("link0", "link4"),
        ("link0", "link5"),
        ("link0", "ee_base"),
        ("link0", "ee_left"),
        ("link0", "ee_right"),
        ("link1", "link3"),
        ("link1", "link4"),
        ("link1", "link5"),
        ("link1", "ee_base"),
        ("link1", "ee_left"),
        ("link1", "ee_right"),
        ("link2", "link4"),
        ("link2", "link5"),
        ("link2", "ee_base"),
        ("link2", "ee_left"),
        ("link2", "ee_right"),
        ("link3", "ee_base"),
        ("link3", "ee_left"),
        ("link3", "ee_right"),
        ("link4", "ee_base"),
        ("link4", "ee_left"),
        ("link4", "ee_right"),
    ]
    for pair in collision_pairs:
        collision_model.addCollisionPair(
            pinocchio.CollisionPair(
                collision_model.getGeometryId(pair[0]),
                collision_model.getGeometryId(pair[1]),
            )
        )


def print_collisions(collision_model, collision_data):
    num_collisions = 0
    for k in range(len(collision_model.collisionPairs)):
        cr = collision_data.collisionResults[k]
        cp = collision_model.collisionPairs[k]
        if cr.isCollision():
            num_collisions += 1
            print(
                "collision between:",
                collision_model.geometryObjects[cp.first].name,
                " and ",
                collision_model.geometryObjects[cp.second].name,
            )
    if num_collisions == 0:
        print("No collisions found!")


if __name__ == "__main__":
    model = create_model()
    visual_model = create_visual_model(model)
    collision_model = visual_model  # Use the same models in this case

    # Setup collisions
    add_all_collisions = False
    if add_all_collisions:
        collision_model.addAllCollisionPairs()
    else:
        setup_collisions(collision_model)

    # Create model data
    data = model.createData()
    collision_data = collision_model.createData()

    # Set up visualization using MeshCat
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.displayFrames(True)

    for idx in range(20):
        print(f"\nITERATION {idx}\n")
        q = np.random.uniform(model.lowerPositionLimit, model.upperPositionLimit)
        print(f"Joint configuration: {q}\n")

        target_frame_id = model.getFrameId("hand")

        pinocchio.framesForwardKinematics(model, data, q)
        hand_tform = data.oMf[target_frame_id]
        print(f"Hand frame Transform:\n{hand_tform}")

        J = pinocchio.computeFrameJacobian(
            model,
            data,
            q,
            target_frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        print(f"Hand frame Jacobian:\n{J}")

        # Calculate the collision information.
        pinocchio.computeCollisions(
            model, data, collision_model, collision_data, q, False
        )
        print_collisions(collision_model, collision_data)

        viz.display(q)
        time.sleep(1.0)
