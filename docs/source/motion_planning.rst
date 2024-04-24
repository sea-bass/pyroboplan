Motion Planning
===============

Motion planning is a broad area of robotics, and often consists of several components.

Basic Kinematics
^^^^^^^^^^^^^^^^

Pinocchio offers all the tools necessary to perform basic forward kinematics and get a frame Jacobian.


Inverse Kinematics
^^^^^^^^^^^^^^^^^^

Inverse kinematics (IK) is less straightforward.

* If you have a relatively simple robot, you can analytically solve for a solution.
* For most robotics applications, we require numerical optimization methods instead.

Currently, the only IK implementation is **Differential IK**.


Free-Space Path Planning
^^^^^^^^^^^^^^^^^^^^^^^^

TODO


Cartesian Path Planning
^^^^^^^^^^^^^^^^^^^^^^^

TODO


Trajectory Generation
^^^^^^^^^^^^^^^^^^^^^

TODO
