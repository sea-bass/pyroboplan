pyroboplan
==========

``pyroboplan`` is an educational Python library for manipulator motion planning.

This library extensively uses the `Pinocchio <https://github.com/stack-of-tasks/pinocchio>`_ Python bindings for modeling robot kinematics and dynamics.

Note the **educational** in this library description.
It is not designed for performance, but rather to teach newcomers the fundamentals of motion planning in an easy-to-use environment with :examples:`many examples <>`.

If you are doing more serious motion planning and control work, check out `RoboPlan <https://github.com/open-planning/roboplan>`_.

.. image:: _static/gifs/pyroboplan_rrt_traj.gif
  :width: 640
  :alt: RRT based motion planning and trajectory execution

.. image:: _static/gifs/pyroboplan_cartesian_path.gif
  :width: 640
  :alt: Cartesian motion planning

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   design
   motion_planning
   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
