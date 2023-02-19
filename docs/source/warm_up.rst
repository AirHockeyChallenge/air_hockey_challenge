.. _warm_up:

Warm Up
=======

In this stage, you will handle two tasks with a simplified 3-DoF robot. The robot is set
such that the end-effector will on the height of the table surface. In simulation, the
collision between the robot and the table are disabled.

Tasks
-----
.. list-table:: Tasks in Warm Up Stage
   :widths: 50 50
   :header-rows: 0
   :align: center

   * - .. image:: ../assets/3dof-hit-static.gif
     - .. image:: ../assets/3dof-defend.gif
   * - Hit, ``3dof-hit``
     - Defend, ``3dof-defend``

Environment Specifications
--------------------------
Here we list the some useful information about the environment.

.. important::
    In the constraint, the joint position and velocity limits for constraint computation
    is 95% of the actual limits. For example, the upper bound of the position limit for
    joint 1 is 2.967. In the ``Evaluation`` and ``Constraints``, we check if the joint
    position exceeds 2.967 * 0.95 = 2.818707.

+----------------------------------------------------------------------------------------+
| **Robot Specifications**                                                               |
+-----------------------------------------+----------------------------------------------+
| Robot Position Upper Limit (rad)        | [ 2.967060,  1.8,  2.094395]                 |
+-----------------------------------------+----------------------------------------------+
| Robot Position Lower Limit (rad)        | [-2.967060, -1.8, -2.094395]                 |
+-----------------------------------------+----------------------------------------------+
| Robot Velocity Limit (rad/s)            | +/- [1.570796,  1.570796,  2.094395]         |
+-----------------------------------------+----------------------------------------------+
| Robot Link Length (m)                   | [0.55, 0.44, 0.44]                           |
+-----------------------------------------+----------------------------------------------+
| **Environment Specifications**                                                         |
+-----------------------------------------+----------------------------------------------+
| Environments                            | ``3dof-hit``, ``3dof-defend``                |
+-----------------------------------------+----------------------------------------------+
| Initial Robot Position (fixed)          |        [-1.156,  1.300,  1.443]              |
+-----------------------------------------+----------------------------------------------+
| Initial Robot Velocity                  | 0                                            |
+-----------------------------------------+----------------------------------------------+
| Range of Puck's Initial Position [x, y] | **Hit**:                                     |
|                                         |     [[0.81, 1.31], [-0.39, 0.39]]            |
| (robot's base frame)                    |                                              |
|                                         | **Defend**:                                  |
|                                         |     [[1.80, 2.16], [-0.4, 0.4]]              |
+-----------------------------------------+----------------------------------------------+
| Range of Puck's Initial Velocity        | **Hit**:                                     |
|                                         |     0                                        |
|                                         |                                              |
|                                         | **Defend**:                                  |
|                                         |     speed (m/s): [1, 3]                      |
|                                         |                                              |
|                                         |     angle: [-0.5,0.5] + :math:`\pi`          |
+-----------------------------------------+----------------------------------------------+
| Termination Criterion                   | **Hit**:                                     |
|                                         |     puck is moving back or score             |
|                                         |                                              |
|                                         | **Defend**:                                  |
|                                         |     puck return to the opponent's side       |
|                                         |     or score                                 |
+-----------------------------------------+----------------------------------------------+

Evaluation
----------

To evaluate your agent in the cloud server, please follow the :ref:`submission` instruction.
In the warm up stage, the environment for evaluation is the same as the public available one.
