.. _constraints:

Constraints
------------

We also provide a util class to compute the constraints. The ``ConstraintList``
is a collection of the ``Constraint`` instances that is available in ``env_info``. Here
is an simple example explaining how to use the constraint functions.

Here is an example of how to access the constraints:

.. literalinclude:: examples/constraints.py

We also list all of the available constraint here:

.. list-table::
   :widths: 20 10 10 50
   :header-rows: 1

   * - Class Name
     - Key
     - Output Dim
     - Description
   * - JointPositionConstraint
     - "joint_pos_constr"
     - 2 * num_joints
     - :math:`q_l < q_{cmd} < q_u`
   * - JointVelocityConstraint
     - "joint_vel_constr"
     - 2 * num_joints
     - :math:`\dot{q}_l < \dot{q}_{cmd} < \dot{q}_u`
   * - EndEffectorConstraint
     - "ee_constr"
     - 5
     - :math:`l_x < x_{ee},`

       :math:`l_y < y_{ee} < u_y,`

       :math:`z_{ee} > \mathrm{table\,height - tolerance}`,

       :math:`z_{ee} < \mathrm{table\, height + tolerance}`.
   * - LinkConstraint (7DoF Robot Only)
     - "link_constr"
     - 2
     - :math:`z_{elbow} > 0.25`,

       :math:`z_{wrist} > 0.25`