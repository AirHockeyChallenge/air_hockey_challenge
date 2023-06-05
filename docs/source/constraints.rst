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
     - :math:`l_x < p_x,`

       :math:`l_y < p_y < u_y,`

       :math:`p_z > \mathrm{table\,height - tolerance}`,

       :math:`p_z < \mathrm{table\, height + tolerance}`.
   * - JointVelocityConstraint
     - "joint_vel_constr"
     - 2 * num_joints
     - :math:`q_l < q_{cmd} < q_u`
   * - EndEffectorConstraint
     - "ee_constr"
     - 5
     - :math:`\dot{q}_l < \dot{q}_{cmd} < \dot{q}_u`