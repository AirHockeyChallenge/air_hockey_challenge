.. _constraints:

Constraints
------------

We also provide a util class to compute the constraints. The ``ConstraintList``
is a collection of the ``Constraint`` instances that is available in ``env_info``. Here
is an simple example explaining how to use the constraint functions.

The constraints has the form:
:math:`f(q) < 0`

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
     - :math:`q_i - q_{u, i} < 0`,

       :math:`-q_i + q_{l, i} < 0`
   * - JointVelocityConstraint
     - "joint_vel_constr"
     - 2 * num_joints
     - :math:`\dot{q}_i - \dot{q}_{u, i} < 0`,

       :math:`-\dot{q}_i + \dot{q}_{l, i} < 0`
   * - EndEffectorConstraint
     - "ee_constr"
     - 5
     - :math:`- p_x(q) + l_x < 0`,

       :math:`- p_y(q) + l_y < 0`,

       :math:`p_y(q) - u_y < 0`,

       :math:`- p_z(q) + \mathrm{table\,height - tolerance} < 0`,

       :math:`p_z(q) - \mathrm{table\,height - tolerance} < 0`,