API
===

ChallengeCore
-------------
``air_hockey_challenge.framework.challenge_core``

.. autoclass:: air_hockey_challenge.framework.ChallengeCore
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: reset, learn


AgentBase
---------
``air_hockey_challenge.framework.agent_base``

.. autoclass:: air_hockey_challenge.framework.agent_base.AgentBase
    :members:
    :inherited-members:
    :undoc-members:
    :show-inheritance:
    :exclude-members: fit, set_logger, reset, draw_action

    .. automethod:: reset
    .. automethod:: draw_action


AirHockeyChallengeWrapper
-------------------------
``air_hockey_challenge.framework.AirHockeyChallengeWrapper``

.. autoclass:: air_hockey_challenge.framework.AirHockeyChallengeWrapper
    :members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: info, make


Constraints
-----------
``air_hockey_challenge.constraints``

.. autoclass:: air_hockey_challenge.constraints.ConstraintList

    .. automethod:: keys
    .. automethod:: get
    .. automethod:: add
    .. automethod:: delete
    .. automethod:: fun
    .. automethod:: jacobian

.. autoclass:: air_hockey_challenge.constraints.Constraint
    :undoc-members:

    .. automethod:: __init__
    .. automethod:: fun
    .. automethod:: jacobian
    .. autoattribute:: name

.. autoclass:: air_hockey_challenge.constraints.JointPositionConstraint
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: air_hockey_challenge.constraints.JointVelocityConstraint
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: air_hockey_challenge.constraints.EndEffectorConstraint
    :members:
    :undoc-members:
    :show-inheritance:


Utils
-----

Kinematics
~~~~~~~~~~

``air_hockey_challenge.utils.kinematics``

.. automodule:: air_hockey_challenge.utils.kinematics
    :members:

Transformation
~~~~~~~~~~~~~~

``air_hockey_challenge.utils.transformations``

.. automodule:: air_hockey_challenge.utils.transformations
    :members:


