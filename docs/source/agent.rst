.. _agent:

Agent
=====

We provide a base class ``AgentBase`` with some utils functions to extract the desired
state from the observation. You can inherit the base class and implement your onw method
in the ``air_hockey_agent/agent_builder.py`` file. A Dummy Agent example can be found
in :ref:`Dummy Agent <dummy_agent>`.

Load and Save Agent
-------------------

We also provide a simple and effective way of save and load your agent. We extend the
``Dummy Agent`` example and set different type of variables. You can add these variables
into saving list by calling ``self.__add_save_attr`` function.

The available methods are:

* **primitive**, to store any primitive type. This includes lists and dictionaries of primitive values.
* **numpy**, to store NumPy arrays.
* **torch**, to store any torch object.
* **pickle**, to store any Python object that cannot be stored with the above methods.
* **json**, can be used if you need a textual output version, that is easy to read.
* **none**, add the attributes, you can assign the values to the attribute later.


.. literalinclude:: examples/save_load_agent_example.py

AgentBase
---------
``air_hockey_challenge.framework.agent_base``

.. autoclass:: air_hockey_challenge.framework.agent_base.AgentBase
    :noindex:

    .. automethod:: __init__
        :noindex:
    .. automethod:: reset
        :noindex:
    .. automethod:: draw_action
        :noindex:
    .. automethod:: get_puck_state
        :noindex:
    .. automethod:: get_joint_state
        :noindex:
    .. automethod:: get_puck_pos
        :noindex:
    .. automethod:: get_puck_vel
        :noindex:
    .. automethod:: get_joint_pos
        :noindex:
    .. automethod:: get_joint_vel
        :noindex:
    .. automethod:: get_ee_pose
        :noindex:
    .. automethod:: save
            :noindex:
    .. automethod:: load_agent
            :noindex:

