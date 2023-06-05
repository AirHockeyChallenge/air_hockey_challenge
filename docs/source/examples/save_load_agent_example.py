import numpy as np
import torch

from air_hockey_challenge.framework import AgentBase, AirHockeyChallengeWrapper


def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return DummyAgent(env_info, **kwargs)


class DummyAgent(AgentBase):
    def __init__(self, env_info, value, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.primitive_variable = value  # Primitive python variable
        self.numpy_vector = np.array([1, 2, 3]) * value  # Numpy array
        self.list_variable = [1, 'list', [2, 3]]  # Numpy array

        # Dictionary
        self.dictionary = dict(some='random', keywords=2, fill='the dictionary')

        # Building a torch object
        data_array = np.ones(3) * value
        data_tensor = torch.from_numpy(data_array)
        self.torch_object = torch.nn.Parameter(data_tensor)

        # A non serializable object
        self.object_instance = object()

        # A variable that is not important e.g. a buffer
        self.not_important = np.zeros(10000)

        # Here we specify how to save each component
        self._add_save_attr(
            primitive_variable='primitive',
            numpy_vector='numpy',
            list_variable='primitive',
            dictionary='pickle',
            torch_object='torch',
            object_instance='none',
            # The '!' is to specify that we save the variable only if full_save is True
            not_important='numpy!',
        )

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

        velocity = np.zeros_like(self.hold_position)
        action = np.vstack([self.hold_position, velocity])
        return action


if __name__ == '__main__':
    env = AirHockeyChallengeWrapper("3dof-hit")

    # Construct Agent
    args = {'value': 1.1}
    agent_save = build_agent(env.env_info, **args)

    print("######################################################")
    print("Save Agent Variables")
    print("######################################################")
    print("agent_save.primitive_variable: ", agent_save.primitive_variable)
    print("agent_save.numpy_vector: ", agent_save.numpy_vector)
    print("agent_save.list_variable: ", agent_save.list_variable)
    print("agent_save.dictionary: ", agent_save.dictionary)
    print("agent_save.torch_object: ", agent_save.torch_object)

    # The not_important variable will not be saved unless the full_save is set True
    agent_save.save("agent.msh", full_save=False)

    agent_load = DummyAgent.load_agent("agent.msh", env.env_info)
    print("######################################################")
    print("Load the Agent")
    print("######################################################")
    print("agent_load.primitive_variable: ", agent_load.primitive_variable)
    print("agent_load.numpy_vector: ", agent_load.numpy_vector)
    print("agent_load.list_variable: ", agent_load.list_variable)
    print("agent_load.dictionary: ", agent_load.dictionary)
    print("agent_load.torch_object: ", agent_load.torch_object)
    print("agent_load.object_instance: ", agent_load.object_instance)

    print("------------------------------------------------------")
    print("These variable will not be saved while full_save is False")
    print("agent_load.not_important: ", agent_load.not_important)

    print("------------------------------------------------------")
    print("These variable will be parsed from env_info:")
    print("agent_load.env_info.keys()s: ", agent_load.env_info.keys())
    print("agent_load.agent_id: ", agent_load.agent_id)
    print("agent_load.robot_model: ", agent_load.robot_model)
    print("agent_load.robot_data: ", agent_load.robot_data)
