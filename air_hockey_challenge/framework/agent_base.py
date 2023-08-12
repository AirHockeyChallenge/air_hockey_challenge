import copy

from air_hockey_challenge.utils.kinematics import forward_kinematics
from mushroom_rl.core import Agent


class AgentBase(Agent):
    def __init__(self, env_info, agent_id=1, **kwargs):
        """
        Initialization of the Agent.

        Args:
            env_info [dict]:
                A dictionary contains information about the environment;
            agent_id [int, default 1]:
                1 by default, agent_id will be used for the tournament;
            kwargs [dict]:
                A dictionary contains agent related information.

        """
        super().__init__(env_info['rl_info'], None)
        self.env_info = env_info
        self.agent_id = agent_id
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])

        self._add_save_attr(
            env_info='none',
            agent_id='none',
            robot_model='none',
            robot_data='none',
        )

    def reset(self):
        """
        Reset the agent

        Important:
            To be implemented

        """
        raise NotImplementedError

    def draw_action(self, observation):
        """ Draw an action, i.e., desired joint position and velocity, at every time step.

        Args:
            observation (ndarray): Observed state including puck's position/velocity, joint position/velocity,
                opponent's end-effector position (if applicable).

        Returns:
            numpy.ndarray, (2, num_joints): The desired [Positions, Velocities] of the next step

        Important:
            To be implemented

        """

        raise NotImplementedError

    def episode_start(self):
        self.reset()

    @classmethod
    def load_agent(cls, path, env_info, agent_id=1):
        """ Load the Agent

        Args:
            path (Path, str): Path to the object
            env_info (dict): A dictionary parsed from the AirHockeyChallengeWrapper
            agent_id (int, default 1): will be specified for two agents game

        Returns:
            Returns the loaded agent

        """
        agent = cls.load(path)

        agent.env_info = env_info
        agent.agent_id = agent_id
        agent.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        agent.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        return agent

    def get_puck_state(self, obs):
        """
        Get the puck's position and velocity from the observation

        Args
        ----
        obs: numpy.ndarray
            observed state.

        Returns
        -------
        joint_pos: numpy.ndarray, (3,)
            [x, y, theta] position of the puck w.r.t robot's base frame
        joint_vel: numpy.ndarray, (3,)
            [vx, vy, dtheta] position of the puck w.r.t robot's base frame

        """
        return self.get_puck_pos(obs), self.get_puck_vel(obs)

    def get_joint_state(self, obs):
        """
        Get the joint positions and velocities from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        joint_pos: numpy.ndarray
            joint positions of the robot;
        joint_vel: numpy.ndarray
            joint velocities of the robot.

        """
        return self.get_joint_pos(obs), self.get_joint_vel(obs)

    def get_puck_pos(self, obs):
        """
        Get the Puck's position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's position of the robot

        """
        return obs[self.env_info['puck_pos_ids']]

    def get_puck_vel(self, obs):
        """
        Get the Puck's velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            Puck's velocity of the robot

        """
        return obs[self.env_info['puck_vel_ids']]

    def get_joint_pos(self, obs):
        """
        Get the joint position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint position of the robot

        """
        return obs[self.env_info['joint_pos_ids']]

    def get_joint_vel(self, obs):
        """
        Get the joint velocity from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            joint velocity of the robot

        """
        return obs[self.env_info['joint_vel_ids']]

    def get_ee_pose(self, obs):
        """
        Get the End-Effector's Position from the observation.

        Args
        ----
        obs: numpy.ndarray
            Agent's observation.

        Returns
        -------
        numpy.ndarray
            opponent's end-effector's position

        """
        return forward_kinematics(self.robot_model, self.robot_data, self.get_joint_pos(obs))
