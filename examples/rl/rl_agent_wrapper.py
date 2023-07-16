import numpy as np

from air_hockey_challenge.framework import AgentBase
from mushroom_rl.core import Agent


class RlAgent(AgentBase):
    def __init__(self, env_info, double_integration, rl_agent: Agent):
        self.env_info = env_info
        self.rl_agent = rl_agent
        super().__init__(env_info)

        self._double_integration = double_integration
        self._k_pos = 5
        self._k_vel = 10

        self._dt = env_info["dt"]

        self._pos_limit = env_info["robot"]["joint_pos_limit"]
        self._vel_limit = env_info["robot"]["joint_vel_limit"]
        self._acc_limit = env_info["robot"]["joint_acc_limit"]
        self._acc = np.zeros_like(self._acc_limit[0])

        self._add_save_attr(
            _dt='primitive',
            _pos_limit='numpy',
            _vel_limit='numpy',
            _acc_limit='numpy',
            _init_pos='numpy',
            _k_pos='numpy',
            _k_vel='numpy',
            _double_integration='primitive',
            rl_agent='pickle',
            env_info='pickle'
        )

    def draw_action(self, state):
        """
        Args:
            state (ndarray): state after the preprocessing. In reinforcement learning, it's desirable
            to hava normalized state, which is done by preprocessor. The input here is an normalized
            state. We need to denormalize it for double integration.

        Returns:
            numpy.ndarray, (3, num_joints): The desired [Positions, Velocities, Acceleration] of the
            next step. The environment will take first two arguments of the to control the robot.
            The third array is used for the training of the SAC as the output is acceleration. This
            action tuple will be saved in the dataset buffer
        """
        state_orig = state.copy()
        action = self.rl_agent.draw_action(self.preprocess(state))
        pos, vel = self.integrator(action, state_orig)
        return np.vstack([pos, vel, action])

    def integrator(self, action, state_orig):
        """
        We first convert the state to the original state and get actual position and velocity.
        """
        pos = self.get_joint_pos(state_orig)
        vel = self.get_joint_vel(state_orig)

        # Compute the soft limit of the acceleration,
        # details can be found here: http://wiki.ros.org/pr2_controller_manager/safety_limits
        vel_soft_limit = np.clip(-self._k_pos * (pos - self._pos_limit), self._vel_limit[0], self._vel_limit[1])
        acc_soft_limit = np.clip(-self._k_vel * (vel - vel_soft_limit), self._acc_limit[0], self._acc_limit[1])
        if self._double_integration:
            clipped_acc = np.clip(action, *acc_soft_limit)

            coeffs = np.vstack([pos, vel, self._acc / 2, (clipped_acc - self._acc) / 6 / self._dt]).T
            pos = coeffs @ np.array([1, self._dt, self._dt ** 2, self._dt ** 3])
            vel = coeffs @ np.array([0., 1., 2 * self._dt, 3 * self._dt ** 2])
            self._acc = coeffs @ np.array([0., 0., 2, 6 * self._dt])
        else:
            clipped_vel = np.clip(action, *vel_soft_limit)
            pos += clipped_vel * self._dt
            vel = clipped_vel.copy()

        return pos, vel

    def reset(self):
        self._acc = np.zeros_like(self._acc_limit[0])

    def fit(self, dataset, **info):
        """
        Prepare the dataset fot the reinforcement learning.
        1. Normalize the state; 2. Get the original sampled action
        """
        for i in range(len(dataset)):
            temp = list(dataset[i])

            # Preprocess the state and next_state from the dataset
            temp[0] = self.preprocess(temp[0])
            temp[3] = self.preprocess(temp[3])

            # Select original action
            temp[1] = temp[1][2]
            dataset[i] = tuple(temp)
        self.rl_agent.fit(dataset, **info)

    def preprocess(self, state):
        state_norm = state.copy()
        for p in self.rl_agent.preprocessors:
            state_norm = p(state_norm)
        return state_norm
