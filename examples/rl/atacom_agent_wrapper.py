import json

import numpy as np

from air_hockey_challenge.framework import AgentBase
from mushroom_rl.core import Agent
from mushroom_rl.utils.preprocessors import Serializable
from examples.rl.air_hockey_contraints import EndEffectorConstraint, EndEffectorPosConstraint, JointPosConstraint
from examples.rl.atacom import ATACOMController, ConstraintList
from examples.rl.atacom.system import AccelerationControlSystem, VelocityControlSystem


class ATACOMAgent(AgentBase):
    def __init__(self, env_info, double_integration, rl_agent: Agent, atacom_controller: ATACOMController):
        self.env_info = env_info
        self.rl_agent = rl_agent
        self.atacom_controller = atacom_controller
        super().__init__(env_info)

        self._double_integration = double_integration
        self._k_pos = 5
        self._k_vel = 10

        self._dt = env_info["dt"]

        self._pos_limit = env_info["robot"]["joint_pos_limit"] * 0.9
        self._vel_limit = env_info["robot"]["joint_vel_limit"] * 0.9
        self._acc_limit = env_info["robot"]["joint_acc_limit"] * 0.9
        self._acc = np.zeros_like(self._acc_limit[0])
        self.prev_action = np.zeros_like(self._vel_limit[0])
        self._filter_ratio = np.max(self._acc_limit[1] / self._vel_limit[1] * self._dt) / 2

        self.config = {
            'slack_beta': self.atacom_controller.slack.beta,
            'slack_type': self.atacom_controller.slack.dynamics_type,
            'slack_tol': self.atacom_controller.slack.tol,
        }

        self.add_preprocessor(AppendActionHistPrePro(self._vel_limit[0].size))

        self._add_save_attr(
            _dt='primitive',
            _pos_limit='numpy',
            _vel_limit='numpy',
            _acc_limit='numpy',
            _k_pos='numpy',
            _k_vel='numpy',
            _double_integration='primitive',
            _filter_ratio='primitive',
            rl_agent='pickle',
            config='primitive',
            atacom_controller='none',
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
        
        # Draw action from reinforcement learning
        action = self.rl_agent.draw_action(self.preprocess(state))
        self.prev_action = (1 - self._filter_ratio) * self.prev_action + self._filter_ratio * action
        self.preprocessors[0].update_action_hist(self.prev_action)
	
	# Atacom controller transform the sample action to the tangent space
        s = np.concatenate(self.get_joint_state(state_orig))
        normalized_vel = self.atacom_controller.compose_action(s, self.prev_action, x_dot=0.)
        dq = self._vel_limit[1] * normalized_vel
	
	# Integrate the velocity
        pos, vel = self.integrator(dq, state_orig)
        
        # The position and velocity is used to inteact with the environment
        # The `action` is used for reinforcement learning agent. The action is unpacked in the 
        # fit() function
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
        self.prev_action = np.zeros_like(self._vel_limit[0])
        self.preprocessors[0].update_action_hist(self.prev_action)

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

            # Unpack the action and use only the sampled action for training
            temp[1] = temp[1][2]
            dataset[i] = tuple(temp)
        self.rl_agent.fit(dataset, **info)

    def preprocess(self, state):
        state_norm = state.copy()
        for p in self.rl_agent.preprocessors:
            state_norm = p(state_norm)
        return state_norm

    @classmethod
    def load_agent(cls, path, env_info, agent_id=1):
        agent = super().load_agent(path, env_info, agent_id)

        agent.atacom_controller = build_ATACOM_Controller(env_info, agent.config['slack_type'],
                                                          agent.config['slack_beta'], agent.config['slack_tol'],
                                                          double_integration=agent._double_integration)
        return agent


def build_ATACOM_Controller(env_info, slack_type, slack_beta, slack_tol, **kwargs):
    lambda_c = 1 / env_info['dt']
    dim_q = env_info['robot']['n_joints']
    kwargs['double_integration'] = False

    if kwargs['double_integration']:
        # The state of second-order system is the stacked state q_hat = [q, dq]
        constraints = ConstraintList(dim_q * 2)
        # constraints.add_constraint(JointPosVelConstraint(env_info))
        constraints.add_constraint(EndEffectorConstraint(env_info))
        system = AccelerationControlSystem(dim_q, env_info['robot']['joint_acc_limit'][1])
    else:
        constraints = ConstraintList(dim_q)
        constraints.add_constraint(JointPosConstraint(env_info))
        constraints.add_constraint(EndEffectorPosConstraint(env_info))
        system = VelocityControlSystem(dim_q, env_info['robot']['joint_vel_limit'][1] * 0.95)
    return ATACOMController(constraints, system, slack_beta=slack_beta,
                            slack_dynamics_type=slack_type, slack_tol=slack_tol, lambda_c=lambda_c)


class AppendActionHistPrePro(Serializable):
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.prev_action = np.zeros(action_dim)
        self._add_save_attr(
            action_dim='primitive'
        )

    def update_action_hist(self, prev_action):
        self.prev_action = prev_action.copy()

    def __call__(self, obs):
        return np.concatenate([obs, self.prev_action])
