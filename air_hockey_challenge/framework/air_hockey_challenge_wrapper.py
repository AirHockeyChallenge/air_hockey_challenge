from air_hockey_challenge.environments import position_control_wrapper as position
from air_hockey_challenge.constraints import *
from air_hockey_challenge.utils import robot_to_world

from mushroom_rl.core import Environment
from copy import deepcopy


class AirHockeyChallengeWrapper(Environment):
    def __init__(self, env, action_type="position-velocity", interpolation_order=3,
                 custom_reward_function=None, **kwargs):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            action_type [string, default "position-velocity"]:
                The action type of the environment. **Do not change this value**
            interpolation_order [int, default 3]:
                The order of the polynomial interpolator. **Do not change this value**
            custom_reward_function [callable]:
                You can customize your reward function here.

        """

        env_dict = {
            "3dof-hit": (position.PlanarPositionHit, {}),
            "3dof-defend": (position.PlanarPositionDefend, {}),
        }

        self.base_env = env_dict[env][0](action_type=action_type, interpolation_order=interpolation_order,
                                         **env_dict[env][1], **kwargs)
        self.env_name = env
        self.env_info = self.base_env.env_info

        if custom_reward_function:
            self.base_env.reward = lambda state, action, next_state, absorbing: custom_reward_function(self.base_env,
                                                                                                       state, action,
                                                                                                       next_state,
                                                                                                       absorbing)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))

        self.env_info['constraints'] = constraint_list
        self.env_info['env_name'] = self.env_name

        super().__init__(self.base_env.info)

    def step(self, action):
        obs, reward, done, info = self.base_env.step(action)

        if "opponent" in self.env_name:
            action = self.base_env.action[:, :self.env_info['robot']["n_joints"]]
        else:
            action = self.base_env.action

        if self.base_env.n_agents == 1:
            info["constraints_value"] = deepcopy(self.env_info['constraints'].fun(obs[self.env_info['joint_pos_ids']],
                                                                                  obs[self.env_info['joint_vel_ids']]))
            info["jerk"] = self.base_env.jerk
            info["success"] = self.check_success(obs)

        if "competition" in self.env_name:
            info["constraints_value"] = list()
            info["jerk"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs)/2): (i+1) * int(len(obs)/2)]
                info["constraints_value"].append(deepcopy(self.env_info['constraints'].fun(
                    obs_agent[self.env_info['joint_pos_ids']], obs_agent[self.env_info['joint_vel_ids']])))
                info["jerk"].append(
                    self.base_env.jerk[i * self.env_info['robot']['n_joints']:(i + 1) * self.env_info['robot'][
                        'n_joints']])

        return obs, reward, done, info

    def render(self):
        self.base_env.render()

    def reset(self, state=None):
        return self.base_env.reset(state)

    def check_success(self, obs):
        puck_pos, puck_vel = self.base_env.get_puck(obs)

        puck_pos, _ = robot_to_world(self.base_env.env_info["robot"]["base_frame"][0], translation=puck_pos)
        success = 0

        if "hit" in self.env_name:
            if puck_pos[0] - self.base_env.env_info['table']['length'] / 2 > 0 and \
                    np.abs(puck_pos[1]) - self.base_env.env_info['table']['goal_width'] / 2 < 0:
                success = 1

        elif "defend" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.29 and puck_vel[0] < 0.1:
                success = 1

        elif "prepare" in self.env_name:
            if -0.8 < puck_pos[0] <= -0.29 and puck_vel[0] < 0.1:
                success = 1
        return success


if __name__ == "__main__":
    env = AirHockeyChallengeWrapper(env="3dof-hit")
    env.reset()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.random.uniform(-1, 1, (2, env.env_info['robot']['n_joints'])) * 3
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
