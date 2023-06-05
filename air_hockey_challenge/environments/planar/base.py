import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from air_hockey_challenge.environments.data.planar import __file__ as env_path
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box


class AirHockeyBase(MuJoCo):
    """
    Abstract class for all AirHockey Environments.

    """

    def __init__(self, gamma=0.99, horizon=500, timestep=1 / 1000., n_intermediate_steps=20, n_substeps=1,
                 n_agents=1, viewer_params={}):
        """
        Constructor.

        Args:
            n_agents (int, 1): number of agent to be used in the environment (one or two)
        """

        self.n_agents = n_agents

        action_spec = []

        observation_spec = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                            ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                            ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                            ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                            ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                            ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        additional_data = [("puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                           ("puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                           ("puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                           ("puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                           ("puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                           ("puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL)]

        collision_spec = [("puck", ["puck"]),
                          ("rim", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r", "rim_left", "rim_right"]),
                          ("rim_short_sides", ["rim_home_l", "rim_home_r", "rim_away_l", "rim_away_r"])]

        if 1 <= self.n_agents <= 2:
            scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "single.xml")

            action_spec += ["planar_robot_1/joint_1", "planar_robot_1/joint_2", "planar_robot_1/joint_3"]

            observation_spec += [("robot_1/joint_1_pos", "planar_robot_1/joint_1", ObservationType.JOINT_POS),
                                 ("robot_1/joint_2_pos", "planar_robot_1/joint_2", ObservationType.JOINT_POS),
                                 ("robot_1/joint_3_pos", "planar_robot_1/joint_3", ObservationType.JOINT_POS),
                                 ("robot_1/joint_1_vel", "planar_robot_1/joint_1", ObservationType.JOINT_VEL),
                                 ("robot_1/joint_2_vel", "planar_robot_1/joint_2", ObservationType.JOINT_VEL),
                                 ("robot_1/joint_3_vel", "planar_robot_1/joint_3", ObservationType.JOINT_VEL)]

            additional_data += [("robot_1/ee_pos", "planar_robot_1/body_ee", ObservationType.BODY_POS),
                                ("robot_1/ee_vel", "planar_robot_1/body_ee", ObservationType.BODY_VEL)]

            collision_spec += [("robot_1/ee", ["planar_robot_1/ee"])]

            if self.n_agents == 2:
                scene = os.path.join(os.path.dirname(os.path.abspath(env_path)), "double.xml")

                observation_spec += [("robot_1/opponent_ee_pos", "planar_robot_2/body_ee", ObservationType.BODY_POS)]

                action_spec += ["planar_robot_2/joint_1", "planar_robot_2/joint_2", "planar_robot_2/joint_3"]
                # Add puck pos/vel again to transform into second agents frame
                observation_spec += [("robot_2/puck_x_pos", "puck_x", ObservationType.JOINT_POS),
                                     ("robot_2/puck_y_pos", "puck_y", ObservationType.JOINT_POS),
                                     ("robot_2/puck_yaw_pos", "puck_yaw", ObservationType.JOINT_POS),
                                     ("robot_2/puck_x_vel", "puck_x", ObservationType.JOINT_VEL),
                                     ("robot_2/puck_y_vel", "puck_y", ObservationType.JOINT_VEL),
                                     ("robot_2/puck_yaw_vel", "puck_yaw", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_1_pos", "planar_robot_2/joint_1", ObservationType.JOINT_POS),
                                     ("robot_2/joint_2_pos", "planar_robot_2/joint_2", ObservationType.JOINT_POS),
                                     ("robot_2/joint_3_pos", "planar_robot_2/joint_3", ObservationType.JOINT_POS),
                                     ("robot_2/joint_1_vel", "planar_robot_2/joint_1", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_2_vel", "planar_robot_2/joint_2", ObservationType.JOINT_VEL),
                                     ("robot_2/joint_3_vel", "planar_robot_2/joint_3", ObservationType.JOINT_VEL)]

                observation_spec += [("robot_2/opponent_ee_pos", "planar_robot_1/body_ee", ObservationType.BODY_POS)]

                additional_data += [("robot_2/ee_pos", "planar_robot_2/body_ee", ObservationType.BODY_POS),
                                    ("robot_2/ee_vel", "planar_robot_2/body_ee", ObservationType.BODY_VEL)]

                collision_spec += [("robot_2/ee", ["planar_robot_2/ee"])]
        else:
            raise ValueError('n_agents should be 1 or 2')

        self.env_info = dict()
        self.env_info['table'] = {"length": 1.948, "width": 1.038, "goal_width": 0.25}
        self.env_info['puck'] = {"radius": 0.03165}
        self.env_info['mallet'] = {"radius": 0.04815}
        self.env_info['n_agents'] = self.n_agents
        self.env_info['robot'] = {
            "n_joints": 3,
            "ee_desired_height": 0.1,
            "joint_vel_limit": np.array([[-np.pi / 2, -np.pi / 2, -np.pi * 2 / 3],
                                         [np.pi / 2, np.pi / 2, np.pi * 2 / 3]]),

            "joint_acc_limit": np.array([[-2 * np.pi, -2 * np.pi, -2 * 4 / 3 * np.pi],
                                         [2 * np.pi, 2 * np.pi, 2 * 4 / 3 * np.pi]]),
            "base_frame": [],
            "control_frequency": 50,
        }

        self.env_info['puck_pos_ids'] = [0, 1, 2]
        self.env_info['puck_vel_ids'] = [3, 4, 5]
        self.env_info['joint_pos_ids'] = [6, 7, 8]
        self.env_info['joint_vel_ids'] = [9, 10, 11]
        if self.n_agents == 2:
            self.env_info['opponent_ee_ids'] = [13, 14, 15]
        else:
            self.env_info['opponent_ee_ids'] = []

        max_joint_vel = ([np.inf] * 3 + list(self.env_info["robot"]["joint_vel_limit"][1, :3])) * self.n_agents

        super().__init__(scene, action_spec, observation_spec, gamma, horizon, timestep, n_substeps,
                         n_intermediate_steps, additional_data, collision_spec, max_joint_vel, **viewer_params)

        # Construct the mujoco model at origin
        robot_model = mujoco.MjModel.from_xml_path(
            os.path.join(os.path.dirname(os.path.abspath(env_path)), "planar_robot_1.xml"))
        robot_model.body('planar_robot_1/base').pos = np.zeros(3)
        robot_data = mujoco.MjData(robot_model)

        # Add env_info that requires mujoco models
        self.env_info['dt'] = self.dt
        self.env_info["robot"]["joint_pos_limit"] = np.array(
            [self._model.joint(f"planar_robot_1/joint_{i + 1}").range for i in range(3)]).T
        self.env_info["robot"]["robot_model"] = robot_model
        self.env_info["robot"]["robot_data"] = robot_data
        self.env_info["rl_info"] = self.info

        frame_T = np.eye(4)
        temp = np.zeros((9, 1))
        mujoco.mju_quat2Mat(temp, self._model.body("planar_robot_1/base").quat)
        frame_T[:3, :3] = temp.reshape(3, 3)
        frame_T[:3, 3] = self._model.body("planar_robot_1/base").pos
        self.env_info['robot']['base_frame'].append(frame_T.copy())

        if self.n_agents == 2:
            mujoco.mju_quat2Mat(temp, self._model.body("planar_robot_2/base").quat)
            frame_T[:3, :3] = temp.reshape(3, 3)
            frame_T[:3, 3] = self._model.body("planar_robot_2/base").pos
            self.env_info['robot']['base_frame'].append(frame_T.copy())

        # Ids of the joint, which are controller by the action space
        self.actuator_joint_ids = [self._model.joint(name).id for name in action_spec]

    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([0, -1, -np.pi, -20., -20., -100,
                            *np.array([self._model.joint(f"planar_robot_1/joint_{i + 1}").range[0]
                                       for i in range(self.env_info['robot']['n_joints'])]),
                            *self.env_info['robot']['joint_vel_limit'][0]])
        obs_high = np.array([3.02, 1, np.pi, 20., 20., 100,
                             *np.array([self._model.joint(f"planar_robot_1/joint_{i + 1}").range[1]
                                        for i in range(self.env_info['robot']['n_joints'])]),
                             *self.env_info['robot']['joint_vel_limit'][1]])
        if self.n_agents == 2:
            obs_low = np.concatenate([obs_low, [1.5, -1.5, -1.5]])
            obs_high = np.concatenate([obs_high, [4.5, 1.5, 1.5]])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def is_absorbing(self, obs):
        boundary = np.array([self.env_info['table']['length'], self.env_info['table']['width']]) / 2
        puck_pos, puck_vel = self.get_puck(obs)

        if np.any(np.abs(puck_pos[:2]) > boundary) or np.linalg.norm(puck_vel) > 100:
            return True
        return False

    @staticmethod
    def _puck_2d_in_robot_frame(puck_in, robot_frame, type='pose'):
        if type == 'pose':
            puck_w = np.eye(4)
            puck_w[:2, 3] = puck_in[:2]
            puck_w[:3, :3] = R.from_euler("xyz", [0., 0., puck_in[2]]).as_matrix()

            puck_r = np.linalg.inv(robot_frame) @ puck_w
            puck_out = np.concatenate([puck_r[:2, 3],
                                       R.from_matrix(puck_r[:3, :3]).as_euler('xyz')[2:3]])

        if type == 'vel':
            rot_mat = robot_frame[:3, :3]

            vel_lin = np.array([*puck_in[:2], 0])
            vel_ang = np.array([0., 0., puck_in[2]])

            vel_lin_r = rot_mat.T @ vel_lin
            vel_ang_r = rot_mat.T @ vel_ang

            puck_out = np.concatenate([vel_lin_r[:2], vel_ang_r[2:3]])
        return puck_out

    def get_puck(self, obs):
        """
        Getting the puck properties from the observations
        Args:
            obs: The current observation

        Returns:
            ([pos_x, pos_y, yaw], [lin_vel_x, lin_vel_y, yaw_vel])

        """
        puck_pos = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_pos"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_pos")])
        puck_vel = np.concatenate([self.obs_helper.get_from_obs(obs, "puck_x_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_y_vel"),
                                   self.obs_helper.get_from_obs(obs, "puck_yaw_vel")])
        return puck_pos, puck_vel

    def get_ee(self):
        raise NotImplementedError

    def get_joints(self, obs):
        raise NotImplementedError
