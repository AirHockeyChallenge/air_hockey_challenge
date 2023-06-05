import mujoco
import numpy as np

from air_hockey_challenge.environments.planar import AirHockeyBase


class AirHockeySingle(AirHockeyBase):
    """
    Base class for single agent air hockey tasks.

    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        """
        Constructor.

        """
        self.init_state = np.array([-1.15570723, 1.30024401, 1.44280414])
        super().__init__(gamma=gamma, horizon=horizon, n_agents=1, viewer_params=viewer_params)

        self.filter_ratio = 0.274
        self.q_pos_prev = np.zeros(self.env_info["robot"]["n_joints"])
        self.q_vel_prev = np.zeros(self.env_info["robot"]["n_joints"])

    def get_ee(self):
        """
        Getting the ee properties from the current internal state. Can also be obtained via forward kinematics
        on the current joint position, this function exists to avoid redundant computations.

        Returns:
            ([pos_x, pos_y, pos_z], [ang_vel_x, ang_vel_y, ang_vel_z, lin_vel_x, lin_vel_y, lin_vel_z])
        """
        ee_pos = self._read_data("robot_1/ee_pos")

        ee_vel = self._read_data("robot_1/ee_vel")

        return ee_pos, ee_vel

    def get_joints(self, obs):
        """
        Get joint position and velocity of the robot
        """
        q_pos = np.zeros(3)
        q_vel = np.zeros(3)
        for i in range(3):
            q_pos[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_pos")[0]
            q_vel[i] = self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_vel")[0]

        return q_pos, q_vel

    def _modify_observation(self, obs):
        new_obs = obs.copy()
        puck_pos, puck_vel = self.get_puck(obs)

        puck_pos = self._puck_2d_in_robot_frame(puck_pos, self.env_info['robot']['base_frame'][0])

        puck_vel = self._puck_2d_in_robot_frame(puck_vel, self.env_info['robot']['base_frame'][0], type='vel')

        self.obs_helper.get_from_obs(new_obs, "puck_x_pos")[:] = puck_pos[0]
        self.obs_helper.get_from_obs(new_obs, "puck_y_pos")[:] = puck_pos[1]
        self.obs_helper.get_from_obs(new_obs, "puck_yaw_pos")[:] = puck_pos[2]

        self.obs_helper.get_from_obs(new_obs, "puck_x_vel")[:] = puck_vel[0]
        self.obs_helper.get_from_obs(new_obs, "puck_y_vel")[:] = puck_vel[1]
        self.obs_helper.get_from_obs(new_obs, "puck_yaw_vel")[:] = puck_vel[2]

        return new_obs

    def setup(self, state=None):
        for i in range(3):
            self._data.joint("planar_robot_1/joint_" + str(i + 1)).qpos = self.init_state[i]
            self.q_pos_prev[i] = self.init_state[i]
            self.q_vel_prev[i] = self._data.joint("planar_robot_1/joint_" + str(i + 1)).qvel

        mujoco.mj_fwdPosition(self._model, self._data)
        super().setup(state)

    def _create_observation(self, obs):
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter

        for i in range(3):
            self.obs_helper.get_from_obs(obs, "robot_1/joint_" + str(i + 1) + "_vel")[:] = q_vel_filter[i]

        yaw_angle = self.obs_helper.get_from_obs(obs, "puck_yaw_pos")
        self.obs_helper.get_from_obs(obs, "puck_yaw_pos")[:] = (yaw_angle + np.pi) % (2 * np.pi) - np.pi
        return obs
