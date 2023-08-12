import numpy as np

from air_hockey_challenge.utils.kinematics import forward_kinematics


class UniversalJointPlugin:
    def __init__(self, env_model, env_data, env_info):
        self.env_info = env_info
        self.env_model = env_model
        self.env_data = env_data
        self.Kp = 20
        self.Kd = 0.31

        self.universal_joint_ids = []
        self.universal_joint_ctrl_ids = []
        self.universal_joint_ids += [env_model.joint("iiwa_1/striker_joint_1").id,
                                     env_model.joint("iiwa_1/striker_joint_2").id]
        self.universal_joint_ctrl_ids += [env_model.actuator("iiwa_1/striker_joint_1").id,
                                          env_model.actuator("iiwa_1/striker_joint_2").id]
        action_spec = ["iiwa_1/joint_1", "iiwa_1/joint_2", "iiwa_1/joint_3", "iiwa_1/joint_4", "iiwa_1/joint_5",
                       "iiwa_1/joint_6", "iiwa_1/joint_7"]

        if self.env_info['n_agents'] >= 2:
            self.universal_joint_ids += [env_model.joint("iiwa_2/striker_joint_1").id,
                                         env_model.joint("iiwa_2/striker_joint_2").id]
            self.universal_joint_ctrl_ids += [env_model.actuator("iiwa_2/striker_joint_1").id,
                                              env_model.actuator("iiwa_2/striker_joint_2").id]
            action_spec += ["iiwa_2/joint_1", "iiwa_2/joint_2", "iiwa_2/joint_3", "iiwa_2/joint_4", "iiwa_2/joint_5",
                            "iiwa_2/joint_6", "iiwa_2/joint_7"]

        self.actuator_joint_ids = [self.env_model.joint(name).id for name in action_spec]

        self.filter_ratio = 0.273
        self.u_joint_pos_des = np.zeros(2 * self.env_info['n_agents'])
        self.u_joint_pos_prev = None
        self.u_joint_vel_prev = np.zeros(2 * self.env_info['n_agents'])

    def reset(self):
        self.u_joint_pos_prev = None
        self._control_universal_joint()
        for i in range(self.env_info['n_agents']):
            self.u_joint_vel_prev = self.env_data.qvel[self.universal_joint_ids]

        self.env_data.qpos[self.universal_joint_ctrl_ids] = self.u_joint_pos_des

    def update(self):
        self._control_universal_joint()

    def _control_universal_joint(self):
        self._compute_universal_joint()
        self.u_joint_pos_prev = self.env_data.qpos[self.universal_joint_ids]
        self.u_joint_vel_prev = self.filter_ratio * self.env_data.qvel[self.universal_joint_ids] + \
                                (1 - self.filter_ratio) * self.u_joint_vel_prev

        Kp = 4
        Kd = 0.31
        torque = Kp * (self.u_joint_pos_des - self.u_joint_pos_prev) - Kd * self.u_joint_vel_prev
        self.env_data.ctrl[self.universal_joint_ctrl_ids] = torque

    def _compute_universal_joint(self):
        for i in range(self.env_info['n_agents']):
            q = self.env_data.qpos[self.actuator_joint_ids[i * 7: (i + 1) * 7]]
            # Have to exclude the puck joints
            pos, rot_mat = forward_kinematics(self.env_info['robot']['robot_model'],
                                              self.env_info['robot']['robot_data'], q)

            v_x = rot_mat[:, 0]
            v_y = rot_mat[:, 1]

            # The desired position of the x-axis is the cross product of the desired z (0, 0, 1).T
            # and the current y-axis. (0, 0, 1).T x v_y
            x_desired = np.array([-v_y[1], v_y[0], 0])

            # Find the signed angle from the current to the desired x-axis around the y-axis
            # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
            q1 = np.arctan2(self._cross_3d(v_x, x_desired) @ v_y, v_x @ x_desired)
            if self.u_joint_pos_prev is not None:
                if q1 - self.u_joint_pos_prev[0] > np.pi:
                    q1 -= np.pi * 2
                elif q1 - self.u_joint_pos_prev[0] < -np.pi:
                    q1 += np.pi * 2

            # Rotate the X axis by the calculated amount
            w = np.array([[0, -v_y[2], v_y[1]],
                          [v_y[2], 0, -v_y[0]],
                          [-v_y[1], v_y[0], 0]])

            r = np.eye(3) + w * np.sin(q1) + w ** 2 * (1 - np.cos(q1))
            v_x_rotated = r @ v_x

            # The desired position of the y-axis is the negative cross product of the desired z (0, 0, 1).T and the current
            # x-axis, which is already rotated around y. The negative is there because the x-axis is flipped.
            # -((0, 0, 1).T x v_x))
            y_desired = np.array([v_x_rotated[1], - v_x_rotated[0], 0])

            # Find the signed angle from the current to the desired y-axis around the new rotated x-axis
            q2 = np.arctan2(self._cross_3d(v_y, y_desired) @ v_x_rotated, v_y @ y_desired)

            if self.u_joint_pos_prev is not None:
                if q2 - self.u_joint_pos_prev[1] > np.pi:
                    q2 -= np.pi * 2
                elif q2 - self.u_joint_pos_prev[1] < -np.pi:
                    q2 += np.pi * 2

            alpha_y = np.minimum(np.maximum(q1, -np.pi / 2 * 0.95), np.pi / 2 * 0.95)
            alpha_x = np.minimum(np.maximum(q2, -np.pi / 2 * 0.95), np.pi / 2 * 0.95)

            if self.u_joint_pos_prev is None:
                self.u_joint_pos_des[i * 2: i * 2 + 2] = np.array([alpha_y, alpha_x])
            else:
                self.u_joint_pos_des[i * 2: i * 2 + 2] += np.minimum(np.maximum(
                    10 * (np.array([alpha_y, alpha_x]) - self.u_joint_pos_des[i * 2: i * 2 + 2]),
                    -np.pi * 0.01), np.pi * 0.01)

            self.u_joint_pos_des[i * 2: i * 2 + 2] = np.array([alpha_y, alpha_x])
            
        return self.u_joint_pos_des

    def _cross_3d(self, a, b):
        return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])
