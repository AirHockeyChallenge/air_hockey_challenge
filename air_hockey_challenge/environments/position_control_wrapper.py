import mujoco
import numpy as np
import scipy
from collections import deque

from air_hockey_challenge.environments import planar as planar

class PositionControl:
    def __init__(self, p_gain, d_gain, i_gain, action_type="position-velocity", interpolation_order=3,
                 debug=False, *args, **kwargs):
        self.action_type = action_type
        self.debug = debug

        super(PositionControl, self).__init__(*args, **kwargs)

        self.robot_model = self.env_info['robot']['robot_model']
        self.robot_data = self.env_info['robot']['robot_data']

        self.p_gain = np.array(p_gain * self.n_agents)
        self.d_gain = np.array(d_gain * self.n_agents)
        self.i_gain = np.array(i_gain * self.n_agents)

        self.prev_pos = np.zeros(len(self.actuator_joint_ids))
        self.prev_vel = np.zeros(len(self.actuator_joint_ids))
        self.prev_acc = np.zeros(len(self.actuator_joint_ids))
        self.i_error = np.zeros(len(self.actuator_joint_ids))
        self.prev_controller_cmd_pos = np.zeros(len(self.actuator_joint_ids))

        self.interp_order = interpolation_order
        self.action = None

        self._num_coeffs = self.interp_order + 1
        self._num_env_joints = len(self.actuator_joint_ids)
        self.n_robot_joints = self.env_info['robot']["n_joints"]

        self.traj = None
        self.jerk = np.zeros(self._num_env_joints)

        if self.debug:
            self.controller_record = deque(maxlen=self.info.horizon * self._n_intermediate_steps)

    def _enforce_safety_limits(self, desired_pos, desired_vel):
        # ROS safe controller
        pos = self.prev_controller_cmd_pos
        k = 20

        joint_pos_lim = np.tile(self.env_info['robot']['joint_pos_limit'], (1, self.n_agents))
        joint_vel_lim = np.tile(self.env_info['robot']['joint_vel_limit'], (1, self.n_agents))

        min_vel = np.clip(-k * (pos - joint_pos_lim[0]), joint_vel_lim[0], joint_vel_lim[1])

        max_vel = np.clip(-k * (pos - joint_pos_lim[1]), joint_vel_lim[0], joint_vel_lim[1])

        clipped_vel = np.clip(desired_vel, min_vel, max_vel)

        min_pos = pos + min_vel * self._timestep
        max_pos = pos + max_vel * self._timestep

        clipped_pos = np.clip(desired_pos, min_pos, max_pos)
        self.prev_controller_cmd_pos = clipped_pos.copy()

        return clipped_pos, clipped_vel

    def _controller(self, desired_pos, desired_vel, desired_acc, current_pos, current_vel):
        clipped_pos, clipped_vel = self._enforce_safety_limits(desired_pos, desired_vel)

        error = (clipped_pos - current_pos)

        self.i_error += self.i_gain * error * self._timestep
        torque = self.p_gain * error + self.d_gain * (clipped_vel - current_vel) + self.i_error

        # Acceleration FeedForward
        tau_ff = np.zeros(self.robot_model.nv)
        for i in range(self.n_agents):
            robot_joint_ids = np.arange(self.n_robot_joints) + self.n_robot_joints * i
            self.robot_data.qpos = current_pos[robot_joint_ids]
            self.robot_data.qvel = current_vel[robot_joint_ids]
            acc_ff = desired_acc[robot_joint_ids]
            mujoco.mj_forward(self.robot_model, self.robot_data)

            mujoco.mj_mulM(self.robot_model, self.robot_data, tau_ff, acc_ff)
            torque[robot_joint_ids] += tau_ff

            # Gravity Compensation and Coriolis and Centrifugal force
            torque[robot_joint_ids] += self.robot_data.qfrc_bias

            torque[robot_joint_ids] = np.clip(torque[robot_joint_ids],
                                              self.robot_model.actuator_ctrlrange[:, 0],
                                              self.robot_model.actuator_ctrlrange[:, 1])

        if self.debug:
            self.controller_record.append(
                np.concatenate([desired_pos, current_pos, desired_vel, current_vel, desired_acc, self.jerk]))

        return torque

    def _interpolate_trajectory(self, action):
        tf = self.dt
        if self.interp_order == 1 and action.ndim == 1:
            coef = np.array([[1, 0], [1, tf]])
            results = np.vstack([self.prev_pos, action])
        elif self.interp_order == 2 and action.ndim == 1:
            coef = np.array([[1, 0, 0], [1, tf, tf ** 2], [0, 1, 0]])
            if np.linalg.norm(action - self.prev_pos) < 1e-3:
                self.prev_vel = np.zeros_like(self.prev_vel)
            results = np.vstack([self.prev_pos, action, self.prev_vel])
        elif self.interp_order == 3 and action.shape[0] == 2:
            coef = np.array([[1, 0, 0, 0], [1, tf, tf ** 2, tf ** 3], [0, 1, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2]])
            results = np.vstack([self.prev_pos, action[0], self.prev_vel, action[1]])
        elif self.interp_order == 4 and action.shape[0] == 2:
            coef = np.array([[1, 0, 0, 0, 0], [1, tf, tf ** 2, tf ** 3, tf ** 4],
                             [0, 1, 0, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3],
                             [0, 0, 2, 0, 0]])
            results = np.vstack([self.prev_pos, action[0], self.prev_vel, action[1], self.prev_acc])
        elif self.interp_order == 5 and action.shape[0] == 3:
            coef = np.array([[1, 0, 0, 0, 0, 0], [1, tf, tf ** 2, tf ** 3, tf ** 4, tf ** 5],
                             [0, 1, 0, 0, 0, 0], [0, 1, 2 * tf, 3 * tf ** 2, 4 * tf ** 3, 5 * tf ** 4],
                             [0, 0, 2, 0, 0, 0], [0, 0, 2, 6 * tf, 12 * tf ** 2, 20 * tf ** 3]])
            results = np.vstack([self.prev_pos, action[0], self.prev_vel, action[1], self.prev_acc, action[2]])
        else:
            raise ValueError("Undefined interpolator order or the action dimension does not match!")

        A = scipy.linalg.block_diag(*[coef] * self._num_env_joints)
        y = results.reshape(-1, order='F')

        weights = np.linalg.solve(A, y).reshape(self._num_env_joints, self._num_coeffs)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        if self.interp_order in [3, 4, 5]:
            self.jerk = np.abs(weights_dd[:, 1]) + np.abs(weights_dd[:, 0] - self.prev_acc) / self._timestep
        else:
            self.jerk = np.ones_like(self.prev_acc) * np.inf

        self.prev_pos = np.polynomial.polynomial.polyval(tf, weights.T)
        self.prev_vel = np.polynomial.polynomial.polyval(tf, weights_d.T)
        self.prev_acc = np.polynomial.polynomial.polyval(tf, weights_dd.T)

        for t in np.linspace(self._timestep, self.dt, self._n_intermediate_steps):
            q = np.polynomial.polynomial.polyval(t, weights.T)
            qd = np.polynomial.polynomial.polyval(t, weights_d.T)
            qdd = np.polynomial.polynomial.polyval(t, weights_dd.T)
            yield q, qd, qdd

    def reset(self, obs=None):
        obs = super(PositionControl, self).reset(obs)
        self.prev_pos = self._data.qpos[self.actuator_joint_ids]
        self.prev_vel = self._data.qvel[self.actuator_joint_ids]
        self.prev_acc = np.zeros(len(self.actuator_joint_ids))
        self.i_error = np.zeros(len(self.actuator_joint_ids))
        self.prev_controller_cmd_pos = self._data.qpos[self.actuator_joint_ids]

        if self.debug:
            self.controller_record = deque(maxlen=self.info.horizon * self._n_intermediate_steps)
        return obs

    def _step_init(self, obs, action):
        super(PositionControl, self)._step_init(obs, action)
        self.action = action

        self.traj = self._interpolate_trajectory(self.action)

    def _compute_action(self, obs, action):
        cur_pos, cur_vel = self.get_joints(obs)

        desired_pos, desired_vel, desired_acc = next(self.traj)

        return self._controller(desired_pos, desired_vel, desired_acc, cur_pos, cur_vel)

    def _preprocess_action(self, action):
        assert action.shape == (2, self.n_robot_joints), "The shape of action should be (2, {}).".format(
            self.n_robot_joints)
        return super(PositionControl, self)._preprocess_action(action)


class PositionControlPlanar(PositionControl):
    def __init__(self, *args, **kwargs):
        p_gain = [960, 480, 240]
        d_gain = [60, 20, 4]
        i_gain = [0, 0, 0]
        super(PositionControlPlanar, self).__init__(p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs)


class PlanarPositionHit(PositionControlPlanar, planar.AirHockeyHit):
    pass


class PlanarPositionDefend(PositionControlPlanar, planar.AirHockeyDefend):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = PlanarPositionHit(viewer_params={"start_paused": True})
    env.reset()
    steps = 0
    phase_time = 20 * env.dt
    nq = env.env_info['robot']['n_joints']
    init_pos = env.init_state[:nq]
    max_vel = np.array(env.env_info['robot']['joint_vel_limit'][1])
    max_acc = max_vel / phase_time
    while True:
        steps += 1
        if steps * env.dt <= phase_time:
            acc = -max_acc
            vel = -max_acc * steps * env.dt
            pos = init_pos - 1 / 2 * max_acc * (steps * env.dt) ** 2
        elif steps * env.dt <= 2 * phase_time:
            acc = np.zeros_like(max_acc)
            vel = -max_vel
            pos = init_pos - 1 / 2 * max_acc * phase_time ** 2 - max_vel * (steps * env.dt - phase_time)
        elif steps * env.dt <= 3 * phase_time:
            acc = max_acc
            vel = -max_vel + max_acc * (steps * env.dt - 2 * phase_time)
            pos = init_pos - 1 / 2 * max_acc * phase_time ** 2 - max_vel * phase_time - max_vel * (
                    steps * env.dt - 2 * phase_time) + 1 / 2 * max_acc * (steps * env.dt - 2 * phase_time) ** 2
        elif steps * env.dt <= 4 * phase_time:
            acc = np.zeros(nq)
            vel = np.zeros(nq)
            pos = init_pos - 2 * max_vel * phase_time
        elif steps * env.dt <= 5 * phase_time:
            acc = max_acc
            vel = max_acc * (steps * env.dt - 4 * phase_time)
            pos = init_pos - 2 * max_vel * phase_time + 1 / 2 * max_acc * (steps * env.dt - 4 * phase_time) ** 2
        elif steps * env.dt <= 6 * phase_time:
            acc = np.zeros(nq)
            vel = max_vel
            pos = init_pos - 2 * max_vel * phase_time + 1 / 2 * max_acc * phase_time ** 2 + max_vel * (
                    steps * env.dt - 5 * phase_time)
        elif steps * env.dt <= 7 * phase_time:
            acc = -max_acc
            vel = max_vel - max_acc * (steps * env.dt - 6 * phase_time)
            pos = init_pos - 2 * max_vel * phase_time + 1 / 2 * max_acc * phase_time ** 2 + max_vel * phase_time + \
                  max_vel * (steps * env.dt - 6 * phase_time) - 1 / 2 * max_acc * (steps * env.dt - 6 * phase_time) ** 2
        else:
            acc = np.zeros(nq)
            vel = np.zeros(nq)
            pos = init_pos

        action = np.stack([pos, vel])
        observation, reward, done, info = env.step(action)
        env.render()

        if done or steps >= 200:
            if env.debug:
                trajectory_record = np.array(env.controller_record)
                fig, axes = plt.subplots(5, nq)
                nq_total = nq * env.n_agents
                for j in range(nq):
                    axes[0, j].plot(trajectory_record[:, j])
                    axes[0, j].plot(trajectory_record[:, j + nq_total])
                    axes[1, j].plot(trajectory_record[:, j + 2 * nq_total])
                    axes[1, j].plot(trajectory_record[:, j + 3 * nq_total])
                    axes[2, j].plot(trajectory_record[:, j + 4 * nq_total])
                    axes[3, j].plot(trajectory_record[:, j + 5 * nq_total])
                    axes[4, j].plot(trajectory_record[:, j + nq_total] - trajectory_record[:, j])
                plt.show()
            steps = 0
            env.reset()
            init_pos = env.init_state[:nq]
