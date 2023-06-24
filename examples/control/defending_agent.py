import threading
import time

import numpy as np
from scipy.interpolate import CubicSpline

from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer, PuckTracker


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
    return DefendingAgent(env_info, **kwargs)


class DefendingAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super(DefendingAgent, self).__init__(env_info, agent_id, **kwargs)
        self.last_cmd = None
        self.joint_trajectory = None
        self.restart = True
        self.optimization_failed = False
        self.tactic_finished = False
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.ee_height = self.env_info['robot']["ee_desired_height"]

        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - 0.05),
                                       -(self.env_info['table']['width'] / 2 - 0.05)],
                                      [-(self.env_info['table']['length'] / 2 - 0.05),
                                       (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.3, -(self.env_info['table']['width'] / 2 - 0.05)]])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        self.bezier_planner = BezierPlanner(table_bounds, self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)
        if self.env_info['robot']['n_joints'] == 3:
            self.joint_anchor_pos = np.array([-0.9273, 0.9273, np.pi / 2])
        else:
            self.joint_anchor_pos = np.array([6.28479822e-11, 7.13520517e-01, -2.96302903e-11, -5.02477487e-01,
                                              -7.67250279e-11, 1.92566224e+00, -2.34645597e-11])

        self.puck_tracker = PuckTracker(env_info, agent_id=agent_id)
        self._obs = None

        self.agent_params = {
            'hit_range': [0.8, 1.3],
            'max_plan_steps': 10,
        }

    def reset(self):
        self.last_cmd = None
        self.joint_trajectory = []
        self.restart = True
        self.tactic_finished = False
        self.optimization_failed = False

        self._obs = None

        self.plan_thread = threading.Thread(target=self._plan_trajectory_thread)
        self.plan_thread.start()

    def draw_action(self, obs):
        if self.restart:
            self.restart = False
            self.puck_tracker.reset(self.get_puck_pos(obs))
            self.last_cmd = np.vstack([self.get_joint_pos(obs), self.get_joint_vel(obs)])
            self.joint_trajectory = np.array([self.last_cmd])

        self.puck_tracker.step(self.get_puck_pos(obs))
        self._obs = obs.copy()

        if len(self.joint_trajectory) > 0:
            joint_pos_des, joint_vel_des = self.joint_trajectory[0]
            self.joint_trajectory = self.joint_trajectory[1:]
            self.last_cmd[1] = joint_vel_des
            self.last_cmd[0] = joint_pos_des
        else:
            self.last_cmd[1] = np.zeros(self.env_info['robot']['n_joints'])
            if not self.tactic_finished:
                time.sleep(0.005)
        return self.last_cmd

    def _plan_trajectory_thread(self):
        while not self.tactic_finished:
            time.sleep(0.01)
            opt_trial = 0
            t_predict = 1.0
            defend_line = 0.8
            state, P, t_predict = self.puck_tracker.get_prediction(t_predict, defend_line)
            if len(self.joint_trajectory) < self.agent_params['max_plan_steps']:
                if np.linalg.det(P[:2, :2]) < 1e-3:
                    joint_pos = self.get_joint_pos(self._obs)
                    joint_vel = self.get_joint_vel(self._obs)
                    puck_pos = state[[0, 1, 4]]
                    ee_pos, _ = self.get_ee_pose(self._obs)

                    ee_traj, switch_idx = self.plan_ee_trajectory(puck_pos, ee_pos, t_predict)
                    _, joint_pos_traj = self.optimizer.optimize_trajectory(ee_traj, joint_pos, joint_vel, None)
                    if len(joint_pos_traj) > 0:
                        if len(self.joint_trajectory) > 0:
                            self.joint_trajectory = np.vstack([self.joint_trajectory,
                                                               self.cubic_spline_interpolation(joint_pos_traj)])
                        else:
                            self.joint_trajectory = self.cubic_spline_interpolation(joint_pos_traj)
                        self.tactic_finished = True
                    else:
                        opt_trial += 1
                        self.optimization_failed = True
                        self.joint_trajectory = np.array([])
            if opt_trial >= 5:
                self.tactic_finished = True
                break

    def plan_ee_trajectory(self, puck_pos, ee_pos, t_plan):
        hit_dir_2d = np.array([0., np.sign(puck_pos[1])])

        hit_pos_2d = puck_pos[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])

        start_pos_2d = ee_pos[:2]

        hit_vel = 0
        self.bezier_planner.compute_control_point(start_pos_2d, np.zeros(2), hit_pos_2d, hit_dir_2d * hit_vel, t_plan)

        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(0, self.bezier_planner.t_final + 1e-6,
                                                                                self.dt)])
        p = res[1:, 0].squeeze()
        dp = res[1:, 1].squeeze()
        ddp = res[1:, 2].squeeze()

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])

        hit_traj = np.hstack([p, dp, ddp])

        last_point_2d = hit_traj[-1, :2]
        last_vel_2d = hit_traj[-1, 3:5]

        # Plan Return Trajectory
        stop_point = np.array([0.65, 0.])
        self.bezier_planner.compute_control_point(last_point_2d, last_vel_2d, stop_point, np.zeros(2), 1.5)

        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(0, self.bezier_planner.t_final + 1e-6,
                                                                                self.dt)])
        p = res[1:, 0].squeeze()
        dp = res[1:, 1].squeeze()
        ddp = res[1:, 2].squeeze()

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.ee_height])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return_traj = np.hstack([p, dp, ddp])

        ee_traj = np.vstack([hit_traj, return_traj])
        return ee_traj, len(hit_traj)

    def cubic_spline_interpolation(self, joint_pos_traj):
        joint_pos_traj = np.array(joint_pos_traj)
        t = np.linspace(1, joint_pos_traj.shape[0], joint_pos_traj.shape[0]) * 0.02

        f = CubicSpline(t, joint_pos_traj, axis=0)
        df = f.derivative(1)
        return np.stack([f(t), df(t)]).swapaxes(0, 1)


def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("tkAgg")

    env = AirHockeyChallengeWrapper(env="7dof-defend", interpolation_order=3, debug=True)

    agent = DefendingAgent(env.base_env.env_info)

    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if done or steps > env.info.horizon / 2:
            nq = env.base_env.env_info['robot']['n_joints']
            if env.base_env.debug:
                trajectory_record = np.array(env.base_env.controller_record)
                fig, axes = plt.subplots(5, nq)
                nq_total = nq * env.base_env.n_agents
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
            obs = env.reset()
            agent.reset()


if __name__ == '__main__':
    main()
