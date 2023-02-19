import time
import threading
import numpy as np
from scipy.interpolate import CubicSpline
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.utils import inverse_kinematics, world_to_robot
from baseline.baseline_agent import BezierPlanner, TrajectoryOptimizer


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
    return HittingAgent(env_info, **kwargs)


class HittingAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, **kwargs):
        super(HittingAgent, self).__init__(env_info, agent_id, **kwargs)
        self.last_cmd = None
        self.joint_trajectory = None
        self.restart = True
        self.optimization_failed = False
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.ee_height = self.env_info['robot']["ee_desired_height"]

        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - 0.05),
                                       -(self.env_info['table']['width'] / 2 - 0.05)],
                                      [-(self.env_info['table']['length'] / 2 - 0.05),
                                       (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.2, (self.env_info['table']['width'] / 2 - 0.05)],
                                      [-0.2, -(self.env_info['table']['width'] / 2 - 0.05)]])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        self.bezier_planner = BezierPlanner(table_bounds, self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)
        if self.env_info['robot']['n_joints'] == 3:
            self.joint_anchor_pos = np.array([-1.15570723,  1.30024401,  1.44280414])
        else:
            self.joint_anchor_pos = np.array([6.28479822e-11, 7.13520517e-01, -2.96302903e-11, -5.02477487e-01,
                                              -7.67250279e-11, 1.92566224e+00, -2.34645597e-11])

    def reset(self):
        self.last_cmd = None
        self.joint_trajectory = []
        self.restart = True
        self.optimization_failed = False

    def draw_action(self, obs):
        if self.restart:
            self.restart = False
            puck_pos = self.get_puck_pos(obs)
            joint_pos = self.get_joint_pos(obs)
            joint_vel = self.get_joint_vel(obs)
            ee_pos, _ = self.get_ee_pose(obs)
            self.last_cmd = np.vstack([joint_pos, joint_vel])
            self.plan_thread = threading.Thread(target=self._plan_trajectory_thread, args=(puck_pos, ee_pos, joint_pos, joint_vel))
            self.plan_thread.start()

        if len(self.joint_trajectory) > 0:
            joint_pos_des, joint_vel_des = self.joint_trajectory[0]
            self.joint_trajectory = self.joint_trajectory[1:]
            self.last_cmd[1] = joint_vel_des
            self.last_cmd[0] = joint_pos_des
        else:
            self.last_cmd[1] = np.zeros(self.env_info['robot']['n_joints'])
            if not self.optimization_failed:
                time.sleep(0.01)
        return self.last_cmd

    def _plan_trajectory_thread(self, puck_pos, ee_pos, joint_pos, joint_vel):
        ee_traj, hit_idx, q_anchor = self.plan_ee_trajectory(puck_pos, ee_pos)
        _, joint_pos_traj = self.optimizer.optimize_trajectory(ee_traj, joint_pos, joint_vel, q_anchor)
        # joint_pos_traj = self.get_joint_trajectory(ee_traj)
        if len(joint_pos_traj) > 0:
            self.cubic_spline_interpolation(joint_pos_traj)
        else:
            self.optimization_failed = True
            self.joint_trajectory = np.array([])

    def plan_ee_trajectory(self, puck_pos, ee_pos):
        goal_pos = np.array([0.98, 0.0, 0.0])
        goal_pos_robot = world_to_robot(self.env_info["robot"]["base_frame"][0], goal_pos)
        goal_pos_2d = goal_pos_robot[0][:2]

        hit_dir_2d = goal_pos_2d - puck_pos[:2]
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)

        hit_pos_2d = puck_pos[:2] - hit_dir_2d * (
                self.env_info['puck']['radius'] + self.env_info['mallet']['radius'])

        start_pos_2d = ee_pos[:2]

        hit_pos_3d = np.concatenate([hit_pos_2d, [self.ee_height]])
        hit_dir_3d = np.concatenate([hit_dir_2d, [0.]])
        success, q_anchor = self.optimizer.solve_hit_config_ik_null(hit_pos_3d, hit_dir_3d, self.joint_anchor_pos)
        if not success:
            q_anchor = self.joint_anchor_pos

        hit_vel = 1.0
        self.bezier_planner.compute_control_point(start_pos_2d, np.zeros(2), hit_pos_2d, hit_dir_2d * hit_vel)

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
        stop_point = last_point_2d + hit_dir_2d * np.minimum(0.1, (1.2 - last_point_2d[1]) / hit_dir_2d[0])
        stop_point[1] = 0
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
        return ee_traj, len(hit_traj), q_anchor

    def get_joint_trajectory(self, ee_traj):
        init_q = self.last_cmd[0]
        joint_pos_traj = list()
        while len(ee_traj) > 0:
            ee_pos_des = ee_traj[0][:3]
            ee_traj = ee_traj[1:]
            success, joint_pos_des = inverse_kinematics(self.robot_model, self.robot_data,
                                                        ee_pos_des, initial_q=init_q)
            if not success:
                joint_pos_traj.clear()
                return joint_pos_traj
            init_q = joint_pos_des
            joint_pos_traj.append(joint_pos_des[:self.env_info['robot']['n_joints']])
        return joint_pos_traj

    def cubic_spline_interpolation(self, joint_pos_traj):
        joint_pos_traj = np.array(joint_pos_traj)
        t = np.linspace(1, joint_pos_traj.shape[0], joint_pos_traj.shape[0]) * 0.02

        f = CubicSpline(t, joint_pos_traj, axis=0)
        df = f.derivative(1)
        self.joint_trajectory = np.stack([f(t), df(t)]).swapaxes(0, 1)


def main():
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, debug=True)

    agent = HittingAgent(env.base_env.env_info)

    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps += 1
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if done or steps > env.info.horizon / 2:
            import matplotlib.pyplot as plt
            trajectory_record = np.array(env.base_env.controller_record)
            nq = env.base_env.env_info['robot']['n_joints']

            fig, axes = plt.subplots(3, nq)
            for j in range(nq):
                axes[0, j].plot(trajectory_record[:, j])
                axes[0, j].plot(trajectory_record[:, j + nq])
                axes[1, j].plot(trajectory_record[:, j + 2 * nq])
                axes[1, j].plot(trajectory_record[:, j + 3 * nq])
                # axes[2, j].plot(trajectory_record[:, j + 4 * nq])
                axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
            plt.show()

            steps = 0
            obs = env.reset()
            agent.reset()


if __name__ == '__main__':
    main()
