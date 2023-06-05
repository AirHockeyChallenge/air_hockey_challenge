import numpy as np
from scipy.interpolate import CubicSpline

from baseline.baseline_agent.bezier_planner_new import BezierPlanner
from baseline.baseline_agent.cubic_linear_planner import CubicLinearPlanner
from baseline.baseline_agent.optimizer import TrajectoryOptimizer
from baseline.baseline_agent.system_state import SystemState


class TrajectoryGenerator:
    def __init__(self, env_info, agent_params, system_state: SystemState):
        self.env_info = env_info
        self.dt = 1 / self.env_info['robot']['control_frequency']
        self.agent_params = agent_params
        self.state = system_state
        self.bezier_planner = self._init_bezier_planner()
        self.cubic_linear_planner = CubicLinearPlanner(self.env_info['robot']['n_joints'], self.dt)
        self.optimizer = TrajectoryOptimizer(self.env_info)

    def generate_stop_trajectory(self):
        q_plan = self.state.q_cmd + self.state.dq_cmd * 0.04
        joint_pos_traj = self.plan_cubic_linear_motion(self.state.q_cmd, self.state.dq_cmd, q_plan,
                                                       np.zeros_like(q_plan), 0.10)[:, :q_plan.shape[0]]

        t = np.linspace(0, joint_pos_traj.shape[0], joint_pos_traj.shape[0] + 1) * 0.02
        f = CubicSpline(t, np.vstack([self.state.q_cmd, joint_pos_traj]), axis=0, bc_type=((1, self.state.dq_cmd),
                                                                                           (2, np.zeros_like(q_plan))))
        df = f.derivative(1)
        self.state.trajectory_buffer = np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
        return True

    def _init_bezier_planner(self):
        self.bound_points = np.array([[-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-(self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius']),
                                       (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])],
                                      [-0.1, -(self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])]
                                      ])
        self.bound_points = self.bound_points + np.tile([1.51, 0.], (4, 1))
        self.boundary_idx = np.array([[0, 1], [1, 2], [0, 3]])

        table_bounds = np.array([[self.bound_points[0], self.bound_points[1]],
                                 [self.bound_points[1], self.bound_points[2]],
                                 [self.bound_points[2], self.bound_points[3]],
                                 [self.bound_points[3], self.bound_points[0]]])
        return BezierPlanner(table_bounds, self.dt)

    def plan_cubic_linear_motion(self, start_pos, start_vel, end_pos, end_vel, t_total=None):
        if t_total is None:
            t_total = np.linalg.norm(end_pos - start_pos) / self.agent_params['default_linear_vel']

        return self.cubic_linear_planner.plan(start_pos, start_vel, end_pos, end_vel, t_total)

    def generate_bezier_trajectory(self, max_steps=-1):
        if max_steps > 0:
            t_plan = np.minimum(self.bezier_planner.t_final, max_steps * self.dt)
        else:
            t_plan = self.bezier_planner.t_final
        res = np.array([self.bezier_planner.get_point(t_i) for t_i in np.arange(self.dt, t_plan + 1e-6, self.dt)])
        p = res[:, 0]
        dp = res[:, 1]
        ddp = res[:, 2]

        p = np.hstack([p, np.ones((p.shape[0], 1)) * self.env_info['robot']["ee_desired_height"]])
        dp = np.hstack([dp, np.zeros((p.shape[0], 1))])
        ddp = np.hstack([ddp, np.zeros((p.shape[0], 1))])
        return np.hstack([p, dp, ddp])

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        success, joint_pos_traj = self.optimizer.optimize_trajectory(cart_traj, q_start, dq_start,
                                                                     q_anchor)
        if len(joint_pos_traj) > 1:
            t = np.linspace(0, joint_pos_traj.shape[0], joint_pos_traj.shape[0] + 1) * 0.02
            f = CubicSpline(t, np.vstack([q_start, joint_pos_traj]), axis=0, bc_type=((1, dq_start),
                                                                                      (2, np.zeros_like(dq_start))))
            df = f.derivative(1)
            return success, np.stack([f(t[1:]), df(t[1:])]).swapaxes(0, 1)
        else:
            return success, []

    def solve_anchor_pos(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star

    def solve_anchor_pos_ik_null(self, hit_pos_2d, hit_dir_2d, q_0):
        hit_pos = np.concatenate([hit_pos_2d, [self.env_info['robot']["ee_desired_height"]]])
        hit_dir = np.concatenate([hit_dir_2d, [0]])
        success, q_star = self.optimizer.solve_hit_config_ik_null(hit_pos, hit_dir, q_0)
        if not success:
            q_star = q_0
        return q_star
