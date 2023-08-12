import numpy as np

from baseline.baseline_agent.system_state import SystemState, TACTICS
from baseline.baseline_agent.trajectory_generator import TrajectoryGenerator


class Tactic:
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator):
        self.env_info = env_info
        self.agent_params = agent_params
        self.state = state
        self.generator = trajectory_generator

        self.state.tactic_finish = True
        self.plan_new_trajectory = True
        self.replan_time = 0
        self.switch_count = 0

    def ready(self):
        pass

    def apply(self):
        pass

    def update_tactic(self):
        self._update_prediction()
        if self.state.switch_tactics_count > self.agent_params['switch_tactics_min_steps'] or \
                self.state.tactic_finish:
            self._update_tactic_impl()
        else:
            self.state.switch_tactics_count += 1

    def _update_prediction(self):
        if self.state.estimated_state[0] < self.agent_params['defend_range'][0]:
            self.state.update_prediction(self.state.predicted_time)
        else:
            self.state.update_prediction(self.state.predicted_time, self.agent_params['defend_range'][0])

    def _update_tactic_impl(self):
        pass

    def _set_tactic(self, tactic):
        if tactic != self.state.tactic_current:
            self.state.is_new_tactic = True
            self.state.switch_tactics_count = 0
            self.state.tactic_current = tactic
            self.state.has_generated_stop_traj = False

    def can_smash(self):
        if self.state.is_puck_static():
            if self.agent_params['hit_range'][0] < self.state.predicted_state[0] < self.agent_params['hit_range'][1] \
                    and np.abs(self.state.predicted_state[1]) < self.env_info['table']['width'] / 2 - \
                    self.env_info['puck']['radius'] - 2 * self.env_info['mallet']['radius']:
                return True
        return False

    def should_defend(self):
        if self.state.is_puck_approaching():
            if self.agent_params['defend_range'][0] <= self.state.predicted_state[0] <= \
                    self.agent_params['defend_range'][1] \
                    and np.abs(self.state.predicted_state[1]) <= self.agent_params['defend_width'] and \
                    self.state.predicted_time >= self.agent_params['max_plan_steps'] * self.generator.dt:
                return True
            elif self.state.predicted_time < self.agent_params['max_prediction_time'] and \
                    self.state.predicted_state[0] > self.agent_params['defend_range'][1]:
                self.state.predicted_time += ((self.agent_params['defend_range'][1] - self.state.predicted_state[0]) /
                                              self.state.predicted_state[2])
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params['max_prediction_time'])
        return False

    def is_puck_stuck(self):
        if self.state.is_puck_static():
            if self.state.predicted_state[0] < self.agent_params['hit_range'][0]:
                return True
            elif self.state.predicted_state[0] < self.agent_params['hit_range'][1] \
                    and np.abs(self.state.predicted_state[1]) > self.env_info['table']['width'] / 2 - \
                    self.env_info['puck']['radius'] - 2 * self.env_info['mallet']['radius']:
                return True
        return False


class Init(Tactic):
    def _update_tactic_impl(self):
        pass

    def ready(self):
        if self.state.is_new_tactic:
            return True
        else:
            return False

    def apply(self):
        if self.state.is_new_tactic:
            if np.linalg.norm(self.state.dq_cmd) > 0.01:
                if not self.state.has_generated_stop_traj:
                    self.generator.generate_stop_trajectory()
                    self.state.has_generated_stop_traj = True
            else:
                self.state.trajectory_buffer = []

            if len(self.state.trajectory_buffer) == 0:
                t_init = 2.0
                for i in range(10):
                    success = self._plan_init_trajectory(t_init)
                    if success:
                        self.state.is_new_tactic = False
                        break
                    t_init *= 1.2

    def _plan_init_trajectory(self, t_final):
        cart_traj = self.generator.plan_cubic_linear_motion(self.state.x_cmd, self.state.v_cmd,
                                                            self.agent_params['x_init'], np.zeros(3), t_final)
        opt_success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
            cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params['joint_anchor_pos'])
        return opt_success


class Ready(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator,
                 only_tactic=None):
        super(Ready, self).__init__(env_info, agent_params, state, trajectory_generator)

        self.only_tactic = only_tactic

    def _update_tactic_impl(self):
        if self.only_tactic is None:
            if self.can_smash():
                self._set_tactic(TACTICS.SMASH)
            elif self.should_defend():
                self._set_tactic(TACTICS.DEFEND)
            elif self.is_puck_stuck():
                self._set_tactic(TACTICS.PREPARE)

        elif self.only_tactic == "hit":
            if self.can_smash():
                self._set_tactic(TACTICS.SMASH)

        elif self.only_tactic == "defend":
            if self.should_defend():
                self._set_tactic(TACTICS.DEFEND)

        elif self.only_tactic == "prepare":
            if self.is_puck_stuck():
                self._set_tactic(TACTICS.PREPARE)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.replan_time = 0
            self.switch_count = 0
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self.t_stop = np.maximum(np.linalg.norm(self.agent_params['x_home'] - self.state.x_cmd) /
                                     self.agent_params['default_linear_vel'], 1.0)
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                self.t_stop = np.maximum(np.linalg.norm(self.agent_params['x_home'] - self.state.x_cmd) /
                                         self.agent_params['default_linear_vel'], 1.0)
                return True
        return False

    def apply(self):
        self.state.is_new_tactic = False

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    self.agent_params['x_home'][:2], np.zeros(2),
                                                                    self.t_stop)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, self.agent_params['x_home'][:2],
                                                                      np.zeros(2), self.t_stop)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.agent_params['joint_anchor_pos'])

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.t_stop = self.generator.bezier_planner.t_final - \
                              self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return
            self.t_stop *= 1.5
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Prepare(Tactic):
    def _update_tactic_impl(self):
        if not self.is_puck_stuck():
            self.switch_count += 1
        else:
            self.switch_count = 0

        if (self.switch_count > 4 or self.state.tactic_finish) and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self.replan_time = 0
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params['joint_anchor_pos']
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        puck_pos_2d = self.state.predicted_state[:2]

        if puck_pos_2d[0] < self.agent_params['prepare_range'][0]:
            hit_dir_2d = np.array([-1, np.sign(puck_pos_2d[1] + 1e-6) * 0.2])
            hit_vel_mag = 0.2
        elif abs(puck_pos_2d[0]) > np.mean(self.agent_params['prepare_range']):
            hit_dir_2d = np.array([-0.5, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2
        else:
            hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])
            hit_vel_mag = 0.2

        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'] +
                                                     self.env_info['puck']['radius'])
        hit_vel_2d = hit_dir_2d * hit_vel_mag

        # self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.state.predicted_time = np.maximum(np.linalg.norm(self.state.x_cmd[:2] - puck_pos_2d) /
                                                       self.agent_params['default_linear_vel'], 1.0)
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= 2 * self.generator.dt:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                break

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                return

            self.state.predicted_time += self.agent_params['max_plan_steps'] * self.generator.dt
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Defend(Tactic):
    def _update_tactic_impl(self):
        self.state.update_prediction(self.state.predicted_time, self.agent_params['defend_range'][0])
        if not self.should_defend():
            self.switch_count += 1
        else:
            self.switch_count = 0

        if (self.switch_count > 4 or self.state.tactic_finish) and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.state.tactic_finish = False
            self.plan_new_trajectory = True
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self.replan_time = 0
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params['joint_anchor_pos']
            return True
        else:
            if len(self.state.trajectory_buffer) == 0:
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        puck_pos_2d = self.state.predicted_state[:2]

        hit_dir_2d = np.array([0, np.sign(puck_pos_2d[1] + 1e-6)])

        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])
        hit_vel_2d = hit_dir_2d * 0.05

        # self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= 2 * self.generator.dt:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                break

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.clip(self.state.predicted_time, 0,
                                                    self.agent_params['max_prediction_time'])
                return

            self.state.predicted_time += self.agent_params['max_plan_steps'] * self.generator.dt
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True


class Repel(Tactic):
    def _update_tactic_impl(self):
        pass

    def ready(self):
        pass

    def apply(self):
        pass


class Smash(Tactic):
    def __init__(self, env_info, agent_params, state: SystemState, trajectory_generator: TrajectoryGenerator):
        super().__init__(env_info, agent_params, state, trajectory_generator)
        self.hit_vel_mag = self.agent_params['max_hit_velocity']
        self.q_anchor_pos = self.agent_params['joint_anchor_pos']

    def _update_tactic_impl(self):
        if not self.can_smash() or self.state.tactic_finish:
            self.switch_count += 1
        else:
            self.switch_count = 0

        if self.switch_count > 4 and len(self.state.trajectory_buffer) == 0:
            self._set_tactic(TACTICS.READY)

    def ready(self):
        if self.state.is_new_tactic:
            self.plan_new_trajectory = True
            self.state.tactic_finish = False
            self.state.predicted_time = self.agent_params['max_prediction_time']
            self.replan_time = 0
            self.hit_vel_mag = self.agent_params['max_hit_velocity']
            self.switch_count = 0
            self.q_anchor_pos = self.agent_params['joint_anchor_pos']
            return True
        else:
            if len(self.state.trajectory_buffer) == 0 and self.can_smash():
                return True
            return False

    def apply(self):
        self.state.is_new_tactic = False

        goal_pos = np.array([2.49, 0.0])
        puck_pos_2d = self.state.predicted_state[:2]

        hit_dir_2d = goal_pos - puck_pos_2d
        hit_dir_2d = hit_dir_2d / np.linalg.norm(hit_dir_2d)
        hit_vel_2d = hit_dir_2d * self.hit_vel_mag

        hit_pos_2d = puck_pos_2d[:2] - hit_dir_2d * (self.env_info['mallet']['radius'])
        self.q_anchor_pos = self.generator.solve_anchor_pos_ik_null(hit_pos_2d, hit_dir_2d, self.q_anchor_pos)

        for i in range(10):
            if self.plan_new_trajectory:
                self.generator.bezier_planner.compute_control_point(self.state.x_cmd[:2], self.state.v_cmd[:2],
                                                                    hit_pos_2d, hit_vel_2d, self.state.predicted_time)
            else:
                if self.generator.bezier_planner.t_final > self.replan_time:
                    self.generator.bezier_planner.update_bezier_curve(self.replan_time, hit_pos_2d,
                                                                      hit_vel_2d, self.state.predicted_time)

            if self.generator.bezier_planner.t_final >= 2 * self.agent_params['max_plan_steps'] * self.generator.dt:
                cart_traj = self.generator.generate_bezier_trajectory(self.agent_params['max_plan_steps'])
            elif self.generator.bezier_planner.t_final >= self.replan_time:
                self.state.tactic_finish = True
                cart_traj = self.generator.generate_bezier_trajectory()
            else:
                self.state.tactic_finish = True
                return

            success, self.state.trajectory_buffer = self.generator.optimize_trajectory(
                cart_traj, self.state.q_cmd, self.state.dq_cmd, self.q_anchor_pos)

            if success:
                self.replan_time = self.agent_params['max_plan_steps'] * self.generator.dt
                self.plan_new_trajectory = False
                self.state.predicted_time = self.generator.bezier_planner.t_final - \
                                            self.state.trajectory_buffer.shape[0] * self.generator.dt
                self.state.predicted_time = np.maximum(0, self.state.predicted_time)
                return

            self.state.predicted_time += 2 * self.generator.dt
            self.hit_vel_mag *= 0.9
            hit_vel_2d = hit_dir_2d * self.hit_vel_mag
            self.replan_time = 0
            self.state.tactic_finish = False

        self.state.tactic_finish = True
