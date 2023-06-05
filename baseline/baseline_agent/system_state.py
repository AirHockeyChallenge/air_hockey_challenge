import copy
from enum import Enum

import numpy as np

from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from baseline.baseline_agent.kalman_filter import PuckTracker


class TACTICS(Enum):
    __order__ = "INIT READY PREPARE DEFEND REPEL SMASH N_TACTICS"
    INIT = 0
    READY = 1
    PREPARE = 2
    DEFEND = 3
    REPEL = 4
    SMASH = 5
    N_TACTICS = 6


class SystemState:
    def __init__(self, env_info, agent_id, agent_params):
        self.env_info = env_info
        self.agent_id = agent_id
        self.agent_params = agent_params
        self.puck_tracker = PuckTracker(self.env_info, agent_id)
        self.robot_model = copy.deepcopy(self.env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self.env_info['robot']['robot_data'])

        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.switch_tactics_count = self.agent_params['switch_tactics_min_steps']
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def reset(self):
        self.restart = True

        self.q_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.q_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.dq_cmd = np.zeros(self.env_info['robot']['n_joints'])
        self.dq_actual = np.zeros(self.env_info['robot']['n_joints'])

        self.x_cmd = np.zeros(3)
        self.x_actual = np.zeros(3)

        self.v_cmd = np.zeros(3)
        self.v_actual = np.zeros(3)

        self.predicted_state = np.zeros(6)
        self.predicted_cov = np.eye(6)
        self.predicted_time = 0.
        self.estimated_state = np.zeros(6)

        self.tactic_current = TACTICS.READY
        self.is_new_tactic = True
        self.tactic_finish = True
        self.has_generated_stop_traj = False
        self.plan_new_trajectory = True
        self.switch_tactics_count = self.agent_params['switch_tactics_min_steps']
        self.puck_static_count = 0
        self.puck_approaching_count = 0
        self.puck_transversal_moving_count = 0

        self.smash_finish = False

        self.trajectory_buffer = list()

    def is_puck_static(self):
        return self.puck_static_count > 3

    def is_puck_approaching(self):
        return self.puck_approaching_count > 3

    def is_puck_transversal_moving(self):
        return self.puck_transversal_moving_count > 3

    def update_observation(self, joint_pos_cur, joint_vel_cur, puck_state):
        if self.restart:
            self.puck_tracker.reset(puck_state)
            self.q_cmd = joint_pos_cur
            self.dq_cmd = joint_vel_cur
            self.x_cmd, self.v_cmd = self.update_ee_pos_vel(self.q_cmd, self.dq_cmd)
            self.restart = False

        self.q_actual = joint_pos_cur
        self.dq_actual = joint_vel_cur
        self.x_actual, self.v_actual = self.update_ee_pos_vel(self.q_actual, self.dq_actual)

        self.puck_tracker.step(puck_state)
        self.estimated_state = self.puck_tracker.state.copy()

        if np.linalg.norm(self.puck_tracker.state[2:4]) < self.agent_params['static_vel_threshold']:
            self.puck_static_count += 1
            self.puck_approaching_count = 0
            self.puck_transversal_moving_count = 0
        else:
            self.puck_static_count = 0
            puck_dir = self.puck_tracker.state[2:4] / np.linalg.norm(self.puck_tracker.state[2:4])
            if np.abs(np.dot(puck_dir, np.array([1., 0]))) < 0.15:
                self.puck_transversal_moving_count += 1
                self.puck_approaching_count = 0
            else:
                if self.puck_tracker.state[2] < 0 and self.puck_tracker.state[0] > self.agent_params['defend_range'][0]:
                    self.puck_approaching_count += 1

    def update_prediction(self, prediction_time, stop_line=0.):
        self.predicted_state, self.predicted_cov, self.predicted_time = \
            self.puck_tracker.get_prediction(prediction_time, stop_line)

    def update_ee_pos_vel(self, joint_pos, joint_vel):
        x_ee, _ = forward_kinematics(self.robot_model, self.robot_data, joint_pos)
        v_ee = jacobian(self.robot_model, self.robot_data, joint_pos)[:3,
               :self.env_info['robot']['n_joints']] @ joint_vel
        return x_ee, v_ee
