import copy
import time

import mujoco
import nlopt
import numpy as np
import osqp
import scipy.linalg
from scipy import sparse

from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name


class TrajectoryOptimizer:
    def __init__(self, env_info):
        self.env_info = env_info
        self.robot_model = copy.deepcopy(env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(env_info['robot']['robot_data'])
        self.n_joints = self.env_info['robot']['n_joints']
        if self.n_joints == 3:
            self.anchor_weights = np.ones(3)
        else:
            self.anchor_weights = np.array([10., 1., 10., 1., 10., 10., 1.])

    def optimize_trajectory(self, cart_traj, q_start, dq_start, q_anchor):
        joint_trajectory = np.tile(np.concatenate([q_start]), (cart_traj.shape[0], 1))
        if len(cart_traj) > 0:
            q_cur = q_start.copy()
            dq_cur = dq_start.copy()

            for i, des_point in enumerate(cart_traj):
                if q_anchor is None:
                    dq_anchor = 0
                else:
                    dq_anchor = (q_anchor - q_cur)

                success, dq_next = self._solve_aqp(des_point[:3], q_cur, dq_anchor)

                if not success:
                    return success, []
                else:
                    q_cur += (dq_cur + dq_next) / 2 * self.env_info['dt']
                    # q_cur += dq_next * self.env_info['dt']
                    dq_cur = dq_next
                    joint_trajectory[i] = q_cur.copy()
            return True, joint_trajectory
        else:
            return False, []

    def _solve_aqp(self, x_des, q_cur, dq_anchor):
        x_cur = forward_kinematics(self.robot_model, self.robot_data, q_cur)[0]
        jac = jacobian(self.robot_model, self.robot_data, q_cur)[:3, :self.n_joints]
        N_J = scipy.linalg.null_space(jac)
        b = np.linalg.lstsq(jac, (x_des - x_cur) / self.env_info['dt'], rcond=None)[0]

        P = (N_J.T @ np.diag(self.anchor_weights) @ N_J) / 2
        q = (b - dq_anchor).T @ np.diag(self.anchor_weights) @ N_J
        A = N_J.copy()
        u = np.minimum(self.env_info['robot']['joint_vel_limit'][1] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][1] * 0.92 - q_cur) / self.env_info['dt']) - b
        l = np.maximum(self.env_info['robot']['joint_vel_limit'][0] * 0.9,
                       (self.env_info['robot']['joint_pos_limit'][0] * 0.92 - q_cur) / self.env_info['dt']) - b

        solver = osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u, verbose=False, polish=False)

        result = solver.solve()
        if result.info.status == 'solved':
            return True, N_J @ result.x + b
        else:
            return False, b

    def solve_hit_config(self, x_des, v_des, q_0):
        reg = 1e-6
        dim = q_0.shape[0]
        opt = nlopt.opt(nlopt.LD_SLSQP, dim)

        def _nlopt_f(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(_nlopt_f, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        def _nlopt_h(q, grad):
            if grad.size > 0:
                grad[...] = 2 * (forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) @ \
                            jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return np.linalg.norm(forward_kinematics(self.robot_model, self.robot_data, q)[0] - x_des) ** 2 - 1e-4

        opt.set_max_objective(_nlopt_f)
        opt.set_lower_bounds(self.env_info['robot']['joint_pos_limit'][0])
        opt.set_upper_bounds(self.env_info['robot']['joint_pos_limit'][1])
        opt.add_inequality_constraint(_nlopt_h)
        opt.set_ftol_abs(1e-6)
        opt.set_xtol_abs(1e-8)
        opt.set_maxtime(5e-3)

        success, x = inverse_kinematics(self.robot_model, self.robot_data, x_des, initial_q=q_0)
        if not success:
            raise NotImplementedError("Need to check")
        xopt = opt.optimize(x[:dim])
        return opt.last_optimize_result() > 0, xopt

    def solve_hit_config_ik_null(self, x_des, v_des, q_0, max_time=5e-3):
        t_start = time.time()
        reg = 0e-6
        dim = q_0.shape[0]
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        i = 0
        TIME_MAX = max_time
        success = False

        dtype = self.robot_data.qpos.dtype

        self.robot_data.qpos = q_0

        q_l = self.robot_model.jnt_range[:, 0]
        q_h = self.robot_model.jnt_range[:, 1]
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        name = link_to_xml_name(self.robot_model, 'ee')

        def objective(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(objective, q)
            f = v_des @ jacobian(self.robot_model, self.robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        null_opt_stop_criterion = False
        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(self.robot_model, self.robot_data)

            x_pos = self.robot_data.body(name).xpos

            err_pos = x_des - x_pos
            error_norm = np.linalg.norm(err_pos)

            f_grad = numerical_grad(objective, self.robot_data.qpos.copy())
            f_grad_norm = np.linalg.norm(f_grad)
            if f_grad_norm > max_update_norm:
                f_grad = f_grad / f_grad_norm

            if error_norm < eps:
                success = True
            if time.time() - t_start > TIME_MAX or i >= IT_MAX or null_opt_stop_criterion:
                break

            jac_pos = np.empty((3, self.robot_model.nv), dtype=dtype)
            mujoco.mj_jacBody(self.robot_model, self.robot_data, jac_pos, None, self.robot_model.body(name).id)

            update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

            # Add Null space Projection
            null_dq = (np.eye(self.robot_model.nv) - np.linalg.pinv(jac_pos) @ jac_pos) @ f_grad
            null_opt_stop_criterion = np.linalg.norm(null_dq) < 1e-4
            update_joints += null_dq

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(self.robot_model, self.robot_data.qpos, update_joints, 1)
            self.robot_data.qpos = np.clip(self.robot_data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = self.robot_data.qpos.copy()

        return success, q_cur


def numerical_grad(fun, q):
    eps = np.sqrt(np.finfo(np.float64).eps)
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[i] += eps
        q_neg[i] -= eps
        grad[i] = (fun(q_pos, np.array([])) - fun(q_neg, np.array([]))) / 2 / eps
    return grad
