import numpy as np

from examples.rl.atacom.constraints import Constraint


class JointPosVelConstraint(Constraint):
    def __init__(self, env_info):
        name = "joint_pos_vel"
        self.n_joints = env_info['robot']['n_joints']
        self.joint_pos_constr = env_info['constraints'].get('joint_pos_constr')
        self.joint_vel_constr = env_info['constraints'].get('joint_vel_constr')
        self.K = 0.2
        super().__init__(name, dim_q=self.n_joints * 2, dim_k=self.n_joints * 4, dim_x=0)

    def fun(self, q, x=None) -> np.ndarray:
        """
        For second order dynamics, the state is augmented as q_hat = [q, dq]
        We modify the constraint to viability constraint as
        c(q) + K c_dot(q) = c(q) + K J_c(q) q_dot < 0
        """
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        c_pos = self.joint_pos_constr.fun(pos, vel) + \
                self.K * self.joint_pos_constr.jacobian(pos, vel)[:, :self.n_joints] @ vel
        c_vel = self.joint_vel_constr.fun(pos, vel)
        return np.concatenate([c_pos, c_vel])

    def df_dq(self, q, x=None):
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        J_pos = self.joint_pos_constr.jacobian(pos, vel)[:, :self.n_joints]
        J_dot_pos = 0.  # This term does not change
        J_vel = self.K * J_pos
        df_pos_dq = np.hstack([J_pos + self.K * J_dot_pos, J_vel])
        df_vel_dq = np.hstack([np.zeros_like(J_pos), self.joint_vel_constr.jacobian(pos, vel)[:, self.n_joints:]])
        return np.vstack([df_pos_dq, df_vel_dq])


class EndEffectorConstraint(Constraint):
    def __init__(self, env_info):
        name = "ee_pos_vel"
        self.n_joints = env_info['robot']['n_joints']
        self.ee_constr = env_info['constraints'].get('ee_constr')
        self.K = 0.8
        self.eps = 1e-6

        if self.n_joints == 3:
            dim_k = 3
        else:
            dim_k = 5
        super().__init__(name, dim_q=self.n_joints * 2, dim_k=dim_k, dim_x=0)

    def fun(self, q, x=None) -> np.ndarray:
        """
        For second order dynamics, the state is augmented as q_hat = [q, dq]
        We modify the constraint to viability constraint as
        c(q) + K c_dot(q) = c(q) + K J_c(q) q_dot < 0
        """
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        J_q_ = self.ee_constr.jacobian(pos, vel)[:, :self.n_joints].copy()
        return (self.ee_constr.fun(pos, vel) + self.K * J_q_ @ vel)[:self.dim_k]

    def df_dq(self, q, x=None) -> np.ndarray:
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]

        J_q_ = self.ee_constr.jacobian(pos, vel)[:, :self.n_joints].copy()
        J_dq_ = self.K * J_q_[:, :self.n_joints]

        pos_n_ = pos + vel * self.eps
        dot_J = (self.ee_constr.jacobian(pos_n_, vel)[:, :self.n_joints] - J_q_) / self.eps
        return np.hstack([J_q_ + self.K * dot_J, J_dq_])[:self.dim_k, :]


class EndEffectorPosConstraint(Constraint):
    def __init__(self, env_info):
        name = "ee_pos"
        self.n_joints = env_info['robot']['n_joints']
        self.ee_constr = env_info['constraints'].get('ee_constr')
        self.link_constr = env_info['constraints'].get('link_constr')

        if self.n_joints == 3:
            dim_k = 3
        else:
            dim_k = 7
        super().__init__(name, dim_q=self.n_joints, dim_k=dim_k, dim_x=0)

    def fun(self, q, x=None) -> np.ndarray:
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        val = np.concatenate([self.ee_constr.fun(pos, vel), self.link_constr.fun(pos, vel)])
        return val + 2e-2

    def df_dq(self, q, x=None) -> np.ndarray:
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]

        J_q_ = np.vstack([self.ee_constr.jacobian(pos, vel)[:, :self.n_joints].copy(),
                          self.link_constr.jacobian(pos, vel)[:, :self.n_joints].copy()])
        return J_q_[:self.dim_k, :]


class JointPosConstraint(Constraint):
    def __init__(self, env_info):
        name = "joint_pos"
        self.n_joints = env_info['robot']['n_joints']
        self.joint_pos_constr = env_info['constraints'].get('joint_pos_constr')
        super().__init__(name, dim_q=self.n_joints, dim_k=self.n_joints * 2, dim_x=0)

    def fun(self, q, x=None) -> np.ndarray:
        """
        For second order dynamics, the state is augmented as q_hat = [q, dq]
        We modify the constraint to viability constraint as
        c(q) + K c_dot(q) = c(q) + K J_c(q) q_dot < 0
        """
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        c_pos = self.joint_pos_constr.fun(pos, vel)
        return c_pos + 1e-2

    def df_dq(self, q, x=None):
        pos = q[:self.n_joints]
        vel = q[self.n_joints:]
        J_pos = self.joint_pos_constr.jacobian(pos, vel)[:, :self.n_joints]
        return J_pos
