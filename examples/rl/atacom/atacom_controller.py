import numpy as np
from scipy import linalg

from .constraints import ConstraintList
from .slack import Slack
from .system import ControlAffineSystem
from .utils import smooth_basis


class ATACOMController:
    def __init__(self, constraint: ConstraintList, system: ControlAffineSystem, slack_beta=3.,
                 slack_dynamics_type="soft_corner", slack_tol=1e-6, lambda_c=1):
        self.constraints = constraint
        self.system_dynamics = system
        self.dim_q = self.system_dynamics.dim_q
        self.dim_x = self.constraints.dim_x
        self.lambda_c = lambda_c

        self.slack = Slack(self.constraints.dim_k, beta=slack_beta, dynamics_type=slack_dynamics_type, tol=slack_tol)

    def get_q(self, s):
        """
        The state should be ordered as s = [q x].
        """
        return s[:self.dim_q]

    def get_x(self, s):
        """
        The state should be ordered as s = [q x]. When dim_x == 0, return None
        """
        if self.dim_x != 0:
            return s[self.dim_q:]
        else:
            return None

    def compose_action(self, s, u, x_dot=0.):
        u = np.clip(u, -1, 1)
        q = self.get_q(s)
        x = self.get_x(s)
        mu = self.get_mu(q, x)
        psi = self.psi(q, x, x_dot)
        J_G = self.J_G(q, x)
        J_u = self.J_u(J_G, mu)

        B_u = smooth_basis(J_u)[:, :self.system_dynamics.dim_u]
        u_drift_compensation = -np.linalg.pinv(J_G) @ psi
        u_contraction = -np.linalg.pinv(J_u)[:-self.constraints.dim_k] @ (
                    self.lambda_c * (self.constraints.k(q, x) + mu))

        u_auxiliary = u_drift_compensation + u_contraction
        u_tangent = B_u[:-self.constraints.dim_k] @ u

        scale_plus = (1 - u_auxiliary) / (u_tangent + 1e-8)
        scale_minus = (-1 - u_auxiliary) / (u_tangent + 1e-8)
        scale = np.clip(np.maximum(scale_plus, scale_minus).min(), 0, 1)
        u_tangent = scale * u_tangent

        u_s = u_auxiliary + u_tangent
        return u_s

    def get_mu(self, q, x):
        """
        q is controllable state, x is uncontrollable state
        """
        return self.slack.mu(self.constraints.k(q, x))

    def G_aug(self, q, mu):
        """
        G_aug = [G(q) 0; 0 A(mu)]
        """
        return linalg.block_diag(self.system_dynamics.G(q), self.slack.alpha(mu))

    def J_c(self, q, x):
        """
        J_c = [J_q(q, x) I]
        """
        return np.hstack([self.constraints.J_q(q, x), np.eye(self.constraints.dim_k)])

    def J_G(self, q, x):
        J_q = self.constraints.J_q(q, x)
        G = self.system_dynamics.G(q)
        return J_q @ G

    def J_u(self, J_G, mu):
        """
        J_u = J_c @ G_aug = [J_q(q, x)G(q) A(mu)]
        """
        return np.hstack([J_G, self.slack.alpha(mu)])

    def psi(self, q, x, x_dot):
        """
        psi = J_q(q, x)f(q) + J_x(q, x)x_dot
        """
        psi = self.constraints.J_q(q, x) @ self.system_dynamics.f(q)
        if self.dim_x != 0:
            psi += self.constraints.J_x(q, x) @ x_dot
        return psi
