from abc import ABC, abstractmethod

import numpy as np


class ControlAffineSystem(ABC):
    def __init__(self, dim_q, dim_u):
        self.dim_q = dim_q
        self.dim_u = dim_u

    @abstractmethod
    def f(self, q):
        pass

    @abstractmethod
    def G(self, q):
        pass

    def dq(self, q, u):
        assert u.shape[-1] == self.dim_u
        return self.f(q) + self.G(q) @ u


class VelocityControlSystem(ControlAffineSystem):
    def __init__(self, dim_q, vel_limit):
        self.vel_limit = vel_limit
        super().__init__(dim_q, dim_q)

    def f(self, q):
        assert q.shape[-1] == self.dim_q
        return np.zeros(self.dim_q)

    def G(self, q):
        assert q.shape[-1] == self.dim_q
        return np.diag(self.vel_limit)


class AccelerationControlSystem(ControlAffineSystem):
    """
    We assume the q_hat is ordered by [q, q_dot], we therefore have
    """

    def __init__(self, dim_q, acc_limit):
        self.acc_limit = acc_limit
        super().__init__(dim_q * 2, dim_q)

    def f(self, q_hat):
        assert q_hat.shape[-1] == self.dim_q
        n_dim = int(self.dim_q / 2)
        return np.concatenate([q_hat[n_dim:], np.zeros(n_dim)])

    def G(self, q_hat):
        assert q_hat.shape[-1] == self.dim_q
        n_dim = int(self.dim_q / 2)
        return np.vstack([np.zeros((n_dim, n_dim)), np.diag(self.acc_limit)])
