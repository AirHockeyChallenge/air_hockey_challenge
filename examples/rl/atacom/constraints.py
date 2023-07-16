from abc import ABC, abstractmethod

import numpy as np


class Constraint(ABC):
    def __init__(self, name, dim_q, dim_k, dim_x=0):
        self.name = name
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_x = dim_x

    @abstractmethod
    def fun(self, q, x=None) -> np.ndarray:
        pass

    @abstractmethod
    def df_dq(self, q, x=None) -> np.ndarray:
        pass

    def df_dx(self, q, x=None) -> np.ndarray:
        pass

    def k_(self, q, x=None):
        k_out = self.fun(q, x)
        assert k_out.shape[0] == self.dim_k
        return k_out

    def J_q(self, q, x=None):
        jac_out = self.df_dq(q, x)
        assert jac_out.shape == (self.dim_k, self.dim_q)
        return jac_out

    def J_x(self, q, x=None):
        if self.dim_x == 0:
            return 0.
        else:
            jac_out = self.df_dx(q, x)
            assert jac_out.shape == (self.dim_k, self.dim_x)
            return jac_out


class ConstraintList:
    def __init__(self, dim_q, dim_x=0):
        self.dim_q = dim_q
        self.dim_x = dim_x

        self.dim_k = 0
        self.constraints = []
        self.constraints_idx = {}

    def add_constraint(self, k: Constraint):
        assert self.dim_q == k.dim_q
        self.constraints.append(k)
        self.constraints_idx.update({k.name: len(self.constraints)})
        self.dim_k += k.dim_k

    def k(self, q, x=None):
        k_tmp = np.array([])
        for k_i in self.constraints:
            k_tmp = np.concatenate([k_tmp, k_i.k_(q, x)])
        return k_tmp

    def J_q(self, q, x=None):
        J_tmp = list()
        for k_i in self.constraints:
            J_tmp.append(k_i.J_q(q, x))
        J_tmp = np.vstack(J_tmp)
        assert J_tmp.shape == (self.dim_k, self.dim_q)
        return J_tmp

    def J_x(self, q, x=None):
        if self.dim_x != 0:
            J_tmp = list()
            for k_i in self.constraints:
                J_tmp.append(k_i.J_x(q, x))
            J_tmp = np.vstack(J_tmp)

            assert J_tmp.shape == (self.dim_k, self.dim_x)
            return J_tmp
        else:
            return np.zeros(self.dim_k)
