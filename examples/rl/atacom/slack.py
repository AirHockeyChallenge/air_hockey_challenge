import numpy as np


class Slack:
    def __init__(self, dim, beta=1., dynamics_type="soft_corner", tol=1e-6):
        self.dim = dim
        self.beta = beta
        self.dynamics_type = dynamics_type
        self.tol = tol

    def mu(self, k):
        return np.maximum(-k, self.tol)

    def alpha(self, mu):
        mu = np.atleast_1d(mu)
        if self.dynamics_type == "soft_corner":
            return np.diag(1 / np.maximum(np.exp(-self.beta * mu), self.tol) - 1)
        else:
            return np.diag(self.beta * mu)
