import numpy as np
import scipy


class CubicLinearPlanner:
    def __init__(self, n_joints, step_size):
        self.n_joints = n_joints
        self.step_size = step_size

    def plan(self, start_pos, start_vel, end_pos, end_vel, t_total):
        t_total = self._round_time(t_total)
        coef = np.array([[1, 0, 0, 0], [1, t_total, t_total ** 2, t_total ** 3],
                         [0, 1, 0, 0], [0, 1, 2 * t_total, 3 * t_total ** 2]])
        results = np.vstack([start_pos, end_pos, start_vel, end_vel])

        A = scipy.linalg.block_diag(*[coef] * start_pos.shape[-1])
        y = results.reshape(-1, order='F')

        weights = np.linalg.solve(A, y).reshape(start_pos.shape[-1], 4)
        weights_d = np.polynomial.polynomial.polyder(weights, axis=1)
        weights_dd = np.polynomial.polynomial.polyder(weights_d, axis=1)

        t = np.linspace(self.step_size, t_total, int(t_total / self.step_size))

        x = weights[:, 0:1] + weights[:, 1:2] * t + weights[:, 2:3] * t ** 2 + weights[:, 3:4] * t ** 3
        dx = weights_d[:, 0:1] + weights_d[:, 1:2] * t + weights_d[:, 2:3] * t ** 2
        ddx = weights_dd[:, 0:1] + weights_dd[:, 1:2] * t
        return np.hstack([x.T, dx.T, ddx.T])

    def _round_time(self, time):
        return (round(time / self.step_size)) * self.step_size
