import matplotlib.pyplot as plt
import numpy as np


class BezierPlanner:
    def __init__(self, boundary, step_size):
        self.boundary = boundary
        self.step_size = step_size
        self.p0 = None
        self.p1 = None
        self.p2 = None
        self.p3 = None

        self.z0 = None
        self.z1 = None
        self.z2 = None
        self.z3 = None

    def compute_control_point(self, p_start, v_start, p_stop, v_stop, t_plan=None):
        h_01 = np.inf
        h_23 = np.inf
        for b in self.boundary:
            b_s = b[0]
            b_f = b[1]

            if np.linalg.norm(v_start) > 1e-3:
                A_1 = np.vstack([v_start, b_s - b_f]).T
                b_1 = b_s - p_start
                if np.linalg.det(A_1) != 0:
                    h_1 = np.linalg.solve(A_1, b_1)[0]
                    if h_1 > 0:
                        h_01 = np.minimum(h_1, h_01)

            if np.linalg.norm(v_stop) > 1e-3:
                A_2 = np.vstack([-v_stop, b_s - b_f]).T
                b_2 = b_s - p_stop
                if np.linalg.det(A_2) != 0:
                    h_2 = np.linalg.solve(A_2, b_2)[0]
                    if h_2 > 0:
                        h_23 = np.minimum(h_2, h_23)

        self.z0 = np.array([0, 0])

        # Determine the time-phase curve with a second cubic Bézier curve, from [0, 0] to [t_final, 1]
        h_01, h_23 = self._plan_time_bezier(h_01, h_23, t_plan)

        self.p0 = p_start[:, np.newaxis]
        self.p1 = (p_start + v_start * h_01)[:, np.newaxis]
        self.p2 = (p_stop - v_stop * h_23)[:, np.newaxis]
        self.p3 = p_stop[:, np.newaxis]

    def _plan_time_bezier(self, h_01, h_23, t_plan):
        if h_01 == np.inf and h_23 == np.inf:
            # Case 1: Start and Stop velocity are both 0. If no planning time is specified, we use the predefined
            # Cartesian Velocity 1m/s to determine the time, else we use the specified time.
            # The time here is uniformly changing.
            if t_plan is None:
                self.t_final = self._round_time(np.linalg.norm(p_stop - p_start) / 1)  # Predefined Cart Velocity
            else:
                self.t_final = self._round_time(t_plan)
            # Constant Changing Phase
            self.z1 = np.array([self.t_final / 3, 1 / 3])
            self.z2 = np.array([self.t_final / 3 * 2, 2 / 3])
            self.z3 = np.array([self.t_final, 1])
            h_01 = 0
            h_23 = 0
        elif h_01 == np.inf:
            # Case 2: Start velocity is 0 and Stop velocity is non-zero. If no time specified, the total time equals
            # to the final time 1 / (1 / (3 * h_23)), where (3*h_23) is the required scale of dz_dt|f .
            # The third time bezier control point is determined by backtracking the 90% of the minimum required time.
            # The second control point is the average
            if t_plan is None:
                self.t_final = self._round_time(3 * h_23)  # Predefined Cart Velocity
            else:
                self.t_final = self._round_time(t_plan)

            t_neg = np.minimum(3 * h_23, self.t_final) * 0.9

            self.z2 = np.array([self.t_final - t_neg, 1 - 1 / (3 * h_23) * t_neg])
            self.z1 = (self.z2 + self.z0) / 2
            self.z3 = np.array([self.t_final, 1])
            h_01 = 0
        elif h_23 == np.inf:
            # Case 3: Start velocity is non-zero but final velocity is zero. If no time specified, the total time
            # equals to 1 / (1 / (3 * h_01)), where 3 * h_01 is the required scale of dz_dt|0.
            # The second time bezier control point is 1/3 of the minimum required time
            # The third is the average of z1 and z3
            if t_plan is None:
                self.t_final = self._round_time(3 * h_01)  # Predefined Cart Velocity
            else:
                self.t_final = self._round_time(t_plan)

            t_pos = np.minimum(3 * h_01, self.t_final) / 3
            self.z1 = np.array([t_pos, 1 / (3 * h_01) * t_pos])
            self.z3 = np.array([self.t_final, 1])
            self.z2 = (self.z3 + self.z1) / 2
            h_23 = 0
        else:
            dz_start = 1 / h_01 / 3
            dz_stop = 1 / h_23 / 3

            t_min = self._round_time(2 / (dz_start + dz_stop))
            self.t_final = t_min if t_plan is None else t_plan

            if self.t_final > t_min:
                self.z1 = np.array([1 / (dz_start + dz_stop), dz_start / (dz_start + dz_stop)])
                self.z2 = np.array([self.t_final - 1 / (dz_start + dz_stop), dz_start / (dz_start + dz_stop)])
            else:
                self.z1 = np.array([self.t_final / 2, dz_start * self.t_final / 2])
                self.z2 = np.array([self.t_final / 2, 1 - dz_stop * self.t_final / 2])
            self.z3 = np.array([self.t_final, 1])
        return h_01, h_23

    def get_point(self, t):
        z, dz_dt, ddz_ddt = self.get_time_bezier_root(t)

        z2 = z ** 2
        z3 = z ** 3
        nz_1 = 1 - z
        nz_2 = nz_1 * nz_1
        nz_3 = nz_2 * nz_1

        p = nz_3 * self.p0 + 3 * nz_2 * z * self.p1 + 3 * nz_1 * z2 * self.p2 + z3 * self.p3
        dp_dz = 3 * nz_2 * (self.p1 - self.p0) + 6 * nz_1 * z * (self.p2 - self.p1) + 3 * z2 * (self.p3 - self.p2)
        ddp_ddz = 6 * nz_1 * (self.p2 - 2 * self.p1 + self.p0) + 6 * z * (self.p3 - 2 * self.p2 + self.p1)
        return p, dp_dz * dz_dt, ddp_ddz * dz_dt ** 2 + dp_dz * ddz_ddt

    def get_time_bezier_root(self, t):
        if np.isscalar(t):
            cubic_polynomial = np.polynomial.polynomial.Polynomial([self.z0[0] - t,
                                                                    -3 * self.z0[0] + 3 * self.z1[0],
                                                                    3 * self.z0[0] - 6 * self.z1[0] + 3 * self.z2[0],
                                                                    -self.z0[0] + 3 * self.z1[0] - 3 * self.z2[0] +
                                                                    self.z3[0]])

        tau_orig = cubic_polynomial.roots()
        tau = tau_orig.real[np.logical_and(np.logical_and(tau_orig >= -1e-6, tau_orig <= 1. + 1e-6),
                                           np.logical_not(np.iscomplex(tau_orig)))]
        tau = tau[0]
        z = (1 - tau) ** 3 * self.z0 + 3 * (1 - tau) ** 2 * tau * self.z1 + 3 * (1 - tau) * (tau ** 2) * self.z2 + (
                tau ** 3) * self.z3
        dz_dtau = 3 * (1 - tau) ** 2 * (self.z1 - self.z0) + 6 * (1 - tau) * tau * (
                self.z2 - self.z1) + 3 * tau ** 2 * (self.z3 - self.z2)
        ddz_ddtau = 6 * (1 - tau) * (self.z2 - 2 * self.z1 + self.z0) + 6 * tau * (self.z3 - 2 * self.z2 + self.z1)

        z_t = z[1]
        dz_dt = dz_dtau[1] / dz_dtau[0]
        ddz_ddt = ddz_ddtau[1] / (dz_dtau[0]) ** 2 - dz_dtau[1] * ddz_ddtau[0] / (dz_dtau[0] ** 3)
        return z_t, dz_dt, ddz_ddt

    def _round_time(self, time):
        return (int(time / self.step_size) + 1) * self.step_size


if __name__ == '__main__':
    table_range = np.array([[0, -0.4], [0.8, 0.4]])

    corner_point = np.array([[table_range[0, 0], table_range[0, 1]],
                             [table_range[0, 0], table_range[1, 1]],
                             [table_range[1, 0], table_range[1, 1]],
                             [table_range[1, 0], table_range[0, 1]]])

    table_bounds = np.array([[corner_point[0], corner_point[1]],
                             [corner_point[1], corner_point[2]],
                             [corner_point[2], corner_point[3]],
                             [corner_point[3], corner_point[0]]])

    bezier_planner = BezierPlanner(table_bounds, 0.001)
    for i in range(10):
        p_start = np.random.uniform(table_range[0] * 0.8, table_range[1] * 0.8)
        p_stop = np.random.uniform(table_range[0], table_range[1])
        vel_start = np.random.randn(2) * 3
        vel_stop = np.random.randn(2)

        # p_start = np.array([0.2, 0.0])
        # p_stop = np.array([0.25, 0.0])
        # vel_start = np.array([-0.001, 0.001])
        # vel_stop = np.array([-0.001, -0.001])

        # vel_start = np.zeros(2)
        # vel_stop = np.zeros(2)

        t_total = 0.5
        bezier_planner.compute_control_point(p_start, vel_start, p_stop, vel_stop, t_total)
        t = np.linspace(0, bezier_planner.t_final, 1000)
        res = np.array([bezier_planner.get_point(t_i) for t_i in t])
        p = res[:, 0, :, 0].T
        dp = res[:, 1, :, 0].T
        ddp = res[:, 2, :, 0].T

        res_z_t = np.array([bezier_planner.get_time_bezier_root(t_i) for t_i in t])

        fig_2d, ax_2d = plt.subplots(1)
        ax_2d.arrow(*p_start, *vel_start / 3 * 0.1, width=0.01, color='r')
        ax_2d.arrow(*p_stop, *vel_stop / 3 * 0.1, width=0.01, color='b')
        ax_2d.plot(corner_point[[0, 1, 2, 3, 0], 0], corner_point[[0, 1, 2, 3, 0], 1], lw=5, color='k')
        ax_2d.scatter(*bezier_planner.p1, c='r', s=80, marker='x')
        ax_2d.scatter(*bezier_planner.p2, c='b', s=80, marker='x')
        ax_2d.set_xlim(-0.05, 0.85)
        ax_2d.set_ylim(-0.45, 0.45)
        ax_2d.set_aspect(1)
        ax_2d.plot(p[0], p[1])

        fig_t, axes_t = plt.subplots(3, 2)
        axes_t[0, 0].plot(t, p[0])
        axes_t[0, 1].plot(t, p[1])
        axes_t[1, 0].plot(t, dp[0], label='vel')
        axes_t[1, 1].plot(t, dp[1], label='vel')
        axes_t[1, 0].plot(t[1:], (p[0, 1:] - p[0, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[1, 1].plot(t[1:], (p[1, 1:] - p[1, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[2, 0].plot(t, ddp[0])
        axes_t[2, 1].plot(t, ddp[1])
        axes_t[2, 0].plot(t[1:], (dp[0, 1:] - dp[0, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[2, 1].plot(t[1:], (dp[1, 1:] - dp[1, :-1]) / (t[1:] - t[:-1]), label='fd')

        axes_t[0, 0].plot([0, bezier_planner.t_final], [p_stop[0], p_stop[0]], c='r', ls='--')
        axes_t[0, 1].plot([0, bezier_planner.t_final], [p_stop[1], p_stop[1]], c='r', ls='--')
        axes_t[1, 0].plot([0, bezier_planner.t_final], [vel_stop[0], vel_stop[0]], c='r', ls='--', label='final point')
        axes_t[1, 1].plot([0, bezier_planner.t_final], [vel_stop[1], vel_stop[1]], c='r', ls='--')

        fig_time, axes_time = plt.subplots(3)
        axes_time[0].plot(t, res_z_t[:, 0], label='z(t)')
        axes_time[1].plot(t, res_z_t[:, 1], label='dz_dt')
        axes_time[1].plot(t[1:], (res_z_t[1:, 0] - res_z_t[:-1, 0]) / (t[1:] - t[:-1]), label='fd')
        axes_time[2].plot(t, res_z_t[:, 2], label='ddz_ddt')
        axes_time[2].plot(t[1:], (res_z_t[1:, 1] - res_z_t[:-1, 1]) / (t[1:] - t[:-1]), label='fd')

        axes_t[1, 0].legend()
        axes_t[2, 0].legend()

        axes_time[0].scatter(bezier_planner.z0[0], bezier_planner.z0[1])
        axes_time[0].scatter(bezier_planner.z1[0], bezier_planner.z1[1])
        axes_time[0].scatter(bezier_planner.z2[0], bezier_planner.z2[1])
        axes_time[0].scatter(bezier_planner.z3[0], bezier_planner.z3[1])
        axes_time[0].legend()
        axes_time[1].legend()
        axes_time[2].legend()
        fig_time.suptitle("Z-T Plot")

        plt.show()
