import matplotlib
import numpy as np

matplotlib.use('tkAgg')
import matplotlib.pyplot as plt


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

        self.t_final = 0

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
            else:
                h_01 = 0

            if np.linalg.norm(v_stop) > 1e-3:
                A_2 = np.vstack([-v_stop, b_s - b_f]).T
                b_2 = b_s - p_stop
                if np.linalg.det(A_2) != 0:
                    h_2 = np.linalg.solve(A_2, b_2)[0]
                    if h_2 > 0:
                        h_23 = np.minimum(h_2, h_23)
            else:
                h_23 = 0

        self.p0 = p_start.copy()
        self.p3 = p_stop.copy()

        dz_start = 0
        dz_stop = 0
        if h_01 == 0 and h_23 == 0:
            self.p1 = (p_start + v_start * h_01)
            self.p2 = (p_stop - v_stop * h_23)
        elif h_01 == 0:
            l_start = p_stop - v_stop * h_23
            min_length = np.minimum(0.15 / h_23, 1)
            self.p2 = self.get_closest_point_from_line_to_point(l_start, p_stop, p_start,
                                                                [0, 1 - min_length])
            self.p1 = p_start
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3
        elif h_23 == 0:
            l_end = p_start + v_start * h_01
            min_length = np.minimum(0.15 / h_01, 1)
            self.p1 = self.get_closest_point_from_line_to_point(p_start, l_end, p_stop, [min_length, 1])
            self.p2 = p_stop
            dz_start = np.linalg.norm(v_start) / np.linalg.norm(self.p1 - self.p0) / 3
        else:
            l1_end = p_start + v_start * h_01
            l2_start = p_stop - v_stop * h_23
            min_length1 = np.minimum(0.15 / h_01, 1)
            min_length2 = np.minimum(0.15 / h_23, 1)
            self.p1, self.p2 = self.get_closest_point_between_line_segments(p_start, l1_end, l2_start, p_stop,
                                                                            np.array([[min_length1, 1],
                                                                                      [0, 1 - min_length2]]))
            self.p1 = self.p1
            self.p2 = self.p2
            dz_start = np.linalg.norm(v_start) / np.linalg.norm(self.p1 - self.p0) / 3
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3

        if t_plan is None:
            if abs(dz_start + dz_stop) > 1e-3:
                self.t_final = self._round_time(1 / (dz_start + dz_stop))
            else:
                self.t_final = self._round_time(np.linalg.norm(p_stop - p_start) / 1)  # Predefined Cart Velocity
        else:
            self.t_final = self._round_time(t_plan)

        self.compute_time_bezier(dz_start, dz_stop, self.t_final)

    def compute_time_bezier(self, dz_start, dz_stop, t_plan):
        self.z0 = np.array([0, 0])
        self.z3 = np.array([t_plan, 1])
        if dz_start == 0 and dz_stop == 0:
            self.z1 = np.array([t_plan / 3, 1 / 3])
            self.z2 = np.array([t_plan / 3 * 2, 2 / 3])
        else:
            t_min = np.minimum(1 / (dz_start + dz_stop), self.t_final)
            if t_min == self.t_final:
                a = 0.5
                b = 0.5
            else:
                if dz_start == 0:
                    a = 0.1
                    b = 1 - a
                elif dz_stop == 0:
                    a = 0.9
                    b = 1 - a
                else:
                    a = 0.5
                    b = 0.5

            self.z1 = np.array([a * t_min, dz_start * a * t_min])
            self.z2 = np.array([self.t_final - b * t_min, 1 - dz_stop * b * t_min])

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

    def update_bezier_curve(self, t_start, p_stop, v_stop, t_final):
        z, dz_dt, _ = self.get_time_bezier_root(t_start)
        dp_dz = 3 * (1 - z) ** 2 * (self.p1 - self.p0) + 6 * (1 - z) * z * (self.p2 - self.p1) + 3 * z ** 2 * (
                self.p3 - self.p2)

        h_23 = np.inf
        for b in self.boundary:
            if np.linalg.norm(v_stop) > 1e-3:
                A = np.vstack([-v_stop, b[0] - b[1]]).T
                b = b[0] - p_stop
                if np.linalg.det(A) != 0:
                    h = np.linalg.solve(A, b)[0]
                    if h > 0:
                        h_23 = np.minimum(h, h_23)
            else:
                h_23 = 0
        if h_23 == 0:
            p2_new = p_stop
        else:
            l2_start = p_stop - v_stop * h_23
            min_length2 = np.minimum(0.1 / h_23, 1)
            _, p2_new = self.get_closest_point_between_line_segments(self.p0, self.p1, l2_start,
                                                                     p_stop, np.array([[0, 1], [0, 1 - min_length2]]))
        p3_new = p_stop

        p_new = np.array([[-(z - 1) ** 3, 3 * (z - 1) ** 2 * z, -3 * (z - 1) * z ** 2, z ** 3],
                          [0, (z - 1) ** 2, -2 * (z - 1) * z, z ** 2],
                          [0, 0, 1 - z, z],
                          [0, 0, 0, 1]]) @ np.vstack([self.p0, self.p1, self.p2, self.p3])

        self.p0 = p_new[0].copy()
        self.p1 = p_new[1].copy()
        self.p2 = (1 - z) * p2_new + z * p3_new
        self.p3 = p3_new

        self.t_final = self._round_time(t_final)
        if np.linalg.norm(self.p1 - self.p0) > 1e-3:
            dz_start = np.linalg.norm(dp_dz * dz_dt) / np.linalg.norm(self.p1 - self.p0) / 3
        else:
            dz_start = 0
        if h_23 == 0:
            dz_stop = 0
        else:
            dz_stop = np.linalg.norm(v_stop) / np.linalg.norm(self.p3 - self.p2) / 3
        self.compute_time_bezier(dz_start, dz_stop, self.t_final)

    def _round_time(self, time):
        return (round(time / self.step_size)) * self.step_size

    @staticmethod
    def get_closest_point_from_line_to_point(l0, l1, p, range=None):
        v = l1 - l0
        u = l0 - p
        t = - np.dot(v, u) / np.dot(v, v)
        t = np.clip(t, 0, 1)
        if range is not None:
            t = np.clip(t, range[0], range[1])
        return (1 - t) * l0 + t * l1

    @staticmethod
    def get_closest_point_between_line_segments(l1s, l1e, l2s, l2e, range=None):
        v = l1s - l2s  # d13
        u = l2e - l2s  # d43
        w = l1e - l1s  # d21
        A = np.array([[w @ w, -u @ w],
                      [w @ u, -u @ u]])
        b = -np.array([[v @ w], [v @ u]])
        if np.linalg.det(A) != 0:
            mu = np.linalg.solve(A, b)
            mu = np.clip(mu, 0, 1)
            if range is not None:
                mu = np.clip(mu, range[:, 0:1], range[:, 1:2])
            return l1s + mu[0] * (l1e - l1s), l2s + mu[1] * (l2e - l2s)
        elif np.linalg.norm(u) == 0:
            range_1 = None if range is None else range[0]
            p = BezierPlanner.get_closest_point_from_line_to_point(l1s, l1e, l2e, range_1)
            return p, l2s
        elif np.linalg.norm(w) == 0:
            range_1 = None if range is None else range[1]
            p = BezierPlanner.get_closest_point_from_line_to_point(l2s, l2e, l1s, range_1)
            return l1s, p
        else:
            return l1s + 0.5 * (l1e - l1s), l2s + 0.5 * (l2e - l2s)


if __name__ == '__main__':
    np.random.seed(2)
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
        # pos_start = np.random.uniform(table_range[0] * 0.8, table_range[1] * 0.8)
        # pos_stop = np.random.uniform(table_range[0], table_range[1])
        # vel_start = np.random.randn(2)
        # vel_stop = np.random.randn(2) * 0

        pos_start = np.array([0.1, 0.0])
        pos_stop = np.array([0.5, 0.3])
        vel_start = np.array([0.0, 0.0])
        vel_stop = np.array([2.0, 0.0])

        t_total = 0.8
        bezier_planner.compute_control_point(pos_start, vel_start, pos_stop, vel_stop, t_total)
        t = np.arange(0, bezier_planner.t_final + 1e-6, 0.01)
        res = np.array([bezier_planner.get_point(t_i) for t_i in t])
        p = res[:, 0].T
        dp = res[:, 1].T
        ddp = res[:, 2].T

        res_z_t = np.array([bezier_planner.get_time_bezier_root(t_i) for t_i in t])

        fig_2d, ax_2d = plt.subplots(1)
        ax_2d.arrow(*pos_start, *vel_start / 3 * 0.1, width=0.01, color='r')
        ax_2d.arrow(*pos_stop, *vel_stop / 3 * 0.1, width=0.01, color='b')
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
        axes_t[2, 0].plot(t, ddp[0])
        axes_t[2, 1].plot(t, ddp[1])

        axes_t[0, 0].plot([0, bezier_planner.t_final], [pos_start[0], pos_start[0]], c='g', ls='--')
        axes_t[0, 1].plot([0, bezier_planner.t_final], [pos_start[1], pos_start[1]], c='g', ls='--')
        axes_t[1, 0].plot([0, bezier_planner.t_final], [vel_start[0], vel_start[0]], c='g', ls='--',
                          label='start point')
        axes_t[1, 1].plot([0, bezier_planner.t_final], [vel_start[1], vel_start[1]], c='g', ls='--')
        axes_t[0, 0].plot([0, bezier_planner.t_final], [pos_stop[0], pos_stop[0]], c='r', ls='--')
        axes_t[0, 1].plot([0, bezier_planner.t_final], [pos_stop[1], pos_stop[1]], c='r', ls='--')
        axes_t[1, 0].plot([0, bezier_planner.t_final], [vel_stop[0], vel_stop[0]], c='r', ls='--', label='final point')
        axes_t[1, 1].plot([0, bezier_planner.t_final], [vel_stop[1], vel_stop[1]], c='r', ls='--')

        fig_time, axes_time = plt.subplots(3)
        axes_time[0].plot(t, res_z_t[:, 0], label='z(t)')
        axes_time[1].plot(t, res_z_t[:, 1], label='dz_dt')
        axes_time[2].plot(t, res_z_t[:, 2], label='ddz_ddt')

        axes_t[1, 0].legend()

        axes_time[0].scatter(bezier_planner.z0[0], bezier_planner.z0[1])
        axes_time[0].scatter(bezier_planner.z1[0], bezier_planner.z1[1])
        axes_time[0].scatter(bezier_planner.z2[0], bezier_planner.z2[1])
        axes_time[0].scatter(bezier_planner.z3[0], bezier_planner.z3[1])
        axes_time[0].legend()
        axes_time[1].legend()
        axes_time[2].legend()
        fig_time.suptitle("Z-T Plot")

        t_offset = 0.2
        pos_stop_new = np.array([0.5, 0.3])
        vel_stop_new = np.array([2.0, 0.0])
        bezier_planner.update_bezier_curve(t_offset, pos_stop_new, vel_stop_new, 0.5)
        t = np.arange(0, bezier_planner.t_final + 1e-6, 0.001)
        res = np.array([bezier_planner.get_point(t_i) for t_i in t])
        p2 = res[:, 0].T
        dp2 = res[:, 1].T
        ddp2 = res[:, 2].T

        ax_2d.scatter(*bezier_planner.p1, c='orange', s=80, marker='x')
        ax_2d.scatter(*bezier_planner.p2, c='cyan', s=80, marker='x')
        ax_2d.plot(p2[0], p2[1])

        axes_t[0, 0].plot(t + t_offset, p2[0])
        axes_t[0, 1].plot(t + t_offset, p2[1])
        axes_t[1, 0].plot(t + t_offset, dp2[0], label='vel')
        axes_t[1, 1].plot(t + t_offset, dp2[1], label='vel')
        # axes_t[1, 0].plot(t[1:] + 0.1, (p2[0, 1:] - p2[0, :-1]) / (t[1:] - t[:-1]), label='fd')
        # axes_t[1, 1].plot(t[1:] + 0.1, (p2[1, 1:] - p2[1, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[2, 0].plot(t + t_offset, ddp2[0])
        axes_t[2, 1].plot(t + t_offset, ddp2[1])

        res_z_t2 = np.array([bezier_planner.get_time_bezier_root(t_i) for t_i in t])
        axes_time[0].plot(t + t_offset, res_z_t2[:, 0], label='z(t)')
        axes_time[1].plot(t + t_offset, res_z_t2[:, 1], label='dz_dt')
        axes_time[2].plot(t + t_offset, res_z_t2[:, 2], label='ddz_ddt')
        axes_time[0].scatter(bezier_planner.z0[0] + t_offset, bezier_planner.z0[1])
        axes_time[0].scatter(bezier_planner.z1[0] + t_offset, bezier_planner.z1[1])
        axes_time[0].scatter(bezier_planner.z2[0] + t_offset, bezier_planner.z2[1])
        axes_time[0].scatter(bezier_planner.z3[0] + t_offset, bezier_planner.z3[1])

        t_offset = 0.2
        pos_stop_new = np.array([0.5, 0.3])
        vel_stop_new = np.array([2.0, 0.0])
        bezier_planner.update_bezier_curve(t_offset, pos_stop_new, vel_stop_new, 0.3)
        t = np.arange(0, bezier_planner.t_final + 1e-6, 0.001)
        res = np.array([bezier_planner.get_point(t_i) for t_i in t])
        p2 = res[:, 0].T
        dp2 = res[:, 1].T
        ddp2 = res[:, 2].T

        ax_2d.scatter(*bezier_planner.p1, c='orange', s=80, marker='x')
        ax_2d.scatter(*bezier_planner.p2, c='cyan', s=80, marker='x')
        ax_2d.plot(p2[0], p2[1])

        axes_t[0, 0].plot(t + 2 * t_offset, p2[0])
        axes_t[0, 1].plot(t + 2 * t_offset, p2[1])
        axes_t[1, 0].plot(t + 2 * t_offset, dp2[0], label='vel')
        axes_t[1, 1].plot(t + 2 * t_offset, dp2[1], label='vel')
        # axes_t[1, 0].plot(t[1:] + 0.1, (p2[0, 1:] - p2[0, :-1]) / (t[1:] - t[:-1]), label='fd')
        # axes_t[1, 1].plot(t[1:] + 0.1, (p2[1, 1:] - p2[1, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[2, 0].plot(t + 2 * t_offset, ddp2[0])
        axes_t[2, 1].plot(t + 2 * t_offset, ddp2[1])

        res_z_t2 = np.array([bezier_planner.get_time_bezier_root(t_i) for t_i in t])
        axes_time[0].plot(t + 2 * t_offset, res_z_t2[:, 0], label='z(t)')
        axes_time[1].plot(t + 2 * t_offset, res_z_t2[:, 1], label='dz_dt')
        axes_time[2].plot(t + 2 * t_offset, res_z_t2[:, 2], label='ddz_ddt')
        axes_time[0].scatter(bezier_planner.z0[0] + 2 * t_offset, bezier_planner.z0[1])
        axes_time[0].scatter(bezier_planner.z1[0] + 2 * t_offset, bezier_planner.z1[1])
        axes_time[0].scatter(bezier_planner.z2[0] + 2 * t_offset, bezier_planner.z2[1])
        axes_time[0].scatter(bezier_planner.z3[0] + 2 * t_offset, bezier_planner.z3[1])

        t_offset = 0.2
        pos_stop_new = np.array([0.5, 0.3])
        vel_stop_new = np.array([2.0, 0.0])
        bezier_planner.update_bezier_curve(t_offset, pos_stop_new, vel_stop_new, 0.1)
        t = np.arange(0, bezier_planner.t_final + 1e-6, 0.001)
        res = np.array([bezier_planner.get_point(t_i) for t_i in t])
        p2 = res[:, 0].T
        dp2 = res[:, 1].T
        ddp2 = res[:, 2].T

        ax_2d.scatter(*bezier_planner.p1, c='orange', s=80, marker='x')
        ax_2d.scatter(*bezier_planner.p2, c='cyan', s=80, marker='x')
        ax_2d.plot(p2[0], p2[1])

        axes_t[0, 0].plot(t + 3 * t_offset, p2[0])
        axes_t[0, 1].plot(t + 3 * t_offset, p2[1])
        axes_t[1, 0].plot(t + 3 * t_offset, dp2[0], label='vel')
        axes_t[1, 1].plot(t + 3 * t_offset, dp2[1], label='vel')
        # axes_t[1, 0].plot(t[1:] + 0.1, (p2[0, 1:] - p2[0, :-1]) / (t[1:] - t[:-1]), label='fd')
        # axes_t[1, 1].plot(t[1:] + 0.1, (p2[1, 1:] - p2[1, :-1]) / (t[1:] - t[:-1]), label='fd')
        axes_t[2, 0].plot(t + 3 * t_offset, ddp2[0])
        axes_t[2, 1].plot(t + 3 * t_offset, ddp2[1])

        res_z_t2 = np.array([bezier_planner.get_time_bezier_root(t_i) for t_i in t])
        axes_time[0].plot(t + 3 * t_offset, res_z_t2[:, 0], label='z(t)')
        axes_time[1].plot(t + 3 * t_offset, res_z_t2[:, 1], label='dz_dt')
        axes_time[2].plot(t + 3 * t_offset, res_z_t2[:, 2], label='ddz_ddt')
        axes_time[0].scatter(bezier_planner.z0[0] + 3 * t_offset, bezier_planner.z0[1])
        axes_time[0].scatter(bezier_planner.z1[0] + 3 * t_offset, bezier_planner.z1[1])
        axes_time[0].scatter(bezier_planner.z2[0] + 3 * t_offset, bezier_planner.z2[1])
        axes_time[0].scatter(bezier_planner.z3[0] + 3 * t_offset, bezier_planner.z3[1])

        plt.show()
