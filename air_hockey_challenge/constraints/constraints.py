import copy

import numpy as np

from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian


class Constraint:
    def __init__(self, env_info, output_dim, **kwargs):
        """
        Constructor

        Args
        ----
        env_info: dict
            A dictionary contains information about the environment;
        output_dim: int
            The output dimension of the constraints.
        **kwargs: dict
            A dictionary contains agent related information.
        """
        self._env_info = env_info
        self._name = None

        self.output_dim = output_dim

        self._fun_value = np.zeros(self.output_dim)
        self._jac_value = np.zeros((self.output_dim, 2 * env_info["robot"]["n_joints"]))
        self._q_prev = None
        self._dq_prev = None

    @property
    def name(self):
        """
        The name of the constraints

        """
        return self._name

    def fun(self, q, dq):
        """
        The function of the constraint.

        Args
        ----
        q: numpy.ndarray, (num_joints,)
            The joint position of the robot
        dq: numpy.ndarray, (num_joints,)
            The joint velocity of the robot

        Returns
        -------
        numpy.ndarray, (out_dim,):
            The value computed by the constraints function.
        """
        if np.equal(q, self._q_prev).all() and np.equal(dq, self._dq_prev):
            return self._fun_value
        else:
            self._jacobian(q, dq)
            return self._fun(q, dq)

    def jacobian(self, q, dq):
        """
        Jacobian is the derivative of the constraint function w.r.t the robot joint position and velocity.

        Args
        ----
        q: ndarray, (num_joints,)
            The joint position of the robot
        dq: ndarray, (num_joints,)
            The joint velocity of the robot

        Returns
        -------
        numpy.ndarray, (dim_output, num_joints * 2):
            The flattened jacobian of the constraint function J = [dc / dq, dc / dq_dot]

        """
        if np.equal(q, self._q_prev).all() and np.equal(dq, self._dq_prev):
            return self._fun_value
        else:
            self._fun(q, dq)
            return self._jacobian(q, dq)

    def _fun(self, q, dq):
        raise NotImplementedError

    def _jacobian(self, q, dq):
        raise NotImplementedError


class ConstraintList:
    def __init__(self):
        self.constraints = dict()

    def keys(self):
        return self.constraints.keys()

    def get(self, key):
        return self.constraints.get(key)

    def add(self, c):
        self.constraints.update({c.name: c})

    def delete(self, name):
        del self.constraints[name]

    def fun(self, q, dq):
        return {key: self.constraints[key].fun(q, dq) for key in self.constraints}

    def jacobian(self, q, dq):
        return {key: self.constraints[key].jacobian(q, dq) for key in self.constraints}


class JointPositionConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, output_dim=2 * env_info["robot"]["n_joints"], **kwargs)
        self.joint_limits = self._env_info['robot']['joint_pos_limit'] * 0.95
        self._name = 'joint_pos_constr'

    def _fun(self, q, dq):
        self._fun_value[:int(self.output_dim / 2)] = q - self.joint_limits[1]
        self._fun_value[int(self.output_dim / 2):] = self.joint_limits[0] - q
        return self._fun_value

    def _jacobian(self, q, dq):
        self._jac_value[:int(self.output_dim / 2), :int(self.output_dim / 2)] = np.eye(
            self._env_info['robot']['n_joints'])
        self._jac_value[int(self.output_dim / 2):, :int(self.output_dim / 2)] = -np.eye(
            self._env_info['robot']['n_joints'])
        return self._jac_value


class JointVelocityConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, output_dim=2 * env_info["robot"]["n_joints"], **kwargs)
        self.joint_limits = self._env_info['robot']['joint_vel_limit'] * 0.95
        self._name = 'joint_vel_constr'

    def _fun(self, q, dq):
        self._fun_value[:int(self.output_dim / 2)] = dq - self.joint_limits[1]
        self._fun_value[int(self.output_dim / 2):] = self.joint_limits[0] - dq
        return self._fun_value

    def _jacobian(self, q, dq):
        self._jac_value[:int(self.output_dim / 2), int(self.output_dim / 2):] = np.eye(
            self._env_info['robot']['n_joints'])
        self._jac_value[int(self.output_dim / 2):, int(self.output_dim / 2):] = -np.eye(
            self._env_info['robot']['n_joints'])
        return self._jac_value


class EndEffectorConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        # 1 Dimension on x direction: x > x_lb
        # 2 Dimension on y direction: y > y_lb, y < y_ub
        # 2 Dimension on z direction: z > z_lb, z < z_ub
        super().__init__(env_info, output_dim=5, **kwargs)
        self._name = "ee_constr"
        tolerance = 0.02

        self.robot_model = copy.deepcopy(self._env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self._env_info['robot']['robot_data'])

        self.x_lb = - self._env_info['robot']['base_frame'][0][0, 3] - (
                self._env_info['table']['length'] / 2 - self._env_info['mallet']['radius'])
        self.y_lb = - (self._env_info['table']['width'] / 2 - self._env_info['mallet']['radius'])
        self.y_ub = (self._env_info['table']['width'] / 2 - self._env_info['mallet']['radius'])
        self.z_lb = self._env_info['robot']['ee_desired_height'] - tolerance
        self.z_ub = self._env_info['robot']['ee_desired_height'] + tolerance

    def _fun(self, q, dq):
        ee_pos, _ = forward_kinematics(self.robot_model, self.robot_data, q)
        self._fun_value = np.array([-ee_pos[0] + self.x_lb,
                                    -ee_pos[1] + self.y_lb, ee_pos[1] - self.y_ub,
                                    -ee_pos[2] + self.z_lb, ee_pos[2] - self.z_ub])
        return self._fun_value

    def _jacobian(self, q, dq):
        jac = jacobian(self.robot_model, self.robot_data, q)
        dc_dx = np.array([[-1, 0., 0.], [0., -1., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]])
        self._jac_value[:, :self._env_info['robot']['n_joints']] = dc_dx @ jac[:3, :self._env_info['robot']['n_joints']]
        return self._jac_value


class LinkConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        # 1 Dimension: wrist_z > minimum_height
        # 2 Dimension: elbow_z > minimum_height
        super().__init__(env_info, output_dim=2, **kwargs)
        self._name = "link_constr"

        self.robot_model = copy.deepcopy(self._env_info['robot']['robot_model'])
        self.robot_data = copy.deepcopy(self._env_info['robot']['robot_data'])

        self.z_lb = 0.25

    def _fun(self, q, dq):
        wrist_pos, _ = forward_kinematics(self.robot_model, self.robot_data, q, link="7")
        elbow_pos, _ = forward_kinematics(self.robot_model, self.robot_data, q, link="4")
        self._fun_value = np.array([-wrist_pos[2] + self.z_lb,
                                    -elbow_pos[2] + self.z_lb])
        return self._fun_value

    def _jacobian(self, q, dq):
        jac_wrist = jacobian(self.robot_model, self.robot_data, q, link="7")
        jac_elbow = jacobian(self.robot_model, self.robot_data, q, link="4")
        self._jac_value[:, :self._env_info['robot']['n_joints']] = np.vstack([
            -jac_wrist[2, :self._env_info['robot']['n_joints']],
            -jac_elbow[2, :self._env_info['robot']['n_joints']],
        ])
        return self._jac_value
