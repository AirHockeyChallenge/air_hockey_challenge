import mujoco
import numpy as np


def forward_kinematics(mj_model, mj_data, q, link="ee"):
    """
    Compute the forward kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (np.array):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    position: numpy.ndarray, (3,)
        Position of the link in robot's base frame
    orientation: numpy.ndarray, (3, 3)
        Orientation of the link in robot's base frame
    """

    return _mujoco_fk(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def inverse_kinematics(mj_model, mj_data, desired_position, desired_rotation=None, initial_q=None, link="ee"):
    """
    Compute the inverse kinematics of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is
        the case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        desired_position (numpy.ndarray, (3,)):
            The desired position of the selected link.
        desired_rotation (optional, numpy.array, (3,3)):
            The desired rotation of the selected link.
        initial_q (numpy.ndarray, None):
            The initial configuration of the algorithm, if set to None it will take the initial configuration of the
            mj_data.
        link (str, "ee"):
            Link for which the inverse kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]
    """
    q_init = np.zeros(mj_model.nq)
    if initial_q is None:
        q_init = mj_data.qpos
    else:
        q_init[:initial_q.size] = initial_q

    q_l = mj_model.jnt_range[:, 0]
    q_h = mj_model.jnt_range[:, 1]
    lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
    upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

    desired_quat = None
    if desired_rotation is not None:
        desired_quat = np.zeros(4)
        mujoco.mju_mat2Quat(desired_quat, desired_rotation.reshape(-1, 1))

    return _mujoco_clik(desired_position, desired_quat, q_init, link_to_xml_name(mj_model, link), mj_model,
                        mj_data, lower_limit, upper_limit)


def jacobian(mj_model, mj_data, q, link="ee"):
    """
    Compute the Jacobian of the robots.

    IMPORTANT:
        For the iiwa we assume that the universal joint at the end of the end-effector always leaves the mallet
        parallel to the table and facing down. This assumption only makes sense for a subset of robot configurations
        where the mallet can be parallel to the table without colliding with the rod it is mounted on. If this is the
        case this function will return the wrong values.

    Coordinate System:
        All translations and rotations are in the coordinate frame of the Robot. The zero point is in the center of the
        base of the Robot. The x-axis points forward, the z-axis points up and the y-axis forms a right-handed
        coordinate system

    Args:
        mj_model (mujoco.MjModel):
            mujoco MjModel of the robot-only model
        mj_data (mujoco.MjData):
            mujoco MjData object generated from the model
        q (numpy.ndarray):
            joint configuration for which the forward kinematics are computed
        link (string, "ee"):
            Link for which the forward kinematics is calculated. When using the iiwas the choices are
            ["1", "2", "3", "4", "5", "6", "7", "ee"]. When using planar the choices are ["1", "2", "3", "ee"]

    Returns
    -------
    numpy.ndarray, (6, num_joints):
        The Jacobian matrix for the robot kinematics.
    """
    return _mujoco_jac(q, link_to_xml_name(mj_model, link), mj_model, mj_data)


def link_to_xml_name(mj_model, link):
    try:
        mj_model.body('iiwa_1/base')
        link_to_frame_idx = {
            "1": "iiwa_1/link_1",
            "2": "iiwa_1/link_2",
            "3": "iiwa_1/link_3",
            "4": "iiwa_1/link_4",
            "5": "iiwa_1/link_5",
            "6": "iiwa_1/link_6",
            "7": "iiwa_1/link_7",
            "ee": "iiwa_1/striker_joint_link",
        }
    except:
        link_to_frame_idx = {
            "1": "planar_robot_1/body_1",
            "2": "planar_robot_1/body_2",
            "3": "planar_robot_1/body_3",
            "ee": "planar_robot_1/body_ee",
        }
    return link_to_frame_idx[link]


def _mujoco_fk(q, name, model, data):
    data.qpos[:len(q)] = q
    mujoco.mj_fwdPosition(model, data)
    return data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()


def _mujoco_jac(q, name, model, data):
    data.qpos[:len(q)] = q
    dtype = data.qpos.dtype
    jac = np.empty((6, model.nv), dtype=dtype)
    jac_pos, jac_rot = jac[:3], jac[3:]
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)
    return jac


def _mujoco_clik(desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit):
    IT_MAX = 1000
    eps = 1e-4
    damp = 1e-3
    progress_thresh = 20.0
    max_update_norm = 0.1
    rot_weight = 1
    i = 0

    dtype = data.qpos.dtype

    data.qpos = initial_q

    neg_x_quat = np.empty(4, dtype=dtype)
    error_x_quat = np.empty(4, dtype=dtype)

    if desired_pos is not None and desired_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if desired_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif desired_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("Desired Position and desired rotation is None, cannot compute inverse kinematics")

    while True:
        # forward kinematics
        mujoco.mj_fwdPosition(model, data)

        x_pos = data.body(name).xpos
        x_quat = data.body(name).xquat

        error_norm = 0
        if desired_pos is not None:
            err_pos[:] = desired_pos - x_pos
            error_norm += np.linalg.norm(err_pos)

        if desired_quat is not None:
            mujoco.mju_negQuat(neg_x_quat, x_quat)
            mujoco.mju_mulQuat(error_x_quat, desired_quat, neg_x_quat)
            mujoco.mju_quat2Vel(err_rot, error_x_quat, 1)
            error_norm += np.linalg.norm(err_rot) * rot_weight

        if error_norm < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        mujoco.mj_jacBody(model, data, jac_pos, jac_rot, model.body(name).id)

        hess_approx = jac.T.dot(jac)
        joint_delta = jac.T.dot(err)

        hess_approx += np.eye(hess_approx.shape[0]) * damp
        update_joints = np.linalg.solve(hess_approx, joint_delta)

        update_norm = np.linalg.norm(update_joints)

        # Check whether we are still making enough progress, and halt if not.
        progress_criterion = error_norm / update_norm
        if progress_criterion > progress_thresh:
            success = False
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        mujoco.mj_integratePos(model, data.qpos, update_joints, 1)
        data.qpos = np.clip(data.qpos, lower_limit, upper_limit)
        i += 1
    q_cur = data.qpos.copy()

    return success, q_cur
