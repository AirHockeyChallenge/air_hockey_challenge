from air_hockey_challenge.framework import AirHockeyChallengeWrapper

env = AirHockeyChallengeWrapper('3dof-hit')

# Setup the environment
obs = env.reset()

env_info = env.env_info

# Get the keys of the available constraint
print(env_info['constraints'].keys())
# dict_keys(['joint_pos_constr', 'joint_vel_constr', 'ee_constr'])

# Get the joint position and velocity from the observation
q = obs[env_info['joint_pos_ids']]
dq = obs[env_info['joint_vel_ids']]

# Get a dictionary of the constraint functions {"constraint_name": ndarray}
c = env_info['constraints'].fun(q, dq)

# Get a dictionary of the constraint jacobian {"constraint_name": 2d ndarray}
jac = env_info['constraints'].jacobian(q, dq)

# Get value of the constraint function by name
c_ee = env_info['constraints'].get('ee_constr').fun(q, dq)

# Get jacobian of the constraint function by name
jac_vel = env_info['constraints'].get('joint_vel_constr').jacobian(q, dq)
