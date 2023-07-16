import numpy as np


class HitReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0
        # Get puck's position and velocity (The position is in the world frame, i.e., center of the table)
        puck_pos, puck_vel = mdp.get_puck(next_state)

        # Define goal position
        goal = np.array([0.98, 0])
        # Compute the vector that shoot the puck directly to the goal
        vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])

        # width of table minus radius of puck
        effective_width = mdp.env_info['table']['width'] / 2 - mdp.env_info['puck']['radius']

        # Calculate bounce point by assuming incoming angle = outgoing angle
        w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[0] -
             effective_width * goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)
        side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])

        # Compute the vector that shoot puck with a bounce to the wall
        vec_puck_side = (side_point - puck_pos[:2]) / \
                        np.linalg.norm(side_point - puck_pos[:2])

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        if absorbing or mdp._data.time < mdp.env_info['dt'] * 2:
            # If the hit scores
            if (puck_pos[0] - mdp.env_info['table']['length'] / 2) > 0 > \
                    (np.abs(puck_pos[1]) - mdp.env_info['table']['goal_width'] / 2):
                r = 50
            self.has_hit = False
        else:
            # If the puck has not yet been hit, encourage the robot to get closer to the puck
            if not self.has_hit:
                ee_pos = mdp.get_ee()[0][:2]
                dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos)

                vec_ee_puck = (puck_pos[:2] - ee_pos) / dist_ee_puck

                cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

                # Reward if vec_ee_puck and vec_puck_goal have the same direction
                cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
                cos_ang = np.max([cos_ang_goal, cos_ang_side])

                r = - dist_ee_puck / 2 + (cos_ang - 1) * 0.5
            else:
                r = min([1, 0.3 * np.maximum(puck_vel[0], 0.)])

                # Encourage the puck to end in the middle
                if puck_pos[0] > 0.7 and puck_vel[0] > 0.1:
                    r += 0.5 - np.abs(puck_pos[1])

                # penalizes the joint velocity
                q = mdp.get_joints(next_state, 1)[0]
                r -= 0.01 * np.linalg.norm(q - mdp.init_state)
        return r


class DefendReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        r = 0
        puck_pos, puck_vel = mdp.get_puck(next_state)

        if absorbing or mdp._data.time < mdp.env_info['dt'] * 2:
            self.has_hit = False

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        # This checks weather the puck is in our goal, heavy penalty if it is.
        # If absorbing the puck is out of bounds of the table.
        if absorbing:
            # puck position is behind table going to the negative side
            if puck_pos[0] + mdp.env_info['table']['length'] / 2 < 0 and \
                    (np.abs(puck_pos[1]) - mdp.env_info['table']['goal_width'] / 2) < 0:
                r = -100
            elif np.linalg.norm(puck_vel[:1]) < 0.1:
                # If the puck velocity is smaller than the threshold, the episode terminates with a high reward
                r = 150
        else:
            # If the puck bounced off the head walls, there is no reward.
            if puck_pos[0] <= -0.85 or puck_vel[0] > 0.3:
                r = -1
            # if the puck has been hit, or bounced off the wall
            elif puck_vel[0] > -0.2:
                # Reward if the puck slows down on the defending side
                r = 0.3 - abs(puck_vel[0])
            else:
                # If we did not yet hit the puck, reward is controlled by the distance between end effector and puck
                # on the x axis
                ee_pos = mdp.get_ee()[0][:2]
                ee_des = np.array([-0.6, puck_pos[1]])
                dist_ee_puck = ee_des - ee_pos
                r = - np.linalg.norm(dist_ee_puck)

        # penalizes the joint velocity
        q = mdp.get_joints(next_state)[0]
        r -= 0.005 * np.linalg.norm(q - mdp.init_state)
        return r


class PrepareReward:
    def __init__(self):
        self.has_hit = False

    def __call__(self, mdp, state, action, next_state, absorbing):
        puck_pos, puck_vel = mdp.get_puck(next_state)
        puck_pos = puck_pos[:2]
        puck_vel = puck_vel[:2]
        ee_pos = mdp.get_ee()[0][:2]

        if absorbing or mdp._data.time < mdp.env_info['dt'] * 2:
            self.has_hit = False

        if not self.has_hit:
            self.has_hit = _has_hit(mdp, state)

        if absorbing and abs(puck_pos[1]) < 0.3 and -0.8 < puck_pos[0] < -0.2:
            return 10
        elif absorbing:
            return -10
        else:
            if not self.has_hit:
                # encourage make contact
                dist_ee_puck = np.linalg.norm(puck_pos - ee_pos)
                vec_ee_puck = (puck_pos - ee_pos) / dist_ee_puck
                if puck_pos[0] > -0.65:
                    cos_ang = np.clip(vec_ee_puck @ np.array([0, np.copysign(1, puck_pos[1])]), 0, 1)
                else:
                    cos_ang_side = np.clip(vec_ee_puck @ np.array([np.copysign(0.05, -0.5 - puck_pos[0]),
                                                                   np.copysign(0.8, puck_pos[1])]), 0, 1)
                    cos_ang_bottom = np.clip(vec_ee_puck @ np.array([-1, 0]), 0, 1)
                    cos_ang = max([cos_ang_side, cos_ang_bottom])
                r = - dist_ee_puck / 2 + (cos_ang - 1) * 0.5
            else:
                if -0.5 < puck_pos[0] < -0.2 and puck_pos[1] < 0.3:
                    r = np.clip(np.abs(-0.5 - puck_pos[0]) / 2, 0, 1) + (0.3 - np.abs(puck_pos[1]))
                else:
                    r = 0

                q = mdp.get_joints(next_state)[0]
                r -= 0.005 * np.linalg.norm(q - mdp.init_state)
        return r


def _has_hit(mdp, state):
    ee_pos, ee_vel = mdp.get_ee()
    puck_cur_pos, _ = mdp.get_puck(state)
    if np.linalg.norm(ee_pos[:2] - puck_cur_pos[:2]) < mdp.env_info['puck']['radius'] + \
            mdp.env_info['mallet']['radius'] + 5e-3:
        return True
    else:
        return False