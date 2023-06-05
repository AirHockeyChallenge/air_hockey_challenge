import numpy as np

from air_hockey_challenge.environments.iiwas.env_single import AirHockeySingle


class AirHockeyPrepare(AirHockeySingle):
    """
        Class for the air hockey prepare task. The agent should try to
        improve the puck position.
    """
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        width_high = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - 0.002
        width_low = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2

        self.side_range = np.array([[-0.8, -0.2], [width_low, width_high]])
        self.bottom_range = np.array([[-0.94, -0.8], [self.env_info['table']['goal_width'] / 2, width_high]])

        self.side_area = (self.side_range[0, 1] - self.side_range[0, 0]) * \
                         (self.side_range[1, 1] - self.side_range[1, 0])
        self.bottom_area = (self.bottom_range[0, 1] - self.bottom_range[0, 0]) * \
                           (self.bottom_range[1, 1] - self.bottom_range[1, 0])

    def setup(self, obs):
        if np.random.rand() >= self.side_area / (self.side_area + self.bottom_area):
            start_range = self.bottom_range
        else:
            start_range = self.side_range

        puck_pos = np.random.rand(2) * (start_range[:, 1] - start_range[:, 0]) + start_range[:, 0]
        puck_pos *= [1, [1, -1][np.random.randint(2)]]

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])

        super(AirHockeyPrepare, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)
        if puck_pos[0] > 0 or np.abs(puck_pos[1]) < 1e-2:
            return True
        return super(AirHockeyPrepare, self).is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyPrepare()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0

    env.reset()
    while True:
        # action = np.random.uniform(-1, 1, env.info.action_space.low.shape) * 8
        action = np.zeros(7)
        observation, reward, done, info = env.step(action)
        env.render()
        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        steps += 1
        if done or steps > env.info.horizon:
            print("J: ", J, " R: ", R)
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()
