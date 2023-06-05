import numpy as np

from air_hockey_challenge.environments.planar.single import AirHockeySingle


class AirHockeyDefend(AirHockeySingle):
    """
    Class for the air hockey defending task.
    The agent should stop the puck at the line x=-0.6.
    """

    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):

        self.init_velocity_range = (1, 3)

        self.start_range = np.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame
        self.init_ee_range = np.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

    def setup(self, state=None):
        # Set initial puck parameters
        puck_pos = np.random.rand(2) * (self.start_range[:, 1] - self.start_range[:, 0]) + self.start_range[:, 0]

        lin_vel = np.random.uniform(self.init_velocity_range[0], self.init_velocity_range[1])
        angle = np.random.uniform(-0.5, 0.5)

        puck_vel = np.zeros(3)
        puck_vel[0] = -np.cos(angle) * lin_vel
        puck_vel[1] = np.sin(angle) * lin_vel
        puck_vel[2] = np.random.uniform(-10, 10, 1)

        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_x_vel", puck_vel[0])
        self._write_data("puck_y_vel", puck_vel[1])
        self._write_data("puck_yaw_vel", puck_vel[2])

        super(AirHockeyDefend, self).setup(state)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, state):
        puck_pos, puck_vel = self.get_puck(state)
        # If puck is over the middle line and moving towards opponent
        if puck_pos[0] > 0 and puck_vel[0] > 0:
            return True
        if np.linalg.norm(puck_vel[:2]) < 0.1:
            return True
        return super().is_absorbing(state)


if __name__ == '__main__':
    env = AirHockeyDefend()

    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    env.reset()
    while True:
        action = np.zeros(3)
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
