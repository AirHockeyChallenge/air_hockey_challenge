import mujoco
import numpy as np

from air_hockey_challenge.environments.iiwas.env_double import AirHockeyDouble


class AirHockeyTournament(AirHockeyDouble):
    """
        Class for the air hockey tournament. Consists of 2 robots which should play against each other.
        When the puck is on one side for more than 15 seconds the puck is reset and the player gets a penalty.
        If a player accumulates 3 penalties his score is reduced by 1.
    """
    def __init__(self, gamma=0.99, horizon=45000, viewer_params={}, agent_name="Agent", opponent_name="Opponent"):
        self.agent_name = agent_name
        self.opponent_name = opponent_name

        self.score = [0, 0]
        self.faults = [0, 0]
        self.start_side = np.random.choice([1, -1])

        self.prev_side = self.start_side
        self.timer = 0

        def custom_render_callback(viewport, context):
            names = f"Agents \nScores \nFaults "
            data = f"{self.agent_name} - {self.opponent_name}\n "
            data += f"{self.score[0]} - {self.score[1]}\n "
            data += f"{self.faults[0]} - {self.faults[1]}"
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_BIG, mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, names, data, context)

        viewer_params["custom_render_callback"] = custom_render_callback
        super().__init__(gamma=gamma, horizon=horizon, viewer_params=viewer_params)

        hit_width = self.env_info['table']['width'] / 2 - self.env_info['puck']['radius'] - \
                    self.env_info['mallet']['radius'] * 2

        self.hit_range = np.array([[-0.7, -0.2], [-hit_width, hit_width]])  # Table Frame

    def setup(self, obs):
        # Initial position of the puck
        puck_pos = np.random.rand(2) * (self.hit_range[:, 1] - self.hit_range[:, 0]) + self.hit_range[:, 0]

        self._write_data("puck_x_pos", puck_pos[0] * self.start_side)
        self._write_data("puck_y_pos", puck_pos[1])

        self.prev_side = self.start_side
        self.timer = 0

        super(AirHockeyTournament, self).setup(obs)

    def reward(self, state, action, next_state, absorbing):
        return 0

    def is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)

        # Puck stuck on one side for more than 15s
        if np.sign(puck_pos[0]) == self.prev_side:
            self.timer += self.dt
        else:
            self.prev_side *= -1
            self.timer = 0

        if self.timer > 15.0 and np.abs(puck_pos[0]) >= 0.15:
            if self.prev_side == -1:
                self.faults[0] += 1
                self.start_side = -1
                if self.faults[0] % 3 == 0:
                    self.score[1] += 1
            else:
                self.faults[1] += 1
                self.start_side = 1
                if self.faults[1] % 3 == 0:
                    self.score[0] += 1

            return True

        # Puck in Goal
        if (np.abs(puck_pos[1]) - self.env_info['table']['goal_width'] / 2) <= 0:
            if puck_pos[0] > self.env_info['table']['length'] / 2:
                self.score[0] += 1
                self.start_side = -1
                return True

            if puck_pos[0] < -self.env_info['table']['length'] / 2:
                self.score[1] += 1
                self.start_side = 1
                return True

        # Puck stuck in the middle
        if np.abs(puck_pos[0]) < 0.15 and np.linalg.norm(puck_vel[0]) < 0.025:
            return True
        return super(AirHockeyTournament, self).is_absorbing(obs)


if __name__ == '__main__':
    env = AirHockeyTournament()
    env.reset()

    steps = 0
    while True:
        action = np.zeros(14)
        steps += 1
        observation, reward, done, info = env.step(action)
        env.render()
        if done or steps > env.info.horizon:
            steps = 0
            env.reset()
