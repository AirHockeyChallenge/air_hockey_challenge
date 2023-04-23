
import numpy as np
import mujoco

from air_hockey_challenge.environments.planar import AirHockeyBase


#define our own class
#author: An Dang


class AirHockeyDouble(AirHockeyBase):
    def __init__(self, gamma=0.99, horizon=500, viewer_params={}):
        '''
        Constructor
        '''
        #r0 refers to robot0
        #r1 refers to robot1
        self.r0_init_state = np.array([-1.15570723,  1.30024401,  1.44280414])
        self.r1_init_state = np.array([-1.15570723,  1.30024401,  1.44280414])

        #call init from AirHockeyBase
        super().__init__(gamma=gamma, horizon=horizon, n_agents=2, viewer_params=viewer_params)
        self.filter_ratio = 0.274
        self.r0_q_pos_prev = np.zeros(self.env_info["robot"]["n_joints"])
        self.r0_q_vel_prev = np.zeros(self.env_info["robot"]["n_joints"])

        self.r1_q_pos_prev = np.zeros(self.env_info["robot"]["n_joints"])
        self.r1_q_vel_prev = np.zeros(self.env_info["robot"]["n_joints"])

    def get_ee(self):
        '''
            returns based on _read_data same as single.py get_ee
        '''
        r0_ee_pos = self._read_data("robot_1/ee_pos")
        r0_ee_vel = self._read_data("robot_1/ee_vel")
        r1_ee_pos = self._read_data("robot_2/ee_pos")
        r1_ee_vel = self._read_data("robot_2/ee_vel")

        return r0_ee_pos, r0_ee_vel, r1_ee_pos, r1_ee_vel
    


def main():
    import time
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    from air_hockey_challenge.framework.agent_base import DoubleAgentsWrapper
    np.random.seed(0)


    # env = AirHockeyChallengeWrapper(env="3dof-hit-opponent", action_type="position-velocity",
    #                                 interpolation_order=3, debug=False)
    env = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity",
                                    interpolation_order=3, debug=False)

    agent1 = BaselineAgent(env.env_info, agent_id=1, only_tactic="hit")
    agent2 = BaselineAgent(env.env_info, agent_id=2, only_tactic="hit")

    agents = DoubleAgentsWrapper(env.env_info, agent1, agent2)

    obs = env.reset()
    agents.episode_start()

    steps = 0
    while True:
        steps += 1
        t_start = time.time()
        action = agents.draw_action(obs)
        # print("time: ", time.time() - t_start)
        obs, reward, done, info = env.step(action)

        env.render()

        if done or steps > env.info.horizon:
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use("tkAgg")
            # trajectory_record = np.array(env.base_env.controller_record)
            # nq = env.base_env.env_info['robot']['n_joints']
            #
            # fig, axes = plt.subplots(3, nq)
            # for j in range(nq):
            #     axes[0, j].plot(trajectory_record[:, j])
            #     axes[0, j].plot(trajectory_record[:, j + nq])
            #     axes[1, j].plot(trajectory_record[:, j + 2 * nq])
            #     axes[1, j].plot(trajectory_record[:, j + 3 * nq])
            #     # axes[2, j].plot(trajectory_record[:, j + 4 * nq])
            #     axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
            # plt.show()

            steps = 0
            obs = env.reset()
            agents.episode_start()
            print("Reset")


if __name__ == '__main__':
    main()
