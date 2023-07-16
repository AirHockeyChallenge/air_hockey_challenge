import os

import matplotlib.pyplot as plt
import numpy as np

from air_hockey_challenge.framework.challenge_core import ChallengeCore
from air_hockey_exp import mdp_builder
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length, parse_dataset
from atacom_agent_wrapper import Agent, AgentBase, ATACOMAgent

plt.switch_backend('tkAgg')


def plot_trajectory(unnormalized_dataset, n_joints):
    fig, ax = plt.subplots(3, n_joints)
    for i in range(n_joints):
        ax[0, i].plot(unnormalized_dataset[0][:, i + 6], label="$q$")
        ax[0, i].plot(unnormalized_dataset[1][:, 0, i], label="$q_d$")
        ax[1, i].plot(unnormalized_dataset[0][:, i + 6 + n_joints], label='$dq$')
        ax[1, i].plot(unnormalized_dataset[1][:, 1, i], label='$\dot{q}_d$')
        ax[2, i].plot(unnormalized_dataset[1][:, 2, i], label='$\dot{q}_d$')
    ax[0, 0].legend()
    ax[1, 0].legend()
    plt.show()


def replay():
    log_path = f"./agents"
    env = "7dof-defend"

    mdp = mdp_builder(env, {})
    agent = ATACOMAgent.load_agent(os.path.join(log_path, f'{env}-atacom.msh'), mdp.env_info)

    core = ChallengeCore(agent, mdp, action_idx=[0, 1])

    eval_params = {
        "n_episodes": 1,
        "quiet": False,
        "render": True,
    }

    for i in range(10):
        dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True)
        parsed_dataset = parse_dataset(dataset)

        J = np.mean(compute_J(dataset, core.mdp.info.gamma))
        R = np.mean(compute_J(dataset))

        eps_length = compute_episodes_length(dataset)
        success = 0
        current_idx = 0
        for episode_len in eps_length:
            success += dataset_info["success"][current_idx + episode_len - 1]
            current_idx += episode_len
        success /= len(eps_length)

        c_avg = {key: np.zeros_like(value) for key, value in dataset_info["constraints_value"][0].items()}
        c_max = {key: np.zeros_like(value) for key, value in dataset_info["constraints_value"][0].items()}

        for constraint in dataset_info["constraints_value"]:
            for key, value in constraint.items():
                c_avg[key] += value
                idxs = c_max[key] < value
                c_max[key][idxs] = value[idxs]

        N = len(dataset_info["constraints_value"])
        for key in c_avg.keys():
            c_avg[key] /= N

        print("J: ", J, " R: ", R, "c_max: ", c_max)
        print("Computation Time - MAX: ", np.max(dataset_info['computation_time']), " MEAN: ", np.mean(dataset_info['computation_time']))

        # plot_trajectory(parsed_dataset, mdp.env_info['robot']['n_joints'])


if __name__ == '__main__':
    # import cProfile
    #
    # with cProfile.Profile() as pr:
    #     replay()
    #     pr.print_stats()
    #     pr.dump_stats("stats")

    replay()
