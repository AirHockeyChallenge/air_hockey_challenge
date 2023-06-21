import datetime
import gc
import json
import os
from collections import defaultdict

import numpy as np
import torch
from joblib import Parallel, delayed

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore
from mushroom_rl.core import Logger
from mushroom_rl.utils.dataset import compute_episodes_length

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "link_constr": 3, "joint_vel_constr": 1,
                  "computation_time_minor": 0.5, "computation_time_middle": 1, "computation_time_major": 2}


def evaluate(agent_builder, log_dir, env_list, n_episodes=1080, n_cores=-1, seed=None, generate_score=None,
             quiet=True, render=False, interpolation_order=3, **kwargs):
    """
    Function that will run the evaluation of the agent for a given set of environments. The resulting Dataset and
    constraint stats will be written to folder specified in log_dir. The resulting Dataset can be replayed by the
    replay_dataset function in air_hockey_challenge/utils/replay_dataset.py. This function is intended to be called
    by run.py.

    Args:
        agent_builder ((mdp_info, **kwargs) -> Agent): Function that returns an agent given the env_info and **kwargs.
        log_dir (str): The path to the log directory
        env_list (list): List of environments, on which the agent is tested
        n_episodes (int, 1000): Number of episodes each environment is evaluated
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): Set to "phase-1" or "phase-2" to generate a report for the first or second phase.
            The report shows the performance stats which will be used in the scoring of the agent. Note that the
            env_list has to include ["3dof-hit-opponent", "3dof-defend"] to generate the phase-1 report and
            ["7dof-hit", "7dof-defend", "7dof-prepare"] for the phase-2 report.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        kwargs (any): Argument passed to the Agent init
    """

    path = os.path.join(log_dir, datetime.datetime.now().strftime('eval-%Y-%m-%d_%H-%M-%S'))

    if n_cores == -1:
        n_cores = os.cpu_count()

    assert n_cores != 0
    assert n_episodes >= n_cores

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    results = {}
    env_init_chuncks = []

    # Precompute all initial states for all experiments
    for env in env_list:
        env_init_chuncks.append(generate_init_states(env, n_episodes, n_cores))

    summary = "=================================================\n"
    for env, chunks in zip(env_list, env_init_chuncks):

        # returns: dataset, success, penalty_sum, constraints_dict, computation_time, violations, metric_dict
        data = Parallel(n_jobs=n_cores)(delayed(_evaluate)(path, env, agent_builder, chunks[i], quiet, render,
                                                           sum([len(x) for x in chunks[:i]]), compute_seed(seed, i), i,
                                                           interpolation_order, **kwargs) for i in range(n_cores))

        logger = Logger(log_name=env, results_dir=path)

        success = np.mean([d[0] for d in data])
        penalty_sum = np.sum([d[1] for d in data])
        violations = data[0][2]
        for d in data[1:]:
            violations.update(d[2])

        constraint_names = data[0][3]
        n_steps = [d[4] for d in data]

        # Compute violation stats
        violation_stats = defaultdict(int)
        for eps in violations.values():
            for el in eps:
                if "computation" in el:
                    violation_stats["Computation Time"] += 1

                for name in constraint_names:
                    if name in el:
                        violation_stats[name] += 1

        violation_stats["Total"] = sum(violation_stats.values())

        violations["violation summary"] = violation_stats

        with open(os.path.join(logger.path, "violations.json"), 'w') as f:
            json.dump(violations, f, indent=4)

        del violations

        # Concat the data saved by the workers. One by one so its memory efficient
        computation_time = np.zeros(sum(n_steps))
        for i in range(n_cores):
            computation_time[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                os.path.join(logger.path, f"computation_time-{i}.npy"))
            os.remove(os.path.join(logger.path, f"computation_time-{i}.npy"))

        logger.log_numpy_array(computation_time=computation_time)
        del computation_time

        for name in constraint_names:
            const_0 = np.load(os.path.join(logger.path, f"{name}-0.npy"))
            os.remove(os.path.join(logger.path, f"{name}-0.npy"))
            const = np.zeros((sum(n_steps), const_0.shape[1]))
            const[:n_steps[0]] = const_0
            for i in range(1, n_cores):
                const[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                    os.path.join(logger.path, f"{name}-{i}.npy"))
                os.remove(os.path.join(logger.path, f"{name}-{i}.npy"))

            logger.log_numpy_array(**{name: const})

        del const
        del const_0

        dataset = []
        for i in range(n_cores):
            dataset.extend(np.load(os.path.join(logger.path, f"dataset-{i}.pkl"), allow_pickle=True))
            os.remove(os.path.join(logger.path, f"dataset-{i}.pkl"))

        logger.log_dataset(dataset)

        del dataset

        gc.collect()

        color = 2
        if penalty_sum <= 500:
            color = 0
        elif penalty_sum <= 1500:
            color = 1

        results[env] = {"Success": success, "Penalty": penalty_sum, "Color": color}

        summary += "Environment: ".ljust(20) + f"{env}\n" + "Number of Episodes: ".ljust(20) + f"{n_episodes}\n"
        summary += "Success: ".ljust(20) + f"{success:.4f}\n"
        summary += "Penalty: ".ljust(20) + f"{penalty_sum}\n"
        summary += "Number of Violations: \n"
        for key, value in violation_stats.items():
            summary += f"  {key}".ljust(20) + f"{value}\n"
        summary += "-------------------------------------------------\n"

    print(summary)
    if generate_score:
        weights = defaultdict(float)
        if generate_score == "phase-1":
            weights.update({"3dof-hit": 0.5, "3dof-defend": 0.5})

        elif generate_score == "phase-2":
            weights.update({"7dof-hit": 0.4, "7dof-defend": 0.4, "7dof-prepare": 0.2})

        max_penalty_env = env_list[np.argmax([results[env]["Penalty"] for env in env_list])]
        score = {"Score": sum([results[env]["Success"] * weights[env] for env in env_list]),
                 "Max Penalty": results[max_penalty_env]["Penalty"],
                 "Color": results[max_penalty_env]["Color"]}

        results["Total Score"] = score

        with open(os.path.join(path, "results.json"), 'w') as f:
            json.dump(results, f, indent=4, default=convert)

        os.system("chmod -R 777 {}".format(log_dir))


def _evaluate(log_dir, env, agent_builder, init_states, quiet, render, episode_offset, seed, i, interpolation_order, **kwargs):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    eval_params = {"quiet": quiet, "render": render, "initial_states": init_states}

    mdp = AirHockeyChallengeWrapper(env, interpolation_order=interpolation_order)

    agent = agent_builder(mdp.env_info, **kwargs)
    core = ChallengeCore(agent, mdp, is_tournament=False, init_state=mdp.base_env.init_state)

    dataset, success, penalty_sum, constraints_dict, computation_time, violations = compute_metrics(core, eval_params,
                                                                                                    episode_offset)

    logger = Logger(log_name=env, results_dir=log_dir, seed=i)

    logger.log_dataset(dataset)
    logger.log_numpy_array(computation_time=computation_time, **constraints_dict)

    n_steps = len(computation_time)

    return success, penalty_sum, violations, constraints_dict.keys(), n_steps


def compute_metrics(core, eval_params, episode_offset):
    dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True)

    episode_length = compute_episodes_length(dataset)
    n_episodes = len(episode_length)

    # Turn list of dicts to dict of lists
    constraints_dict = {k: [dic[k] for dic in dataset_info["constraints_value"]] for k in
                        dataset_info["constraints_value"][0]}
    success = 0
    penalty_sum = 0
    current_idx = 0
    current_eps = episode_offset
    violations = defaultdict(list)

    # Iterate over episodes
    for episode_len in episode_length:
        # Only check last step of episode for success
        success += dataset_info["success"][current_idx + episode_len - 1]

        for name in constraints_dict.keys():
            if np.any(np.array(constraints_dict[name][current_idx: current_idx + episode_len]) > 0):
                penalty_sum += PENALTY_POINTS[name]
                violations["Episode " + str(current_eps)].append(name + " violated")

        max_time_violations = np.max(dataset_info["computation_time"][current_idx: current_idx + episode_len])
        mean_time_violations = np.mean(dataset_info["computation_time"][current_idx: current_idx + episode_len])
        if max_time_violations > 0.2 or mean_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_major"]
            if max_time_violations > 0.2:
                violations["Episode " + str(current_eps)].append("max computation_time > 0.2s")
            else:
                violations["Episode " + str(current_eps)].append("mean computation_time > 0.02s")

        elif max_time_violations > 0.1:
            penalty_sum += PENALTY_POINTS["computation_time_middle"]
            violations["Episode " + str(current_eps)].append("max computation_time > 0.1s")

        elif max_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_minor"]
            violations["Episode " + str(current_eps)].append("max computation_time > 0.02s")

        current_idx += episode_len
        current_eps += 1

    success = success / n_episodes

    return dataset, success, penalty_sum, constraints_dict, dataset_info["computation_time"], violations


def compute_seed(seed, i):
    if seed is not None:
        return seed + i
    return None


def convert(o):
    if isinstance(o, np.int_): return int(o)
    return o


def generate_init_states(env, n_episodes, n_parallel_cores):
    mdp = AirHockeyChallengeWrapper(env)
    init_states = []
    chunk_lens = [n_episodes // n_parallel_cores + int(x < n_episodes % n_parallel_cores) for x in
                  range(n_parallel_cores)]

    for chunk_len in chunk_lens:
        mdp.reset()
        init_states_chunk = np.zeros((chunk_len, mdp.base_env._obs.shape[0]))
        for i in range(chunk_len):
            mdp.reset()
            init_states_chunk[i] = mdp.base_env._obs.copy()
        init_states.append(init_states_chunk)
    return init_states
