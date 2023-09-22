import datetime
import json
import os
import time
import tarfile, io, yaml
from collections import defaultdict
from threading import Thread
from statistics import mode
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from air_hockey_challenge.framework import AirHockeyChallengeWrapper, ChallengeCore
from air_hockey_challenge.utils.tournament_agent_wrapper import SimpleTournamentAgentWrapper, \
    RemoteTournamentAgentWrapper
from baseline.baseline_agent.baseline_agent import BaselineAgent
from mushroom_rl.core import Logger

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "link_constr": 3, "joint_vel_constr": 1,
                  "computation_time_minor": 0.5, "computation_time_middle": 1, "computation_time_major": 2}


def run_tournament(build_agent_1, log_dir, build_agent_2=None, agent_2_kwargs={}, steps_per_game=45000, n_episodes=1,
                   n_cores=-1,
                   seed=None, generate_score=None, quiet=True, render=False, save_away_data=False, **kwargs):
    """
    Run tournament games between two agents which are build by build_agent_1 and build_agent_2. The resulting Dataset
    and constraint stats will be written to folder specified in log_dir. If save_away_data is True the data for the
    second Agent is also saved. The amount of games is specified by n_episodes. The resulting Dataset can be replayed by
    the replay_dataset function in air_hockey_challenge/utils/replay_dataset.py. This function is intended to be called
    by run.py.

    For compatibility with run.py the kwargs for agent_1 are passed via **kwargs and the kwargs for agent_2 are passed
    via agent_2_kwargs.

    Args:
        build_agent_1 ((mdp_info, **kwargs) -> Agent): Function that returns agent_1 given the env_info and **kwargs.
        log_dir (str): The path to the log directory.
        build_agent_2 ((mdp_info, **kwargs) -> Agent, None): Function that returns agent_2 given the env_info and
            **agent_2_kwargs. If None the BaselineAgent is used.
        agent_2_kwargs (dict, {}): The arguments for the second agent.
        steps_per_game (int, 45000): The amount of steps a single game will last
        n_episodes (int, 1): The number of games which are played
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): If set to "phase-3" a score and summary is generated. The achieved score against
            Baseline Agent is what the leaderboard is based on.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        save_away_data(bool, False): Set True to save the data and generate a score for the second agent.
        kwargs (any): Argument passed to the agent_1 init
        """

    def agent_builder(mdp, i, build_agent_1, build_agent_2, agent_1_kwargs, agent_2_kwargs):
        agent_1 = build_agent_1(mdp.env_info, **agent_1_kwargs)
        if build_agent_2 is None:
            agent_2 = BaselineAgent(mdp.env_info, 2)
        else:
            agent_2 = build_agent_2(mdp.env_info, **agent_2_kwargs)

        return SimpleTournamentAgentWrapper(mdp.env_info, agent_1, agent_2)

    interpolation_order = [3, 3]
    if "interpolation_order" in kwargs.keys():
        interpolation_order[0] = kwargs["interpolation_order"]

    if "interpolation_order" in agent_2_kwargs.keys():
        interpolation_order[1] = agent_2_kwargs["interpolation_order"]


    _run_tournament(log_dir, agent_builder, "Home", "Away", steps_per_game, n_episodes, n_cores, seed, generate_score,
                    quiet, render, save_away_data, tuple(interpolation_order), build_agent_1=build_agent_1,
                    build_agent_2=build_agent_2, agent_1_kwargs=kwargs, agent_2_kwargs=agent_2_kwargs)


def run_remote_tournament(name_1, name_2, log_dir, steps_per_game=45000, n_episodes=1, n_cores=-1, seed=None,
                          generate_score=None, quiet=True, render=False, save_away_data=False):
    """
    Run a tournament game between two docker containers with the team names name_1 and name_2. The images have to be
    pulled beforehand.

    Args:
        name_1: Name of the first docker image. Will be appended to the prefix
            "swr.eu-west-101.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-"
        name_2: Name of the second docker image. Will be appended to the prefix
            "swr.eu-west-101.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-"
        log_dir (str): The path to the log directory.

        steps_per_game (int, 45000): The amount of steps a single game will last
        n_episodes (int, 1): The number of games which are played
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): If set to "phase-3" a score and summary is generated. The achieved score against
            Baseline Agent is what the leaderboard is based on.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        save_away_data(bool, False): Set True to save the data and generate a score for the second agent.
    """

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    import docker
    BASE_NAME_IMG = "swr.eu-west-101.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-"

    def agent_builder(mdp, i, docker_name_1, docker_name_2):
        port_1 = 8000 + i * 2
        port_2 = 8000 + i * 2 + 1
        host_1 = "localhost"
        host_2 = "localhost"

        client = docker.from_env()

        # Can add CPU and Memory Cap
        host_config_1 = client.api.create_host_config(port_bindings={port_1: port_1}, auto_remove=True)
        host_config_2 = client.api.create_host_config(port_bindings={port_2: port_2}, auto_remove=True)

        container_id_1 = client.api.create_container(BASE_NAME_IMG + docker_name_1, detach=True,
                                                     # network_disabled=True,
                                                     ports=[port_1],
                                                     host_config=host_config_1,
                                                     command=f"python 2023-challenge/run.py -e tournament_server --host 0.0.0.0 --port {port_1}")

        container_1 = client.containers.get(container_id_1)

        container_1.start()

        container_id_2 = client.api.create_container(BASE_NAME_IMG + docker_name_2, detach=True,
                                                     # network_disabled=True,
                                                     ports=[port_2],
                                                     host_config=host_config_2,
                                                     command=f"python 2023-challenge/run.py -e tournament_server --host 0.0.0.0 --port {port_2}")

        container_2 = client.containers.get(container_id_2)

        container_2.start()

        while container_1.status != "running" or container_2.status != "running":
            time.sleep(1)
            container_1.reload()
            container_2.reload()
        time.sleep(5)

        def log_docker_out(log_dir, container_1, container_2, docker_name_1, docker_name_2):
            while True:
                try:
                    container_1.reload()
                    with open(os.path.join(log_dir, f"out_{docker_name_1}.log"), 'ab') as f:
                        for line in container_1.logs(stream=True, follow=False):
                            f.write(line)

                    container_2.reload()
                    with open(os.path.join(log_dir, f"out_{docker_name_2}.log"), 'ab') as f:
                        for line in container_2.logs(stream=True, follow=False):
                            f.write(line)

                    time.sleep(10)
                except Exception:
                    break

        thread = Thread(target=log_docker_out, daemon=True, args=(log_dir, container_1, container_2, docker_name_1, docker_name_2))
        thread.start()

        return RemoteTournamentAgentWrapper(host_1, port_1, host_2, port_2, container_id_1, container_id_2, name_1,
                                            name_2, log_dir)

    # get interpolation order setting from docker containers
    client = docker.from_env()
    interpolation_order = []
    for name in [name_1, name_2]:
        container = client.containers.create(BASE_NAME_IMG + name)
        bits, stats = container.get_archive("/src/2023-challenge/air_hockey_agent/agent_config.yml")

        bytes = b''.join(bits)

        file_like_obj = io.BytesIO(bytes)
        tar = tarfile.open(fileobj=file_like_obj)

        for member in tar.getmembers():
            f = tar.extractfile(member)
            config = yaml.safe_load(f)
            if "interpolation_order" in config.keys():
                interpolation_order.append(config["interpolation_order"])
            else:
                interpolation_order.append(3)
        container.remove()

    start = time.time()

    _run_tournament(log_dir, agent_builder, name_1, name_2, steps_per_game, n_episodes, n_cores, seed, generate_score,
                    quiet, render, save_away_data, interpolation_order=tuple(interpolation_order), docker_name_1=name_1,
                    docker_name_2=name_2)

    print("EXECUTION TOOK ", time.time() - start)


def _run_tournament(log_dir, agent_builder, name_1, name_2, n_steps=45000, n_episodes=10, n_cores=-1, seed=None,
                    generate_score=None, quiet=True, render=False, save_away_data=False, interpolation_order=(3, 3), **kwargs):
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime('eval-%Y-%m-%d_%H-%M-%S'))

    # score, winner, penalty_sum_1, penalty_sum_2, violations_1, violations_2
    data = Parallel(n_jobs=n_cores)(
        delayed(_run_single_tournament)(log_dir, agent_builder, name_1, name_2, n_steps, quiet, render,
                                        compute_seed(seed, i), i, save_away_data, interpolation_order, **kwargs)
        for i in range(n_episodes))

    if generate_score == "phase-3":
        # score, faults, winner, penalty_sum_1, penalty_sum_2, violations_1, violations_2

        result_1 = {}
        winner = [game[2] for game in data]
        result_1["Winner"] = ["Draw", name_1, name_2][mode(winner)]

        violations_1 = defaultdict(list)

        for game in data:
            for key, value in game[5]["Violation Summary"].items():
                violations_1[key].append(value)

        mean_violations_1 = {k: np.mean(v) for k, v in violations_1.items()}
        result_1["Mean Violations"] = mean_violations_1

        result_1["Games"] = {}
        for i in range(len(data)):
            result_1["Games"][f"Game-{i}"] = {}
            result_1["Games"][f"Game-{i}"]["Winner"] = ["Draw", name_1, name_2][data[i][2]]
            result_1["Games"][f"Game-{i}"]["Final score"] = {name_1: data[i][0][0], name_2: data[i][0][1]}
            result_1["Games"][f"Game-{i}"]["Goals scored"] = {name_1: data[i][0][0] - data[i][1][1] // 3,
                                                              name_2: data[i][0][1] - data[i][1][0] // 3}
            result_1["Games"][f"Game-{i}"]["Faults"] = {name_1: data[i][1][0], name_2: data[i][1][1]}
            result_1["Games"][f"Game-{i}"]["Constraint penalty points"] = data[i][3]
            result_1["Games"][f"Game-{i}"]["Constraint violation summary"] = data[i][5]["Violation Summary"]

        with open(os.path.join(log_dir, f"result_{name_1}.json"), 'w') as f:
            json.dump(result_1, f, indent=4, default=convert)

        if save_away_data:
            result_2 = {}

            result_2["Winner"] = ["Draw", name_1, name_2][mode(winner)]

            violations_2 = defaultdict(list)

            for game in data:
                for key, value in game[6]["Violation Summary"].items():
                    violations_2[key].append(value)

            mean_violations_2 = {k: np.mean(v) for k, v in violations_2.items()}
            result_2["Mean Violations"] = mean_violations_2

            result_2["Games"] = {}
            for i in range(len(data)):
                result_2["Games"][f"Game-{i}"] = {}
                result_2["Games"][f"Game-{i}"]["Winner"] = ["Draw", name_1, name_2][data[i][2]]
                result_2["Games"][f"Game-{i}"]["Final score"] = {name_1: data[i][0][0], name_2: data[i][0][1]}
                result_2["Games"][f"Game-{i}"]["Goals scored"] = {name_1: data[i][0][0] - data[i][1][1] // 3,
                                                                  name_2: data[i][0][1] - data[i][1][0] // 3}
                result_2["Games"][f"Game-{i}"]["Faults"] = {name_1: data[i][1][0], name_2: data[i][1][1]}
                result_2["Games"][f"Game-{i}"]["Constraint penalty points"] = data[i][4]
                result_2["Games"][f"Game-{i}"]["Constraint violations"] = data[i][6]["Violation Summary"]

            with open(os.path.join(log_dir, f"result_{name_2}.json"), 'w') as f:
                json.dump(result_2, f, indent=4, default=convert)


def _run_single_tournament(log_dir, agent_builder, name_1, name_2, n_steps, quiet, render, seed, i, save_away_data,
                           interpolation_order, **kwargs):
    log_dir = os.path.join(log_dir, f"Game_{i}")

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    eval_params = {"quiet": quiet, "render": render, "n_steps": n_steps}

    mdp = AirHockeyChallengeWrapper("tournament", interpolation_order=interpolation_order)

    agent = agent_builder(mdp, i, **kwargs)

    core = ChallengeCore(agent, mdp, is_tournament=True, init_state=mdp.base_env.init_state,
                         time_limit=0.02)

    # dataset, score, faults, constraints_dict_1, constraints_dict_2, jerk[:, 0], jerk[:, 1], computation_time[:, 0], \
    #            computation_time[:, 1], penalty_sum_1, penalty_sum_2, violations_1, violations_2
    dataset, score, faults, constraints_dict_1, constraints_dict_2, jerk_1, jerk_2, computation_time_1, computation_time_2, \
    penalty_sum_1, penalty_sum_2, violations_1, violations_2 = compute_metrics(core, eval_params)

    n_eps = n_steps / 500
    winner = 0  # Draw
    if score[0] > score[1] and penalty_sum_1 <= 1.5 * n_eps:
        winner = 1
    if score[0] < score[1] and penalty_sum_2 <= 1.5 * n_eps:
        winner = 2

    logger = Logger(log_name=name_1, results_dir=log_dir)

    logger.log_dataset(dataset)
    logger.log_numpy_array(jerk=jerk_1, computation_time=computation_time_1, **constraints_dict_1)

    with open(os.path.join(log_dir, name_1, "violations.json"), 'w') as f:
        json.dump(violations_1, f, indent=4, default=convert)

    if save_away_data:
        logger = Logger(log_name=name_2, results_dir=log_dir)

        logger.log_dataset(dataset)
        logger.log_numpy_array(jerk=jerk_2, computation_time=computation_time_2, **constraints_dict_2)

        with open(os.path.join(log_dir, name_2, "violations.json"), 'w') as f:
            json.dump(violations_2, f, indent=4, default=convert)

    return score, faults, winner, penalty_sum_1, penalty_sum_2, violations_1, violations_2


def compute_metrics(core, eval_params):
    dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True)
    score = dataset_info["score"][-1]
    faults = dataset_info["faults"][-1]

    constraints_dict_1 = {}
    constraints_dict_2 = {}

    for key in dataset_info["constraints_value"][0][0]:
        constraints_dict_1[key] = []
        constraints_dict_2[key] = []
        for dict in dataset_info["constraints_value"]:
            constraints_dict_1[key].append(dict[0][key])
            constraints_dict_2[key].append(dict[1][key])

    jerk = np.array(dataset_info["jerk"])
    computation_time = np.array(dataset_info["computation_time"])

    n_steps = len(dataset)
    episode_size = 500

    penalty_sum_1, violations_1 = get_violations(n_steps, episode_size, constraints_dict_1, computation_time[:, 0])

    penalty_sum_2, violations_2 = get_violations(n_steps, episode_size, constraints_dict_2, computation_time[:, 1])

    return dataset, score, faults, constraints_dict_1, constraints_dict_2, jerk[:, 0], jerk[:, 1], computation_time[:, 0], \
           computation_time[:, 1], penalty_sum_1, penalty_sum_2, violations_1, violations_2


def get_violations(n_steps, episode_size, constraints_dict, computation_time):
    penalty_sum = 0
    violations = defaultdict(list)

    for i in range(0, n_steps, episode_size):
        for name in constraints_dict.keys():
            if np.any(np.array(constraints_dict[name][i: i + episode_size]) > 0):
                penalty_sum += PENALTY_POINTS[name]
                violations["Episode " + str(i / episode_size)].append(name + " violated")

        max_time_violations = np.max(computation_time[i: i + episode_size])
        mean_time_violations = np.mean(computation_time[i: i + episode_size])
        if max_time_violations > 0.2 or mean_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_major"]
            if max_time_violations > 0.2:
                violations["Episode " + str(i / episode_size)].append("max computation_time > 0.2s")
            else:
                violations["Episode " + str(i / episode_size)].append("mean computation_time > 0.02s")

        elif max_time_violations > 0.1:
            penalty_sum += PENALTY_POINTS["computation_time_middle"]
            violations["Episode " + str(i / episode_size)].append("max computation_time > 0.1s")

        elif max_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_minor"]
            violations["Episode " + str(i / episode_size)].append("max computation_time > 0.02s")

    # Compute violation stats
    violation_stats = defaultdict(int)
    for eps in violations.values():
        for el in eps:
            if "computation" in el:
                violation_stats["Computation Time"] += 1

            for name in constraints_dict.keys():
                if name in el:
                    violation_stats[name] += 1

    violation_stats["Total"] = sum(violation_stats.values())

    violation_stats["Penalty Points"] = penalty_sum

    violations["Violation Summary"] = violation_stats

    return penalty_sum, violations


def compute_seed(seed, i):
    if seed is not None:
        return seed + i
    return None


def convert(o):
    if isinstance(o, np.int_): return int(o)
    return o


if __name__ == "__main__":
    run_remote_tournament("agent1", "agent1", "logs", n_episodes=1, n_cores=1, render=True, quiet=False)
