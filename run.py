"""
Script that will evaluate the performance of an agent on the specified environments. The agent is imported from
air_hockey_agent/agent_builder.py, where it should be implemented.

Args:
    -c --config path/to/config.yml:  Specify the path to the config.yml that should be used. The default config is in
            air_hockey_agent/agent_config.yml. The config specifies all parameters for the evaluation. For convenience
            the environment_list, number_of_experiments and log_dir can be overriden with the args below
    -e --env "3dof":    Overrides the env parameter of the config. Specifies the environments
            which are evaluated. Multiple environments can be provided. All possible envs are: ["3dof-hit",
             "3dof-defend", "7dof-hit", "7dof-defend", "7dof-prepare", "3dof", "7dof"] where "3dof" and "7dof"
             are shortcuts
    -r --render: Set the flag to spawn a viewer which renders the simulation, Overrides the render param of the config
    -g --generate_score phase-1: Set to phase-1 or phase-2 to generate a score for the leaderboard ranking for phase one
            or two. Note that for the generation of the phase-1 score the 3dof-hit, 3dof-defend envs are required. For
            the phase 2 all the 7dof envs are required."
    -n --n_episodes 50: specify the number of episodes used for evaluation
    --n_cores 4: specify the amount of CPU cores used for evaluation. Set to -1 to use all available cores.
    --log_dir: Specify the path to the log directory, Overrides the log_dir of the config
    --example baseline: Load the provided example agents. Can either be "baseline" for the traditional robotics
            solution or "sac" for an end-to-end trained solution with SAC.

Examples:
    To view the baseline Agent on the 3dof-hit environment:
    python run.py --example baseline -e 3dof-hit -n 1 -r

    To generate a report for the first phase:
    python run.py -e 3dof -g phase-1

    Of course all these parameters can also just be set in a config file which is loaded by via the
     --config /path/to/conf.yml. Or just modify the default config at air_hockey_agent/agent_config.yml

"""

import os
from argparse import ArgumentParser
from pathlib import Path

import yaml

from air_hockey_challenge.framework.evaluate_agent import evaluate
from air_hockey_challenge.framework.evaluate_tournament import run_tournament
from air_hockey_challenge.utils.tournament_agent_server import run_tournament_server


def convert_envs(env_list):
    if "3dof" in env_list:
        env_list.remove("3dof")
        env_list.extend(["3dof-hit", "3dof-defend"])

    if "7dof" in env_list:
        env_list.remove("7dof")
        env_list.extend(["7dof-hit", "7dof-defend", "7dof-prepare"])

    return env_list


def get_args():
    parser = ArgumentParser()
    arg_test = parser.add_argument_group('override parameters')

    env_choices = ["3dof-hit", "3dof-defend", "7dof-hit", "7dof-defend", "7dof-prepare", "3dof", "7dof", "tournament",
                   "tournament_server"]

    arg_test.add_argument("-e", "--env", nargs='*',
                          choices=env_choices,
                          help='Environments to be used. Use \'all\' to select all the available environments.')

    arg_test.add_argument("--n_cores", type=int, help="Number of CPU cores used for evaluation.")

    arg_test.add_argument("-n", "--n_episodes", type=int,
                          help="Each seed will run for this number of Episodes.")

    arg_test.add_argument("--log_dir", type=str,
                          help="The directory in which the logs are written")

    arg_test.add_argument("--example", type=str, choices=["hit-agent", "defend-agent", "baseline", "atacom"], default="")

    default_path = Path(__file__).parent.joinpath("air_hockey_agent/agent_config.yml")
    arg_test.add_argument("-c", "--config", type=str, default=default_path,
                          help="Path to the config file.")

    arg_test.add_argument("-r", "--render", action='store_true', help="If set renders the environment")

    arg_test.add_argument("-g", "--generate_score", type=str, choices=["phase-1", "phase-2"],
                          help="Set to phase-1 or phase-2 to generate a report for phase one or two. Note that for the "
                               "generation of the phase-1 report the 3dof-hit, 3dof-defend envs "
                               "are required. For the phase 2 all the 7dof envs are required.")

    arg_test.add_argument("--host", type=str, help="Host IP for tournament agent server")
    arg_test.add_argument("--port", type=int, help="Host port for tournament agent server")

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = get_args()

    # Remove all None entries
    filtered_args = {k: v for k, v in args.items() if v is not None}

    # Load config
    if os.path.exists(filtered_args["config"]):
        with open(filtered_args["config"]) as stream:
            config = yaml.safe_load(stream)
    else:
        print("Could not read config file with path: ", filtered_args["config"])
        config = {"quiet": False, "render": False}
    del filtered_args["config"]

    # Load Agent
    if filtered_args["example"] == "":
        from air_hockey_agent.agent_builder import build_agent
    elif filtered_args["example"] == "hit-agent":
        from examples.control.hitting_agent import build_agent
    elif filtered_args["example"] == "defend-agent":
        from examples.control.defending_agent import build_agent
    elif filtered_args["example"] == "baseline":
        from baseline.baseline_agent.baseline_agent import build_agent
    elif filtered_args["example"] == "atacom":
        from examples.rl.agent_loader import build_agent
    del filtered_args["example"]

    # Update config with command line args
    config.update(filtered_args)
    config["env_list"] = convert_envs(config["env"])
    del config["env"]

    if "tournament" in config["env_list"]:
        run_tournament(build_agent, **config)
    elif "tournament_server" in config["env_list"]:
        run_tournament_server(build_agent, **config)
    else:
        evaluate(build_agent, **config)
