from air_hockey_challenge.framework.evaluate_tournament import run_remote_tournament

from pathlib import Path
import yaml
import docker
import json


def run_docker_tournament():
    team_info_path = Path(__file__).parent.parent.joinpath("air_hockey_agent/team_info.yml")

    with open(team_info_path) as stream:
        team_info = yaml.safe_load(stream)

    client = docker.from_env()

    name = convert_name(team_info["team_name"])

    resp = client.api.build(path=str(Path(__file__).parent.parent), target="eval", quiet=False,
                     tag=f"swr.eu-west-101.myhuaweicloud.eu/his_air_hockey_competition_eu/solution-{name}")

    for chunck in resp:
        for line in chunck.split(b'\r\n'):
            if line != b'':
                temp = line.decode("utf-8")
                data = json.loads(temp)
                if "stream" in data.keys():
                    print(data["stream"].strip().strip("/n"))

    run_remote_tournament(name, name, log_dir="logs", n_episodes=1, steps_per_game=15000, render=True)

def convert_name(name):
    return name.replace(" ", "-").lower()


if __name__ == "__main__":

    run_docker_tournament()
