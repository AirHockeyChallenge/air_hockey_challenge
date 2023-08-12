from air_hockey_challenge.framework.evaluate_tournament import run_tournament
from air_hockey_agent.agent_builder import build_agent

from pathlib import Path
import yaml

# Default path for agent_config have to adjust to your own path. Assumes script is in 2023-challenge/file.py
agent_config_1_path = Path(__file__).parent.joinpath("air_hockey_agent/agent_config.yml")

# For example use the same config for both agents
agent_config_2_path = agent_config_1_path

with open(agent_config_1_path) as stream:
    agent_config_1 = yaml.safe_load(stream)

with open(agent_config_2_path) as stream:
    agent_config_2 = yaml.safe_load(stream)

# Let the agent play against itself. To let 2 different agents play against each other replace the argument of
# build_agent_2 and agent_2_kwargs with the proper function and settings for the different agent
run_tournament(build_agent_1=build_agent, build_agent_2=build_agent, agent_2_kwargs=agent_config_2, **agent_config_1)
