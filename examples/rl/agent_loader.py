from pathlib import Path

from examples.rl.atacom_agent_wrapper import ATACOMAgent
import torch

def build_agent(env_info, **kwargs):
    base_dir = Path(__file__).parent.joinpath("agents")

    path_dict = {
        "3dof-hit": base_dir.joinpath("3dof-hit-atacom.msh"),
        "3dof-defend": base_dir.joinpath("3dof-defend-atacom.msh"),
        "7dof-hit": base_dir.joinpath("7dof-hit-atacom.msh"),
        "7dof-defend": base_dir.joinpath("7dof-defend-atacom.msh"),
        "7dof-prepare": base_dir.joinpath("7dof-prepare-atacom.msh"),
    }

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(4)

    agent = ATACOMAgent.load_agent(path_dict[env_info["env_name"]], env_info)
    return agent
