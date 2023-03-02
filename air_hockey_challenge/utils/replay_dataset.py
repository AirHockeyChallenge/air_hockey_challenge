import mujoco
import numpy as np
from air_hockey_challenge.framework import AirHockeyChallengeWrapper


def replay_dataset(env_name, dataset_path):
    """Replay the Dataset

    Args
    ----
    env_name: str
        The name of the environment
    dataset_path: str
        The path of the dataset to replay

    """
    dataset = np.load(dataset_path, allow_pickle=True)

    # Do not change action_type, it does not have to match the recorded dataset
    mdp = AirHockeyChallengeWrapper(env_name, action_type="acceleration")
    if mdp.base_env.n_agents == 1:
        mdp.reset()
        for step in dataset:
            mdp.base_env._data.qpos[mdp.base_env.obs_helper.joint_mujoco_idx] = step[0][mdp.base_env.obs_helper.joint_pos_idx]
            # Adjust puck back to table frame
            mdp.base_env._data.joint("puck_x").qpos -= 1.51
            mujoco.mj_fwdPosition(mdp.base_env._model, mdp.base_env._data)
            mdp.render()

    else:
        # Assume it's hit opponent env, replay obs of agent and puck, simulate opponent because we don't have
        # his movement in the observations
        mdp.reset()

        # remove idx which belong to opponent
        mujoco_idx = np.array(mdp.base_env.obs_helper.joint_mujoco_idx)
        pos_idx = np.array(mdp.base_env.obs_helper.joint_pos_idx)
        true_idx = np.where(pos_idx < mdp.env_info['robot']["n_joints"] * 2 + 6)
        mujoco_idx = mujoco_idx[true_idx]
        pos_idx = pos_idx[true_idx]

        for step in dataset:

            action = np.zeros(mdp.env_info['robot']["n_joints"])
            mdp.step(action)
            mdp.base_env._data.qpos[mujoco_idx] = step[0][pos_idx]
            mdp.base_env._data.joint("puck_x").qpos -= 1.51

            mujoco.mj_fwdPosition(mdp.base_env._model, mdp.base_env._data)

            mdp.render()

            if step[5]:
                mdp.reset()


if __name__ == "__main__":
    replay_dataset("3dof-hit", "../../logs/eval-2023-02-08_10-41-11/3dof-hit/dataset-3dof-hit.pkl")
