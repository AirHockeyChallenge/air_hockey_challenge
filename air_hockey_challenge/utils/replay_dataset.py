import mujoco
import numpy as np

from air_hockey_challenge.framework import AirHockeyChallengeWrapper


def replay_dataset(env_name, dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)

    mdp = AirHockeyChallengeWrapper(env_name, interpolation_order=1)
    if env_name != "7dof-hit":
        mdp.reset()

        mujoco_idx = mdp.base_env.obs_helper.joint_mujoco_idx.copy()
        obs_idx = mdp.base_env.obs_helper.joint_pos_idx.copy()

        if env_name == "tournament":
            # Remove the second puck obs idx so the correctly oriented one is copied
            del mujoco_idx[10:13]
            del obs_idx[10:13]

        for step in dataset:
            mdp.base_env._data.qpos[mujoco_idx] = step[0][obs_idx]
            # Adjust puck back to table frame
            mdp.base_env._data.joint("puck_x").qpos -= 1.51
            mujoco.mj_fwdPosition(mdp.base_env._model, mdp.base_env._data)
            mdp.render()

    else:
        # replay obs of agent and puck, simulate opponent because we don't have
        # his movement in the observations
        mdp.reset()

        # remove idx which belong to opponent
        mujoco_idx = np.array(mdp.base_env.obs_helper.joint_mujoco_idx)
        pos_idx = np.array(mdp.base_env.obs_helper.joint_pos_idx)
        true_idx = np.where(pos_idx < mdp.env_info['robot']["n_joints"] * 2 + 6)
        mujoco_idx = mujoco_idx[true_idx]
        pos_idx = pos_idx[true_idx]

        for step in dataset:
            action = mdp.base_env.init_state
            mdp.step(action)
            mdp.base_env._data.qpos[mujoco_idx] = step[0][pos_idx]
            mdp.base_env._data.qvel[mujoco_idx] = 0

            mdp.base_env._data.joint("puck_x").qpos -= 1.51

            mujoco.mj_fwdPosition(mdp.base_env._model, mdp.base_env._data)

            mdp.render()

            if step[5]:
                mdp.reset()


if __name__ == "__main__":
    replay_dataset("7dof-hit", "path/to/dataset.pkl")
