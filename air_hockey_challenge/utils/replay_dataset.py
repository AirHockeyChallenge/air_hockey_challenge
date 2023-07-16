import mujoco
import numpy as np

from air_hockey_challenge.framework import AirHockeyChallengeWrapper


def replay_dataset(env_name, dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)

    mdp = AirHockeyChallengeWrapper(env_name, interpolation_order=1)
    mdp.reset()

    # remove idx which belong to opponent
    mujoco_idx = np.array(mdp.base_env.obs_helper.joint_mujoco_idx)
    pos_idx = np.array(mdp.base_env.obs_helper.joint_pos_idx)

    if env_name == "7dof-hit":
        true_idx = np.where(pos_idx < mdp.env_info['robot']["n_joints"] * 2 + 6)
        mujoco_idx = mujoco_idx[true_idx]
        pos_idx = pos_idx[true_idx]
    elif env_name == "tournament":
        # Remove the second puck obs idx so the correctly oriented one is copied
        mujoco_idx = np.delete(mujoco_idx, [10, 11, 12])
        pos_idx = np.delete(pos_idx, [10, 11, 12])

    for step in dataset:
        if env_name == "tournament":
            action = (step[0][pos_idx][3:10], step[0][pos_idx][10:17])
        else:
            action = step[0][pos_idx][3:]

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
