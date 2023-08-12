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

    check_score = True

    if mdp.base_env.n_agents == 2:
        # Remove the second puck obs idx so the correctly oriented one is copied
        mujoco_idx = np.delete(mujoco_idx, [10, 11, 12])
        pos_idx = np.delete(pos_idx, [10, 11, 12])


    if env_name == "7dof-hit":
        # Only get obs for first agent because second one does not exist
        true_idx = np.where(pos_idx < mdp.env_info['robot']["n_joints"] * 2 + 6)
        pos_idx = pos_idx[true_idx]


    for step in dataset:
        # Set puck back to unnormalized position
        mdp.base_env.obs_helper.get_from_obs(step[3], "puck_x_pos")[:] -= 1.51

        if env_name == "7dof-hit":
            mdp.base_env._data.qpos[mujoco_idx] = np.concatenate([step[3][pos_idx], mdp.base_env._opponent_agent(None)[0]])
        else:
            mdp.base_env._data.qpos[mujoco_idx] = step[3][pos_idx]
        
        mdp.base_env._data.qvel[mujoco_idx] = 0

        if "7dof" in env_name or env_name == "tournament":
            mdp.base_env.universal_joint_plugin.update()
            mdp.base_env._data.qpos[mdp.base_env.universal_joint_plugin.universal_joint_ids] = mdp.base_env.universal_joint_plugin.u_joint_pos_des
        
        mujoco.mj_fwdPosition(mdp.base_env._model, mdp.base_env._data)

        if env_name == "tournament":
            # Check for scoring, skip one check if before is true to avoid double counting faults
            if check_score:
                if mdp.base_env.is_absorbing(step[3]):
                    check_score = False
            else:
                check_score = True

        mdp.render()

        if step[5]:
            mdp.reset()


if __name__ == "__main__":
    replay_dataset("7dof-hit", "path/to/dataset.pkl")
