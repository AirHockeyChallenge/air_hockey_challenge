import mujoco
import numpy as np
from tqdm import tqdm

from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from mushroom_rl.utils.record import VideoRecorder


def replay_dataset(env_name, dataset_path, agent_name="Agent", opponent_name="Opponent", viewer_params={},
                   record=False, record_dir="./recording", max_episodes=1000):
    dataset = np.load(dataset_path, allow_pickle=True)
    if record:
        recorder = VideoRecorder(path=record_dir, tag='video', fps=50)

    mdp = AirHockeyChallengeWrapper(env_name, interpolation_order=1, agent_name=agent_name, opponent_name=opponent_name,
                                    viewer_params=viewer_params)
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

    n_episodes = 0
    for step in tqdm(dataset):
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

        frame = mdp.render(record=record)
        if record:
            recorder(frame)

        if step[5]:
            mdp.reset()
            n_episodes += 1

        if n_episodes >= max_episodes:
            break

    mdp.base_env.stop()
    if record:
        recorder.stop()


if __name__ == "__main__":
    import os
    viewer_params = {
        'camera_params': {
            'static': dict(distance=3.0, elevation=-45.0, azimuth=90.0,
                           lookat=(0., 0., 0.))},
        'default_camera_mode': 'static',
        'hide_menu_on_startup': True,
    }

    data_path = f"/home/puze/AirHockeyChallenge/Dataset/PostEvaluations/logsFixSuccess/full"

    env_list = {
        # '7dof-hit': ['airlihockey', 'air-hockit', 'gxu-lipe'],
        #         '7dof-defend': ['airlihockey', 'air-hockit', 'rl3_polimi'],
                '7dof-prepare': ['airlihockey', 'air-hockit', 'aerotron']}
    for env in env_list.keys():
        for team in env_list[env]:
            replay_dataset(env, dataset_path=os.path.join(data_path, team, env, "dataset.pkl"),
                           record=True, viewer_params=viewer_params, max_episodes=20,
                           record_dir=os.path.join(data_path, team, env))
